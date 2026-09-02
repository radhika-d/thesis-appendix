"""
BATCH PLAYA EXTRACTION FOR ALL SENTINEL-2 IMAGES
Extracts pure salt flats with holes preserved as interior rings (donut shapes)
and merges all into a single GeoJSON
"""

import glob
import os
import geopandas as gpd
import numpy as np
import rasterio
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from skimage import measure, morphology
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline on a new set of Sentinel-2 images. Update paths/
# settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# Sentinel-2 GeoTIFFs to process (glob pattern matched against this folder;
# same input folder used by the crest extraction script)
INPUT_TIF_FOLDER = "data/raw/tif"
INPUT_TIF_GLOB    = os.path.join(INPUT_TIF_FOLDER, "sossusvlei_*.tif")

# ---- 2. PARAMETERS ----------------------------------------------------------
SI1_PERCENTILE        = 97   # salinity index (SI-1) threshold percentile
MIN_SIZE_PIXELS       = 100  # remove connected components smaller than this (pixels)
MIN_AREA_M2           = 1000 # discard final polygons smaller than this (m²)
CHAIKIN_ITERATIONS    = 1    # corner-cutting smoothing passes
SIMPLIFY_TOLERANCE_M  = 1    # polygon simplification tolerance (m)

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = "data/processed"
OUTPUT_MERGED_PLAYA = os.path.join(OUTPUT_FOLDER, "merged_playa.geojson")

# ---------------------------------------------------------------------------
# 1. Binary mask
# ---------------------------------------------------------------------------
def extract_playa_mask(
    image_path: str,
    si1_percentile: float = SI1_PERCENTILE,
    min_size_pixels: int = MIN_SIZE_PIXELS,
    closing_radius: int = 3,
    opening_radius: int = 2,
):
    """Return (binary_mask, transform, crs) for pure salt-flat pixels."""
    with rasterio.open(image_path) as src:
        blue = src.read(1).astype(np.float32)
        red = src.read(3).astype(np.float32)
        transform = src.transform
        crs = src.crs

    si1 = np.sqrt(blue * red)
    mask = (si1 > np.percentile(si1, si1_percentile))
    mask = morphology.remove_small_objects(mask, max_size=min_size_pixels)
    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))
    if opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(opening_radius))

    return mask, transform, crs


# ---------------------------------------------------------------------------
# 2. Smoothing — Chaikin corner-cutting
# ---------------------------------------------------------------------------
def chaikin_smooth(coords: np.ndarray, iterations: int = CHAIKIN_ITERATIONS) -> np.ndarray:
    """Chaikin corner-cutting: ring-aware (wraps)."""
    if len(coords) < 4 or iterations == 0:
        return coords
    pts = np.asarray(coords, dtype=np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
        reclose = True
    else:
        reclose = False
    for _ in range(iterations):
        n = len(pts)
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, axis=0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, axis=0)
        pts = np.empty((2 * n, 2), dtype=np.float64)
        pts[0::2] = q
        pts[1::2] = r
    if reclose:
        pts = np.vstack([pts, pts[0]])
    return pts


def _anti_alias_contour(contour: np.ndarray, approx_tolerance: float = 0.5) -> np.ndarray:
    """Remove staircase artefacts from cv2 contours before smoothing."""
    epsilon = approx_tolerance
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx


# ---------------------------------------------------------------------------
# 3. Contour -> world-space coords
# ---------------------------------------------------------------------------
def _contour_to_world(contour: np.ndarray, transform) -> list:
    coords = []
    for pt in contour:
        col, row = pt[0]
        x = transform[2] + col * transform[0]
        y = transform[5] + row * transform[4]
        coords.append((x, y))
    return coords


# ---------------------------------------------------------------------------
# 4. Vectorize polygons with holes (donut shapes)
# ---------------------------------------------------------------------------
def vectorize_playa_donuts(
    mask: np.ndarray,
    transform,
    crs,
    min_area_m2: float = MIN_AREA_M2,
    simplify_tolerance_m: float = SIMPLIFY_TOLERANCE_M,
    chaikin_iterations: int = CHAIKIN_ITERATIONS,
) -> list:
    """Vectorize playa polygons with holes subtracted (donut shapes)."""
    
    labeled, num_features = measure.label(mask, connectivity=2, return_num=True)
    polygons = []

    for i in range(1, num_features + 1):
        single = (labeled == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            single, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            continue

        hier = hierarchy[0]
        holes_by_parent: dict[int, list] = {}
        exterior_list: list[tuple[int, np.ndarray]] = []

        for idx, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            parent_idx = hier[idx][3]
            if parent_idx == -1:
                exterior_list.append((idx, contour))
            else:
                holes_by_parent.setdefault(parent_idx, []).append(contour)

        for ext_idx, ext_contour in exterior_list:
            ext_contour = _anti_alias_contour(ext_contour)
            ext_world = _contour_to_world(ext_contour, transform)
            if len(ext_world) < 3:
                continue
            ext_smooth = chaikin_smooth(np.array(ext_world), chaikin_iterations)
            if not np.allclose(ext_smooth[0], ext_smooth[-1]):
                ext_smooth = np.vstack([ext_smooth, ext_smooth[0]])
            ext_coords = ext_smooth.tolist()

            hole_coords_list = []
            for hole_c in holes_by_parent.get(ext_idx, []):
                hole_c = _anti_alias_contour(hole_c)
                hole_world = _contour_to_world(hole_c, transform)
                if len(hole_world) < 3:
                    continue
                hole_smooth = chaikin_smooth(np.array(hole_world), chaikin_iterations)
                if not np.allclose(hole_smooth[0], hole_smooth[-1]):
                    hole_smooth = np.vstack([hole_smooth, hole_smooth[0]])
                hole_coords_list.append(hole_smooth.tolist())

            try:
                poly = Polygon(ext_coords, hole_coords_list)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue

                poly = poly.simplify(simplify_tolerance_m, preserve_topology=True)

                if poly.geom_type == "MultiPolygon":
                    for part in poly.geoms:
                        if part.area >= min_area_m2:
                            polygons.append(part)
                elif poly.area >= min_area_m2:
                    polygons.append(poly)

            except Exception as exc:
                print(f"    Warning: skipping polygon — {exc}")

    return polygons


# ---------------------------------------------------------------------------
# 5. Main extraction function (returns gdf, doesn't save individually)
# ---------------------------------------------------------------------------
def extract_playa_gdf(image_path: str):
    """Extract playa donut polygons and return as GeoDataFrame."""
    try:
        mask, transform, crs = extract_playa_mask(image_path)
        
        if mask.sum() == 0:
            return None

        polygons = vectorize_playa_donuts(mask, transform, crs)
        
        if not polygons:
            return None

        # Extract acquisition date from filename
        basename = os.path.basename(image_path)
        acquisition_date = basename.replace(".tif", "").split('_')[-1]
        
        rows = []
        for idx, poly in enumerate(polygons):
            rows.append({
                "id": idx,
                "area_m2": poly.area,
                "area_ha": poly.area / 10_000,
                "perimeter_m": poly.length,
                "compactness": (4 * np.pi * poly.area / poly.length ** 2) if poly.length > 0 else 1.0,
                "acquisition_date": acquisition_date,
                "geometry": poly,
            })

        return gpd.GeoDataFrame(rows, crs=crs)

    except Exception as exc:
        print(f"  Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    image_files = sorted(glob.glob(INPUT_TIF_GLOB))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 70)
    print("PLAYA EXTRACTION - HIGH PURITY (HOLES SUBTRACTED)")
    print("=" * 70)
    print(f"  SI-1 percentile   : {SI1_PERCENTILE}")
    print(f"  Min polygon area  : {MIN_AREA_M2} m²")
    print(f"  Chaikin iterations: {CHAIKIN_ITERATIONS}")
    print(f"  Images found      : {len(image_files)}")
    print("=" * 70)

    all_gdfs = []
    ok_count = fail_count = 0

    for idx, path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {os.path.basename(path)}")

        gdf = extract_playa_gdf(path)

        if gdf is not None and not gdf.empty:
            all_gdfs.append(gdf)
            ok_count += 1
            print(f"  Found {len(gdf)} playa polygons")
        else:
            fail_count += 1

    if all_gdfs:
        print("\nMerging all playa polygons...")
        merged = pd.concat(all_gdfs, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=merged.crs)

        merged.to_file(OUTPUT_MERGED_PLAYA, driver='GeoJSON')

        total_area_ha = merged['area_ha'].sum()
        print(f"\nMerged output saved: {OUTPUT_MERGED_PLAYA}")
        print(f"  Total polygons: {len(merged)}")
        print(f"  Total area: {total_area_ha:.2f} ha")
    else:
        print("\nNo playa polygons extracted!")

    print("\n" + "=" * 70)
    print(f"Done — {ok_count} succeeded, {fail_count} failed")
    print(f"Output: {OUTPUT_MERGED_PLAYA}")
    print("=" * 70)