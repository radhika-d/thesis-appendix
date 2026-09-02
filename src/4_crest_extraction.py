"""
BATCH DUNE CREST EXTRACTION FOR ALL SENTINEL-2 IMAGES
Extracts dune crest lines within buffer zones and saves as GeoJSON
Using proper centerline (skeleton) vectorization.

Adds reference line attributes to each extracted crest based on buffer origin.
"""

import glob
import os
from collections import defaultdict

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import binary_erosion, gaussian_filter1d
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge
from skimage import measure, morphology
from skimage.morphology import thin


# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline on a new set of Sentinel-2 images. Update paths/
# settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# Sentinel-2 GeoTIFFs to process (glob pattern matched against this folder)
INPUT_TIF_FOLDER = "data/raw/tif"
INPUT_TIF_GLOB    = os.path.join(INPUT_TIF_FOLDER, "sossusvlei_*.tif")

# Reference lines used to build the search buffer around each dune
# (manually digitized in QGIS — same file referenced by the GEE download script)
GEOJSON_PATH = "data/raw/reference_lines/star_dune_crsts.geojson"

# ---- 2. PARAMETERS ----------------------------------------------------------
BUFFER_METERS       = 20     # search buffer width (m) around each reference line
CANNY_LOW           = 30     # Canny edge detector lower threshold
CANNY_HIGH          = 100    # Canny edge detector upper threshold
MIN_LENGTH_METERS   = 50     # discard extracted segments shorter than this
MAX_GAP_METERS      = 100    # bridge same-reference endpoints closer than this
SMOOTHING_SIGMA     = 1.5    # gaussian smoothing sigma along centerline coords

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = "data/interim/crests"
# Individual GeoJSONs are saved to OUTPUT_FOLDER as:
#   crests_{YYYY}_{MM}_{DD}.geojson
# (date tag is parsed from the matching input filename, e.g.
#  sossusvlei_2026_03_15.tif -> crests_2026_03_15.geojson)

# ---------------------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------------------
def load_image(image_path: str):
    """Return (nir_uint8, transform, crs)."""
    with rasterio.open(image_path) as src:
        nir = src.read(4).astype(np.float32)
        transform = src.transform
        crs = src.crs
    nir_max = nir.max()
    nir_norm = (nir / nir_max * 255).astype(np.uint8) if nir_max > 0 else nir.astype(np.uint8)
    return nir_norm, transform, crs


# ---------------------------------------------------------------------------
# 2. Create buffer mask with reference attributes
# ---------------------------------------------------------------------------
def create_buffer_mask_with_attributes(
    geojson_path: str, transform, crs, height: int, width: int, buffer_meters: float = 20
):
    """
    Rasterize each buffer separately and return:
    - mask: combined binary mask of all buffers
    - reference_map: array with reference_id for each pixel
    - reference_attributes: dict mapping reference_id to attributes
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    # Initialize arrays
    mask = np.zeros((height, width), dtype=np.uint8)
    reference_map = np.zeros((height, width), dtype=np.int32)
    
    # Store attributes by reference_id
    reference_attributes = {}
    
    # Rasterize each reference line separately
    for idx, row in gdf.iterrows():
        ref_id = idx + 1  # 1-based ID
        buffer_geom = row.geometry.buffer(buffer_meters)
        
        # Store attributes
        reference_attributes[ref_id] = {
            'reference_id': ref_id,
            'dune_id': row.get('dune_id', ref_id),
            'name': row.get('name', ref_id)
        }
        
        # Rasterize this buffer
        shapes = [(mapping(buffer_geom), ref_id)]
        buffer_array = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32,
        )
        
        # Update mask and reference_map
        mask[buffer_array > 0] = 1
        reference_map[buffer_array > 0] = ref_id
    
    # Erode mask to drop boundary artifacts
    mask = binary_erosion(mask == 1, iterations=1)
    reference_map = reference_map * mask  # Zero out eroded pixels
    
    return mask, reference_map, reference_attributes


# ---------------------------------------------------------------------------
# 3. Canny edge detection
# ---------------------------------------------------------------------------
def detect_edges(image: np.ndarray, buffer_mask: np.ndarray, low: int, high: int) -> np.ndarray:

    edges = cv2.Canny(image, low, high, apertureSize=7, L2gradient=True)
    return (edges > 0) & buffer_mask


# ---------------------------------------------------------------------------
# 4a. Centerline vectorization with boundary splitting
# ---------------------------------------------------------------------------
def split_path_by_reference(path, path_ref_ids, pixel_ref):
    """
    Split a path into segments where reference_id changes.
    Returns list of (segment_path, reference_id) tuples.
    """
    if not path:
        return []
    
    # If no reference info, assign based on majority of path
    if not path_ref_ids or all(r is None for r in path_ref_ids):
        return [(path, None)]
    
    segments = []
    current_seg = [path[0]]
    current_ref = path_ref_ids[0] if path_ref_ids else None
    
    for i in range(1, len(path)):
        node = path[i]
        node_ref = path_ref_ids[i] if i < len(path_ref_ids) else None
        
        # If reference_id changed (and both are valid)
        if node_ref != current_ref and node_ref is not None and current_ref is not None:
            # Find exact split point at boundary
            split_point = find_boundary_crossing(path[i-1], path[i], pixel_ref)
            if split_point:
                # Add the split point to current segment
                current_seg.append(split_point)
                if len(current_seg) >= 2:
                    segments.append((current_seg.copy(), current_ref))
                
                # Start new segment
                current_seg = [split_point, node]
                current_ref = node_ref
            else:
                # Fallback: just add node to current segment
                current_seg.append(node)
        else:
            # Same reference or one is None - continue
            current_seg.append(node)
            if node_ref is not None:
                current_ref = node_ref
    
    # Add final segment
    if len(current_seg) >= 2:
        segments.append((current_seg, current_ref))
    
    return segments


def find_boundary_crossing(pixel1, pixel2, pixel_ref):
    """
    Find the point where the line between pixel1 and pixel2 crosses 
    from one reference zone to another. Returns interpolated coordinate.
    """
    r1, c1 = pixel1
    r2, c2 = pixel2
    
    ref1 = pixel_ref.get(pixel1)
    ref2 = pixel_ref.get(pixel2)
    
    if ref1 == ref2:
        return None  # No crossing
    
    # Simple midpoint interpolation
    r_mid = (r1 + r2) / 2
    c_mid = (c1 + c2) / 2
    
    # Check neighboring pixels to find exact boundary
    for dr in [-0.3, 0, 0.3]:
        for dc in [-0.3, 0, 0.3]:
            test_r = int(r_mid + dr)
            test_c = int(c_mid + dc)
            if (test_r, test_c) in pixel_ref:
                if pixel_ref[(test_r, test_c)] == ref1:
                    continue
                elif pixel_ref[(test_r, test_c)] == ref2:
                    return (r_mid, c_mid)
    
    return (r_mid, c_mid)  # Fallback to midpoint


def raster_to_centerline_vector_with_boundary_splitting(
    edges: np.ndarray,
    reference_map: np.ndarray,
    reference_attributes: dict,
    transform,
    min_length_meters: float = 50,
    max_gap_meters: float = 100,
    smoothing_sigma: float = 1.5,
) -> list:
    """
    Skeleton -> graph -> traced polylines, SPLITTING at buffer zone boundaries.
    Returns a list of (LineString, reference_id) tuples where each line has 
    a SINGLE reference_id (no mixed attributes).
    """
    # --- skeletonize & clean ---
    skeleton = morphology.skeletonize(edges)
    skeleton = morphology.remove_small_objects(skeleton, max_size=4, connectivity=2)

    rows, cols = np.where(skeleton)
    if len(rows) == 0:
        return []

    # --- build graph with pixel attributes ---
    G = nx.Graph()
    G.add_nodes_from(zip(rows.tolist(), cols.tolist()))
    
    # Store reference_id for each pixel
    pixel_ref = {}
    for r, c in zip(rows, cols):
        if 0 <= r < reference_map.shape[0] and 0 <= c < reference_map.shape[1]:
            ref_id = reference_map[r, c]
            if ref_id > 0:
                pixel_ref[(r, c)] = ref_id

    for r, c in zip(rows, cols):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and 
                    skeleton[nr, nc]):
                    G.add_edge((r, c), (nr, nc))

    degrees = dict(G.degree())
    critical = {node for node, deg in degrees.items() if deg != 2}

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(critical)} critical nodes")

    # --- pixel to world function ---
    def pixel_to_world(r, c):
        x = transform[2] + (c+1) * transform[0]
        y = transform[5] + (r+1) * transform[4]
        return x, y

    def smooth_coords(coords):
        if smoothing_sigma <= 0 or len(coords) < 4:
            return coords
        arr = np.array(coords)
        sx = gaussian_filter1d(arr[:, 0], sigma=smoothing_sigma, mode="nearest")
        sy = gaussian_filter1d(arr[:, 1], sigma=smoothing_sigma, mode="nearest")
        return list(zip(sx.tolist(), sy.tolist()))

    lines_with_ref = []
    visited_edges = set()

    for start in critical:
        for neighbor in G.neighbors(start):
            key = (min(start, neighbor), max(start, neighbor))
            if key in visited_edges:
                continue

            # Trace path while tracking reference_id changes
            path = [start, neighbor]
            visited_edges.add(key)
            prev, current = start, neighbor
            
            # Track reference_id changes along path
            path_ref_ids = []
            for node in [start, neighbor]:
                if node in pixel_ref:
                    path_ref_ids.append(pixel_ref[node])

            # Walk until we hit another critical node
            while current not in critical:
                next_nodes = [n for n in G.neighbors(current) if n != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge_key = (min(current, nxt), max(current, nxt))
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)
                path.append(nxt)
                
                # Record reference_id
                if nxt in pixel_ref:
                    path_ref_ids.append(pixel_ref[nxt])
                
                prev, current = current, nxt
                if len(path) > 10_000:
                    break

            if len(path) < 2:
                continue

            # --- SPLIT PATH at reference_id boundaries ---
            segments = split_path_by_reference(path, path_ref_ids, pixel_ref)
            
            for seg_path, seg_ref_id in segments:
                if len(seg_path) < 2 or seg_ref_id is None:
                    continue
                    
                coords = [pixel_to_world(r, c) for r, c in seg_path]
                coords = smooth_coords(coords)
                line = LineString(coords)
                
                if line.length >= min_length_meters:
                    lines_with_ref.append((line, seg_ref_id))

    print(f"    {len(lines_with_ref)} centerline segments after splitting")
    
    # --- bridge gaps (only same reference_id) ---
    if max_gap_meters > 0 and len(lines_with_ref) > 1:
        lines_with_ref = bridge_gaps_with_ref(lines_with_ref, max_gap_meters)

    print(f"    {len(lines_with_ref)} segments after bridging")
    return lines_with_ref


# ---------------------------------------------------------------------------
# 5. Bridge gaps preserving reference_id
# ---------------------------------------------------------------------------
def bridge_gaps_with_ref(lines_with_ref: list, max_gap_meters: float) -> list:
    """
    Bridge gaps between lines with the same reference_id.
    Returns new list of (LineString, reference_id) tuples.
    """
    if len(lines_with_ref) <= 1:
        return lines_with_ref

    # Group by reference_id
    groups_by_ref = defaultdict(list)
    for line, ref_id in lines_with_ref:
        groups_by_ref[ref_id].append(line)
    
    result = []
    for ref_id, lines in groups_by_ref.items():
        if len(lines) == 1:
            result.append((lines[0], ref_id))
            continue
        
        # Bridge gaps for this reference group
        gdf = gpd.GeoDataFrame(geometry=lines, crs=None)
        parent = list(range(len(lines)))
        
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        
        def union(i, j):
            parent[find(i)] = find(j)
        
        buffered = gdf.geometry.buffer(max_gap_meters)
        sindex = gdf.sindex
        for i, buf in enumerate(buffered):
            candidates = list(sindex.intersection(buf.bounds))
            for j in candidates:
                if j != i and buf.intersects(gdf.geometry.iloc[j]):
                    union(i, j)
        
        groups = defaultdict(list)
        for i, line in enumerate(lines):
            groups[find(i)].append(line)
        
        for group_lines in groups.values():
            merged = linemerge(group_lines)
            if merged.geom_type == "LineString":
                result.append((merged, ref_id))
            else:  # MultiLineString
                for geom in merged.geoms:
                    result.append((geom, ref_id))
    
    return result


# ---------------------------------------------------------------------------
# 6. Save GeoJSON with reference attributes
# ---------------------------------------------------------------------------
def save_as_geojson_with_attributes(lines_with_ref: list, output_path: str, crs, reference_attributes: dict, acquisition_date: str) -> bool:
    if not lines_with_ref:
        print("  No lines to save")
        return False
    
    rows = []
    for idx, (line, ref_id) in enumerate(lines_with_ref):
        attrs = reference_attributes.get(ref_id, {})
        rows.append({
            "id": idx,
            "reference_id": ref_id,
            "dune_id": attrs.get('dune_id', ref_id),
            "dune_name": attrs.get('name'),
            "acquisition_date": acquisition_date,
            "length_m": line.length,
            "geometry": line,
        })
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"  Saved {len(rows)} lines -> {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------
def extract_dune_crests(
    image_path: str,
    output_path: str,
    geojson_path: str = GEOJSON_PATH,
    buffer_meters: float = BUFFER_METERS,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    min_length_meters: float = MIN_LENGTH_METERS,
    max_gap_meters: float = MAX_GAP_METERS,
    smoothing_sigma: float = SMOOTHING_SIGMA,
) -> bool:
    try:
        print("  Loading image...")
        nir, transform, crs = load_image(image_path)
        h, w = nir.shape

        print(f"  Creating buffer mask (r={buffer_meters} m)...")
        mask, reference_map, reference_attributes = create_buffer_mask_with_attributes(
            geojson_path, transform, crs, h, w, buffer_meters
        )

        print(f"  Detecting edges (Canny {canny_low}/{canny_high})...")
        edges = detect_edges(nir, mask, canny_low, canny_high)
        if not edges.any():
            print("  No edges detected")
            return False
        print(f"  {edges.sum()} edge pixels")

        print("  Vectorizing centerlines...")
        lines_with_ref = raster_to_centerline_vector_with_boundary_splitting(
            edges, reference_map, reference_attributes, transform, 
            min_length_meters, max_gap_meters, smoothing_sigma
        )

        if not lines_with_ref:
            print("  No vector lines created")
            return False

        # Extract acquisition date from filename
        import os
        basename = os.path.basename(image_path)
        date_str = basename.replace(".tif", "").split('_')[-1]
        
        return save_as_geojson_with_attributes(lines_with_ref, output_path, crs, reference_attributes, date_str)

    except Exception as exc:
        import traceback
        print(f"  Error: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    image_files = sorted(glob.glob(INPUT_TIF_GLOB))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 60)
    print("DUNE CREST EXTRACTION - CENTERLINE VECTORIZATION WITH BOUNDARY SPLITTING")
    print("=" * 60)
    print(f"  Images found   : {len(image_files)}")
    print(f"  Buffer         : {BUFFER_METERS} m")
    print(f"  Min length     : {MIN_LENGTH_METERS} m")
    print(f"  Max gap        : {MAX_GAP_METERS} m")
    print(f"  Smoothing sigma: {SMOOTHING_SIGMA}")
    print("=" * 60)

    ok = fail = 0
    for idx, path in enumerate(image_files, 1):
        base = os.path.basename(path)           # sossusvlei_YYYY_MM_DD.tif
        stem = base.replace(".tif", "")         # sossusvlei_YYYY_MM_DD
        tag  = "_".join(stem.split("_")[1:])    # YYYY_MM_DD
        out  = os.path.join(OUTPUT_FOLDER, f"crests_{tag}.geojson")

        print(f"\n[{idx}/{len(image_files)}] {base}")
        if extract_dune_crests(path, out):
            ok += 1
        else:
            fail += 1

    print("\n" + "=" * 60)
    print(f"Done - {ok} succeeded, {fail} failed")
    print("=" * 60)