"""
Dune crest uncertainty analysis (line version)
------------------------------------------------
1. Compares GNSS reference crest lines to your extracted (detected) crest lines.
2. Saves the result as GeoJSON:
   - measured points: a short LINE from the GNSS point to the matched detected
     point. The line's length IS the error - long line = big gap, short line =
     good match. This makes the map self-explanatory.
   - unmeasured points: kept as a POINT (flagged is_measured = False) so gaps
     still show up on the map instead of disappearing.
3. Saves a simple CSV of summary statistics (overall + per GNSS line).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this analysis for a different GNSS epoch or dune. Update
# paths/settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# Manually digitized + GNSS reference lines (same file used across the pipeline)
GNSS_FILE = "data/raw/reference_lines/star_dune_crsts.geojson"

# Automated crest extraction output being validated against GNSS
DETECTED_FILE = "data/interim/crests/crests_2026_03_2026-03-21.geojson"

# Restrict validation to a single dune. NOTE: this analysis is only valid for
# The Star Dune, which has real GNSS ground-truth from the March 2026 epoch -
# the other two dunes' reference lines were hastily digitized in QGIS and are
# not comparable to this error assessment.
GNSS_LINE_FILTER_FIELD = "name"
GNSS_LINE_FILTER_VALUE = "The Star Dune" # the crestline is cleaned in QGIS to remove overlap in gnss observations.

# ---- 2. PARAMETERS ----------------------------------------------------------
SAMPLING_INTERVAL_M = 10.0   # spacing between sample points along the GNSS line
SEARCH_LENGTH_M     = 20.0   # how far to search for a match (NOT the output line length)

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = "data/processed/uncertainty_analysis"
OUTPUT_GEOJSON = os.path.join(OUTPUT_FOLDER, "uncertainty_lines_length.geojson")
OUTPUT_STATS_CSV = os.path.join(OUTPUT_FOLDER, "uncertainty_statistics.csv")


# ---------------------------------------------------------------------------
# STEP 1: Compare one GNSS line to the detected lines, point by point
# ---------------------------------------------------------------------------
def compare_line_to_detected(gnss_line, detected_segments, sampling_interval,
                              search_length, crs):
    """
    search_length = how far to look on either side of the GNSS line for a
    match (a search window, not the output line length).
    """
    total_length = gnss_line.length

    distances_along = list(np.arange(0, total_length, sampling_interval))
    if distances_along[-1] < total_length:
        distances_along.append(total_length)

    rows = []
    geoms = []
    prev_perp = (1.0, 0.0)

    for dist_along in distances_along:
        ref_point = gnss_line.interpolate(dist_along)

        # Estimate direction of the line at this point, then rotate 90 degrees
        offset = min(5.0, total_length * 0.02)
        p_before = gnss_line.interpolate(max(0, dist_along - offset))
        p_after = gnss_line.interpolate(min(total_length, dist_along + offset))
        dx, dy = p_after.x - p_before.x, p_after.y - p_before.y
        norm = np.hypot(dx, dy)
        if norm == 0:
            perp_dx, perp_dy = prev_perp
        else:
            perp_dx, perp_dy = -dy / norm, dx / norm
            prev_perp = (perp_dx, perp_dy)

        search_line = LineString([
            Point(ref_point.x - perp_dx * search_length, ref_point.y - perp_dy * search_length),
            Point(ref_point.x + perp_dx * search_length, ref_point.y + perp_dy * search_length)
        ])

        # Find the closest point on any detected line, within the search window
        best_signed_dist = None
        best_match_point = None
        for segment in detected_segments:
            nearest = nearest_points(ref_point, segment)[1]
            if search_line.distance(nearest) < 1.0:
                signed = (nearest.x - ref_point.x) * perp_dx + (nearest.y - ref_point.y) * perp_dy
                if best_signed_dist is None or abs(signed) < abs(best_signed_dist):
                    best_signed_dist = signed
                    best_match_point = nearest

        is_measured = best_signed_dist is not None
        rows.append({
            "distance_along_m": dist_along,
            "signed_error_m": best_signed_dist,               # keeps direction (+/-), for bias check
            "abs_error_m": abs(best_signed_dist) if is_measured else np.nan,  # size of the gap
            "left_or_right": ("right" if best_signed_dist > 0 else "left") if is_measured else None,
            "is_measured": is_measured
        })

        # Output geometry: a line if matched, just a point if not
        if is_measured:
            geoms.append(LineString([ref_point, best_match_point]))
        else:
            geoms.append(ref_point)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


# ---------------------------------------------------------------------------
# STEP 2: Run the comparison for every GNSS line and stack results together
# ---------------------------------------------------------------------------
def run_uncertainty_analysis(gnss_gdf, detected_gdf, sampling_interval=10.0,
                              search_length=20.0, line_id_field="name"):
    if gnss_gdf.crs != detected_gdf.crs:
        detected_gdf = detected_gdf.to_crs(gnss_gdf.crs)
    crs = gnss_gdf.crs

    # Flatten all detected lines into simple LineStrings
    detected_segments = []
    for geom in detected_gdf.geometry:
        if geom.geom_type == "MultiLineString":
            detected_segments.extend(list(geom.geoms))
        elif geom.geom_type == "LineString":
            detected_segments.append(geom)

    all_rows = []
    for idx, row in gnss_gdf.iterrows():
        line_id = row[line_id_field] if line_id_field in row else f"line_{idx}"
        gnss_line = row.geometry
        if gnss_line.geom_type == "MultiLineString":
            gnss_line = max(gnss_line.geoms, key=lambda g: g.length)

        result = compare_line_to_detected(gnss_line, detected_segments, sampling_interval, search_length, crs)
        result["gnss_line_id"] = line_id
        all_rows.append(result)

    return gpd.GeoDataFrame(pd.concat(all_rows, ignore_index=True), crs=crs)


# ---------------------------------------------------------------------------
# STEP 3: Turn the results into a simple, labeled stats table
# ---------------------------------------------------------------------------
def summarize_errors(results_gdf, group_col=None):
    """Returns one row of stats. If group_col is given, returns one row per group too."""

    def stats_for(df):
        valid = df.dropna(subset=["abs_error_m"])
        signed = valid["signed_error_m"]
        abs_err = valid["abs_error_m"]
        n = len(df)
        n_valid = len(valid)

        rmse = np.sqrt(np.mean(signed ** 2)) if n_valid > 0 else np.nan

        return pd.Series({
            "n_sample_points": n,
            "n_valid_measurements": n_valid,
            "measurement_rate_pct": round(100 * n_valid / n, 1) if n > 0 else np.nan,
            "mean_error_ME_m": round(signed.mean(), 3) if n_valid > 0 else np.nan,        # bias: + or - shift
            "std_dev_m": round(signed.std(), 3) if n_valid > 0 else np.nan,                # spread of the error
            "rmse_m": round(rmse, 3) if n_valid > 0 else np.nan,                           # overall error size
            "mean_abs_error_m": round(abs_err.mean(), 3) if n_valid > 0 else np.nan,
            "median_abs_error_m": round(abs_err.median(), 3) if n_valid > 0 else np.nan,
            "min_abs_error_m": round(abs_err.min(), 3) if n_valid > 0 else np.nan,
            "max_abs_error_m": round(abs_err.max(), 3) if n_valid > 0 else np.nan,
            "margin_of_error_95pct_m": round(1.96 * rmse, 3) if n_valid > 0 else np.nan,   # "accurate to +/- this many meters"
        })

    if group_col is None:
        return stats_for(results_gdf).to_frame().T.assign(group="ALL")

    per_group = results_gdf.groupby(group_col).apply(stats_for).reset_index().rename(columns={group_col: "group"})
    overall = stats_for(results_gdf).to_frame().T.assign(group="ALL")
    return pd.concat([overall, per_group], ignore_index=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gnss_gdf = gpd.read_file(GNSS_FILE)
    gnss_gdf = gnss_gdf[gnss_gdf[GNSS_LINE_FILTER_FIELD] == GNSS_LINE_FILTER_VALUE]
    detected_gdf = gpd.read_file(DETECTED_FILE)

    results_gdf = run_uncertainty_analysis(
        gnss_gdf, detected_gdf,
        sampling_interval=SAMPLING_INTERVAL_M,
        search_length=SEARCH_LENGTH_M,
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. GeoJSON - error-vector lines (length = actual error) + points where unmeasured
    results_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

    # 2. CSV of statistics, overall + per GNSS line
    stats_df = summarize_errors(results_gdf)
    stats_df.to_csv(OUTPUT_STATS_CSV, index=False)
    print(f"Saved lines/points: {OUTPUT_GEOJSON}")
    print(f"Saved stats:        {OUTPUT_STATS_CSV}")
    print(stats_df.to_string(index=False))