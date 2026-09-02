"""
Process crest lines per date file:
1. Within each date file: dissolve by dune_id only (combine all segments of same dune)
2. Merge all dates together
3. Create connection lines to fill gaps
4. For each dune: calculate envelope and centerline using manual reference line
   - NEW: Also store signed distance for each date as separate attributes
"""

import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline on a new set of extracted crest dates. Update
# paths/settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# Per-date crest GeoJSONs, one per acquisition date (output of the crest
# extraction script — filenames matched as crests_*.geojson)
CRESTS_FOLDER = "data/interim/crests"
CRESTS_GLOB   = os.path.join(CRESTS_FOLDER, "crests_*.geojson")

# Manually digitized reference centerline per dune (same file used by the
# GEE download and crest extraction scripts)
MANUAL_CENTERLINE_FILE = "data/raw/reference_lines/star_dune_crsts.geojson"

# ---- 2. PARAMETERS ----------------------------------------------------------
SAMPLING_INTERVAL_M    = 10    # spacing between sample points along reference line
PERPENDICULAR_LENGTH_M = 100   # half-length of perpendicular probe line (each side)

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = 'data/processed/crest_post_process'
OUTPUT_EXTENDED         = os.path.join(OUTPUT_FOLDER, 'extended_centerlines.geojson') # crestlines detected + connection lines
OUTPUT_CENTERLINE       = os.path.join(OUTPUT_FOLDER, 'centerline_points.geojson') # crest movement points with per-date signed distances

# ============================================================================
# SETUP
# ============================================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ---------------------------------------------------------------------------
# 1. Process each date file: dissolve by dune_id only
# ---------------------------------------------------------------------------
def process_date_file(filepath: str) -> gpd.GeoDataFrame:
    filename = os.path.basename(filepath)
    date_str = filename.replace(".geojson", "").split('_')[-1]
    
    gdf = gpd.read_file(filepath)
    gdf['acquisition_date'] = date_str
    
    if 'dune_id' not in gdf.columns:
        print(f"  Warning: No dune_id column in {filename}, assigning all to dune 1")
        gdf['dune_id'] = 1
    
    def combine_to_multiline(geometries):
        lines = list(geometries)
        if len(lines) == 0:
            return None
        elif len(lines) == 1:
            return lines[0]
        else:
            return MultiLineString(lines)
    
    # Build agg dict for ALL columns
    agg_dict = {}

    for col in gdf.columns:
        if col == 'geometry':
            agg_dict[col] = combine_to_multiline
        elif col == 'length_m':
            agg_dict[col] = 'sum'
        else:
            agg_dict[col] = 'first'

    dissolved = gdf.groupby('dune_id', as_index=False).agg(agg_dict)
    
    dissolved_gdf = gpd.GeoDataFrame(dissolved, geometry='geometry', crs=gdf.crs)
    return dissolved_gdf


# ---------------------------------------------------------------------------
# 2. Load and process all date files
# ---------------------------------------------------------------------------
def load_all_dates(crests_folder: str) -> gpd.GeoDataFrame:
    all_dates_data = []
    filepaths = glob.glob(os.path.join(crests_folder, "crests_*.geojson"))
    print(f"Found {len(filepaths)} date files")
    
    for filepath in sorted(filepaths):
        filename = os.path.basename(filepath)
        print(f"  Processing {filename}...")
        
        try:
            processed = process_date_file(filepath)
            all_dates_data.append(processed)
            print(f"    Found {len(processed)} dunes in this date")
        except Exception as e:
            print(f"    Error processing {filename}: {e}")
    
    merged = pd.concat(all_dates_data, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=merged.crs if hasattr(merged, 'crs') else None)


# ---------------------------------------------------------------------------
# 2.5 Find shortest connections between segments
# ---------------------------------------------------------------------------
def find_all_connections_in_order(merged_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Find connections between segments in sequential order to fill all gaps.
    Dynamically chooses sorting axis based on dune orientation.
    """
    connection_results = []
    
    for (dune_id, date), group in merged_gdf.groupby(['dune_id', 'acquisition_date']):
        # Extract all segments
        segments = []
        for idx, row in group.iterrows():
            geom = row.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    segments.append(line)
            elif geom.geom_type == 'LineString':
                segments.append(geom)
        
        if len(segments) >= 2:
            # Determine orientation: get all centroids
            centroids = [seg.centroid for seg in segments]
            xs = [c.x for c in centroids]
            ys = [c.y for c in centroids]
            
            # Calculate span (range) in X and Y directions
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            
            # Choose sorting axis: X if dune is more horizontal, Y if more vertical
            if y_span > x_span:
                # Dune is vertically oriented - sort by Y coordinate
                segments_sorted = sorted(segments, key=lambda seg: seg.centroid.y)
                print(f"     Dune {dune_id} ({date}): Using Y-axis sort (vertical orientation, span Y={y_span:.1f} > X={x_span:.1f})")
            else:
                # Dune is horizontally oriented - sort by X coordinate
                segments_sorted = sorted(segments, key=lambda seg: seg.centroid.x)
                print(f"     Dune {dune_id} ({date}): Using X-axis sort (horizontal orientation, span X={x_span:.1f} >= Y={y_span:.1f})")
            
            # Connect consecutive segments
            for i in range(len(segments_sorted) - 1):
                p1, p2 = nearest_points(segments_sorted[i], segments_sorted[i+1])
                connection = LineString([p1, p2])
                
                connection_results.append({
                    'dune_id': dune_id,
                    'acquisition_date': date,
                    'gap_distance_m': p1.distance(p2),
                    'segment_pair': f"{i}-{i+1}",
                    'type': 'connection',
                    'geometry': connection
                })
                
                print(f"       Gap {i+1} = {p1.distance(p2):.2f}m")
    
    return gpd.GeoDataFrame(connection_results, geometry='geometry', crs=merged_gdf.crs)


# ---------------------------------------------------------------------------
# 2.6 Create extended lines (detected + connections) for reference/sampling
# ---------------------------------------------------------------------------
def create_extended_lines(detected_gdf: gpd.GeoDataFrame, connections_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge detected lines with connection lines to create continuous reference lines.
    Detected lines keep type='detected', connections keep type='connection'.
    """
    print("\nCreating extended lines for reference (detected + connections)...")
    
    # Add type to detected lines if not present
    detected_gdf = detected_gdf.copy()
    detected_gdf['type'] = 'detected'
    
    if connections_gdf is None or len(connections_gdf) == 0:
        print("   No connections to add, using only detected lines")
        return detected_gdf
    
    # Combine detected and connection lines
    extended_gdf = pd.concat([detected_gdf, connections_gdf], ignore_index=True)
    extended_gdf = gpd.GeoDataFrame(extended_gdf, geometry='geometry', crs=detected_gdf.crs)
    
    print(f"   Extended lines: {len(detected_gdf)} detected + {len(connections_gdf)} connections = {len(extended_gdf)} total")
    
    return extended_gdf


# ---------------------------------------------------------------------------
# 3. Calculate envelope and centerline with per-date signed distances
# ---------------------------------------------------------------------------
def calculate_envelope_and_centerline_with_dates(detected_lines: gpd.GeoDataFrame, 
                                                   reference_line: LineString,
                                                   sampling_interval: float, 
                                                   perp_length: float,
                                                   start_counter=0) -> dict:
    """
    Calculate envelope, centerline, AND store signed distance for EACH date.
    
    Returns:
        dict with keys:
        - distance_along: list of distances along reference line
        - centerline_point: list of Point geometries
        - min_distance: list of minimum signed distances (envelope left)
        - max_distance: list of maximum signed distances (envelope right)
        - width_m: list of widths
        - num_dates: list of number of dates with data
        - per_date_distances: dict where keys are date strings, values are lists of signed distances
        - point_id: list of unique identifiers for each sample point
    """
    if len(detected_lines) == 0 or reference_line is None:
        return None
    
    detected_lines = detected_lines.sort_values('acquisition_date')
    
    # STEP 1: Get ALL unique dates from the dataset
    all_dates = sorted(detected_lines['acquisition_date'].unique())
    print(f"     Found {len(all_dates)} unique dates: {', '.join(all_dates[:5])}{'...' if len(all_dates) > 5 else ''}")
    
    # STEP 2: Organize lines by date for quick lookup
    date_to_lines = {}
    for idx, row in detected_lines.iterrows():
        date = row['acquisition_date']
        geom = row.geometry
        
        lines_list = []
        if geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                lines_list.append(line)
        else:
            lines_list.append(geom)
        
        date_to_lines[date] = lines_list
    
    # STEP 3: Sample points along reference line
    sample_points = []
    total_length = reference_line.length
    current_dist = 0
    
    while current_dist <= total_length:
        point = reference_line.interpolate(current_dist)
        sample_points.append((current_dist, point))
        current_dist += sampling_interval
    
    # STEP 4: Initialize results containers
    results = {
        'distance_along': [],
        'centerline_point': [],
        'min_distance': [],
        'max_distance': [],
        'width_m': [],
        'num_dates': [],
        'per_date_distances': {date: [] for date in all_dates},
        'point_id': [],
        'orientation': []
    }
    
    # STEP 5: For each sample point, find signed distances for ALL dates
    point_counter = start_counter
    
    for dist_along, point in sample_points:
        point_id = f"point_{point_counter}"
        
        # Calculate perpendicular direction at this point
        offset = min(5.0, total_length * 0.01)
        dist_before = max(0, dist_along - offset)
        dist_after = min(total_length, dist_along + offset)
        
        point_before = reference_line.interpolate(dist_before)
        point_after = reference_line.interpolate(dist_after)
        
        dx = point_after.x - point_before.x
        dy = point_after.y - point_before.y
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            # Skip this point if direction cannot be determined
            for date in all_dates:
                results['per_date_distances'][date].append(np.nan)
            results['point_id'].append(point_id)
            results['orientation'].append(np.nan)
            point_counter += 1
            continue
        
        dx /= length
        dy /= length
        
        perp_dx = -dy
        perp_dy = dx
        
        # Calculate orientation angle (degrees from north, clockwise, 0-360)
        angle_rad = np.arctan2(perp_dx, perp_dy)
        angle_deg = np.degrees(angle_rad)
        orientation = angle_deg % 360
        
        # Create perpendicular line
        perp_start = Point(point.x - perp_dx * perp_length, 
                          point.y - perp_dy * perp_length)
        perp_end = Point(point.x + perp_dx * perp_length, 
                        point.y + perp_dy * perp_length)
        perpendicular = LineString([perp_start, perp_end])
        
        # For each date, find signed distance
        date_distances = {}
        
        for date in all_dates:
            signed_dist = np.nan
            
            if date in date_to_lines:
                for line in date_to_lines[date]:
                    intersection = perpendicular.intersection(line)
                    
                    if not intersection.is_empty:
                        # Extract intersection point
                        if intersection.geom_type == 'MultiPoint':
                            closest = min(intersection.geoms, 
                                        key=lambda p: point.distance(p))
                            intersection_point = closest
                        elif intersection.geom_type == 'Point':
                            intersection_point = intersection
                        elif intersection.geom_type == 'LineString':
                            intersection_point = intersection.interpolate(0.5, normalized=True)
                        else:
                            continue
                        
                        # Calculate signed distance
                        vec_x = intersection_point.x - point.x
                        vec_y = intersection_point.y - point.y
                        signed_dist = vec_x * perp_dx + vec_y * perp_dy
                        break
            
            date_distances[date] = signed_dist
        
        # Collect all signed distances (non-NaN) for envelope calculation
        valid_distances = [d for d in date_distances.values() if not np.isnan(d)]
        
        if len(valid_distances) >= 2:
            min_dist = min(valid_distances)
            max_dist = max(valid_distances)
            width = max_dist - min_dist
            
            # Centerline point is midpoint between min and max
            center_x = point.x + ((min_dist + max_dist) / 2) * perp_dx
            center_y = point.y + ((min_dist + max_dist) / 2) * perp_dy
            centerline_point = Point(center_x, center_y)
            
            # Store results for this sample point
            results['distance_along'].append(dist_along)
            results['centerline_point'].append(centerline_point)
            results['min_distance'].append(min_dist)
            results['max_distance'].append(max_dist)
            results['width_m'].append(width)
            results['num_dates'].append(len(valid_distances))
            results['orientation'].append(orientation)
            
            # Store per-date signed distances
            for date in all_dates:
                results['per_date_distances'][date].append(date_distances[date])
        else:
            # Not enough valid distances for envelope
            results['distance_along'].append(dist_along)
            results['centerline_point'].append(None)
            results['min_distance'].append(np.nan)
            results['max_distance'].append(np.nan)
            results['width_m'].append(np.nan)
            results['num_dates'].append(len(valid_distances))
            results['orientation'].append(orientation)
            
            for date in all_dates:
                results['per_date_distances'][date].append(date_distances[date])
        
        results['point_id'].append(point_id)
        point_counter += 1
    
    return results, point_counter


# ---------------------------------------------------------------------------
# 4. Main processing
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CREST POST-PROCESSING: Dissolve by Dune ID -> Merge Dates -> Envelope")
    print("(Using manual centerline as reference for sampling)")
    print("=" * 70)
    
    print("\n1. Processing individual date files...")
    all_dunes_all_dates = load_all_dates(CRESTS_FOLDER)
    print(f"\n   Total features: {len(all_dunes_all_dates)}")
    
    # Step 2.5: Find shortest connections
    connections_gdf = find_all_connections_in_order(all_dunes_all_dates)
    
    # Step 2.6: Create extended lines (detected + connections) for reference
    extended_gdf = create_extended_lines(all_dunes_all_dates, connections_gdf)
    extended_gdf.to_file(OUTPUT_EXTENDED, driver="GeoJSON")
    print(f"\n   Saved extended lines to {OUTPUT_EXTENDED}")
    
    # Step 3: Calculate envelope and centerline using manual reference line
    print("\n2. Loading manual centerline file...")
    manual_gdf = gpd.read_file(MANUAL_CENTERLINE_FILE)
    print(f"   Loaded {len(manual_gdf)} manual lines")
    
    print("\n3. Calculating envelope and centerline for each dune...")
    print(f"   Sampling interval: {SAMPLING_INTERVAL_M}m")
    
    centerline_results = []
    global_counter = 0
    
    for dune_id in sorted(all_dunes_all_dates['dune_id'].unique()):
        print(f"\n   Processing Dune {dune_id}...")
        
        detected_data = all_dunes_all_dates[all_dunes_all_dates['dune_id'] == dune_id]
        num_dates = len(detected_data)
        
        if num_dates < 2:
            print(f"     Warning: Only {num_dates} date(s), skipping")
            continue
        
        dates = sorted(detected_data['acquisition_date'].unique())
        print(f"     Dates: {', '.join(dates)}")
        
        manual_data = manual_gdf[manual_gdf['id'] == dune_id]
        if len(manual_data) == 0:
            print(f"     Warning: No manual line found for id={dune_id}, skipping")
            continue
        
        manual_line = manual_data.iloc[0].geometry
        if manual_line.geom_type == 'MultiLineString':
            parts = sorted(manual_line.geoms, key=lambda x: x.length, reverse=True)
            manual_line = parts[0]
        
        print(f"     Manual reference length: {manual_line.length:.1f}m")
        
        results, global_counter = calculate_envelope_and_centerline_with_dates(
            detected_data, manual_line, SAMPLING_INTERVAL_M, PERPENDICULAR_LENGTH_M, start_counter=global_counter
        )
        
        if results and len(results['distance_along']) > 0:
            # Get all date columns from results
            date_columns = sorted(results['per_date_distances'].keys())
            print(f"     Adding {len(date_columns)} date columns to output")
            
            for i in range(len(results['distance_along'])):
                # Skip if centerline point is None (insufficient data)
                if results['centerline_point'][i] is None:
                    continue
                    
                # Build feature dictionary with existing attributes
                feature = {
                    'point_id': results['point_id'][i],
                    'dune_id': dune_id,
                    'distance_along_m': results['distance_along'][i],
                    'width_m': results['width_m'][i],
                    'min_distance_m': results['min_distance'][i],
                    'max_distance_m': results['max_distance'][i],
                    'num_dates': results['num_dates'][i],
                    'orientation_deg': results['orientation'][i],
                    'geometry': results['centerline_point'][i]
                }
                
                # Add per-date signed distances as attributes
                for date in date_columns:
                    value = results['per_date_distances'][date][i]
                    date_clean = date.replace('-', '_')
                    col_name = f"date_{date_clean}"
                    feature[col_name] = value if not np.isnan(value) else None
                
                centerline_results.append(feature)
            
            valid_sample_points = len([r for r in centerline_results if r.get('dune_id') == dune_id])
            print(f"     Created {valid_sample_points} sample points with {len(date_columns)} date columns")
            valid_widths = [w for w in results['width_m'] if not np.isnan(w) and w > 0]
            if valid_widths:
                print(f"     Average width: {np.mean(valid_widths):.2f}m")
                print(f"     Max width: {np.max(valid_widths):.2f}m")
        else:
            print(f"     Warning: No valid results")
    
    print("\n4. Saving centerline results...")
    if centerline_results:
        centerline_gdf = gpd.GeoDataFrame(centerline_results, geometry='geometry', 
                                          crs=all_dunes_all_dates.crs)
        centerline_gdf.to_file(OUTPUT_CENTERLINE, driver="GeoJSON")
        print(f"   Saved {len(centerline_gdf)} centerline points to {OUTPUT_CENTERLINE}")
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print(f"  Extended lines: {OUTPUT_EXTENDED}")
    print(f"  Centerline points with date columns: {OUTPUT_CENTERLINE}")
    print("=" * 70)


if __name__ == "__main__":
    main()