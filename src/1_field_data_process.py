"""
GNSS Point Processing Script
Converts multiple shapefiles to unified GeoJSON and creates lines for specific features
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline on a new data drop. Update paths below, nothing
# else in the script should need to change.
# ---- 2. PARAMETERS ---------------------------------------------------------
# Target CRS (UTM Zone 33S for Namibia)
TARGET_CRS = "EPSG:32733"

# Keywords for line creation
LINE_KEYWORDS = ['crest', 'edge', 'bowl']
EXCLUDED_KEYWORDS = ['P', 'CCP']  # These won't be made into lines

# ---- 1. INPUTS ------------------------------------------------------------
# GNSS point shapefiles (merged, then crest/edge/bowl lines derived from them)
SHAPEFILES = [
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_GNSS_exports/20260319_SOS1/20260319_SOS1 points.shp",
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_GNSS_exports/20260318_SOS1_shapefiles/20260318_SOS1_shpfile points.shp",
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_GNSS_exports/20260317sos_shapefiles/20260317sos points.shp",
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_GNSS_exports/20260320_SOS1_WEATHERSTATION_WE_shapefiles/20260320_SOS1_CCP points.shp",
]

# Geomorphology shapefiles (passed through: reprojected + saved, no line-building)
ADDITIONAL_FILES = [
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_Geomorpmapping_SOS1/Geomorph-SOS1_line-features.shp",
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_Geomorpmapping_SOS1/Geomorph-SOS1_polygon-features.shp",
    "2026SS_ThesisCarto_Radhika/202604_data/202603_Namibia_Geomorpmapping_SOS1/Geomorph-SOS1_point-features.shp"
]
# ---- 3. OUTPUTS -------------------------------------------------------------
# Output folder - now inside main_data/gnss_processed
OUTPUT_FOLDER = "data/processed/gnss_processed"
OUTPUT_POINTS = os.path.join(OUTPUT_FOLDER, "GNSS_all_points.geojson")
OUTPUT_EDGE_BOWL_LINES = os.path.join(OUTPUT_FOLDER, "GNSS_edge_bowl_lines.geojson")  # Only edge and bowl
OUTPUT_CREST_LINES = os.path.join(OUTPUT_FOLDER, "GNSS_crest_lines.geojson")  # Only crest lines
# Additional geomorph files are saved to OUTPUT_FOLDER using their original
# basenames (e.g. Geomorph-SOS1_line-features.geojson) — see step 8 in main().

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_attributes(name):
    """
    Extract group, direction, type, and sequence number from point name

    Examples:
    - SOS1_E_P4_001 -> group: SOS1, direction: E, type: P4, seq: 001
    - SOS1_E_crest_136 -> group: SOS1, direction: E, type: crest, seq: 136
    - SOS1_W2_CREST_00 -> group: SOS1, direction: W2, type: CREST, seq: 00
    - SOS1_SE_crest_045 -> group: SOS1, direction: SE, type: crest, seq: 045
    - SOS1_NW2_CREST_012 -> group: SOS1, direction: NW2, type: CREST, seq: 012
    - SOS1_N_bowl_074 -> group: SOS1, direction: N, type: bowl, seq: 074
    - SOS1_W_CREST004 -> group: SOS1, direction: W1, type: CREST, seq: 004
    """
    name_str = str(name)

    # Initialize defaults
    group = "Unknown"
    direction = "Unknown"
    feature_type = "Unknown"
    sequence = 0
    full_name = name_str

    # Parse the name
    parts = name_str.split('_')

    if len(parts) >= 2:
        group = parts[0]

        # Find feature type
        type_idx = -1
        for i, part in enumerate(parts):
            part_lower = part.lower()
            for keyword in LINE_KEYWORDS + EXCLUDED_KEYWORDS:
                if keyword.lower() in part_lower:
                    feature_type = part
                    type_idx = i
                    break
            if type_idx != -1:
                break

        if type_idx != -1:
            # Extract direction
            if type_idx > 1:
                # Direction is everything between group and type (has underscore)
                direction_parts = parts[1:type_idx]
                direction = '_'.join(direction_parts)
                # Keep as-is (no "1" added)
            elif type_idx == 1:
                # No underscore before type - check if type has attached direction
                # e.g., CREST004 - look at what's before the type in the original name
                before_type = name_str[len(group):].split(feature_type)[0]
                before_type = before_type.strip('_')
                if before_type:
                    # This means direction was attached directly to type (no underscore)
                    # Add "1" to indicate this
                    direction = before_type + '1'
                else:
                    direction = "Unknown"

            # Extract sequence
            if type_idx + 1 < len(parts):
                seq_part = parts[type_idx + 1]
                numbers = re.findall(r'\d+', seq_part)
                if numbers:
                    sequence = int(numbers[0])
            else:
                # No underscore after type (e.g., CREST004)
                type_part = parts[type_idx]
                numbers = re.findall(r'\d+', type_part)
                if numbers:
                    sequence = int(numbers[0])
                    # Remove numbers from type
                    feature_type = re.sub(r'\d+$', '', feature_type)

    # Clean up direction
    if direction == "" or direction is None:
        direction = "Unknown"

    # Determine if this should be a line feature
    should_make_line = feature_type.lower() in [k.lower() for k in LINE_KEYWORDS]

    return {
        'group': group,
        'direction': direction,
        'type': feature_type,
        'sequence': sequence,
        'full_name': full_name,
        'should_make_line': should_make_line
    }


def transform_shapefile(input_path, target_crs=TARGET_CRS):
    """
    Read shapefile and transform to target CRS
    """
    try:
        gdf = gpd.read_file(input_path)

        # Check if CRS is set
        if gdf.crs is None:
            print(f"Warning: {os.path.basename(input_path)} has no CRS defined. Assuming WGS84.")
            gdf = gdf.set_crs("EPSG:4326")

        return gdf.to_crs(target_crs)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def create_line_from_points(points_gdf, group_name, direction, feature_type):
    """
    Create a LineString from sorted points
    """
    # Filter points by group, direction, and type
    group_points = points_gdf[
        (points_gdf['group'] == group_name) &
        (points_gdf['direction'] == direction) &
        (points_gdf['type'] == feature_type)
    ]

    if len(group_points) < 2:
        return None

    # Sort by sequence number
    group_points_sorted = group_points.sort_values('sequence')

    # Check for duplicate sequences
    if len(group_points_sorted) != len(group_points_sorted['sequence'].unique()):
        print(f"  Warning: duplicate sequence numbers for {group_name}_{direction}_{feature_type} "
              f"- sorting by position instead")

        group_points_sorted = group_points_sorted.copy()
        group_points_sorted['temp_x'] = group_points_sorted.geometry.x
        group_points_sorted['temp_y'] = group_points_sorted.geometry.y

        # Sort along the dominant spread direction
        x_range = group_points_sorted['temp_x'].max() - group_points_sorted['temp_x'].min()
        y_range = group_points_sorted['temp_y'].max() - group_points_sorted['temp_y'].min()
        sort_cols = ['temp_x', 'temp_y'] if x_range >= y_range else ['temp_y', 'temp_x']
        group_points_sorted = group_points_sorted.sort_values(sort_cols)
        group_points_sorted = group_points_sorted.drop(columns=['temp_x', 'temp_y'])

    # Create line
    line = LineString(list(group_points_sorted.geometry))
    line_name = f"{group_name}_{direction}_{feature_type}" if direction != "Unknown" \
        else f"{group_name}_{feature_type}"

    return {
        'name': line_name,
        'geometry': line,
        'num_points': len(group_points_sorted),
        'length_m': line.length,
        'group': group_name,
        'direction': direction,
        'type': feature_type
    }


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Process all shapefiles
    all_gdfs = []
    for shp_path in SHAPEFILES:
        if not os.path.exists(shp_path):
            print(f"Warning: file not found: {shp_path}")
            continue

        gdf = transform_shapefile(shp_path)
        if gdf is not None and not gdf.empty:
            gdf['source_file'] = os.path.basename(shp_path)
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("Error: no shapefiles could be processed.")
        return

    # 2. Merge all shapefiles
    merged_gdf = pd.concat(all_gdfs, ignore_index=True)
    merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs=TARGET_CRS)

    # 3. Extract attributes from Name column
    attributes = merged_gdf['Name'].apply(extract_attributes)
    attr_df = pd.DataFrame(attributes.tolist())
    for col in attr_df.columns:
        merged_gdf[col] = attr_df[col]

    # 4. Save all points
    merged_gdf.to_file(OUTPUT_POINTS, driver="GeoJSON")
    print(f"✓ Points: {len(merged_gdf)} -> {OUTPUT_POINTS}")

    # 5. Create lines for crest, edge, bowl features
    line_points = merged_gdf[merged_gdf['should_make_line'] == True]
    line_groups = line_points.groupby(['group', 'direction', 'type'])
    
    crest_lines = []
    edge_bowl_lines = []

    for (group_name, direction, feature_type), points_subset in line_groups:
        line_info = create_line_from_points(points_subset, group_name, direction, feature_type)
        if line_info and len(line_info['geometry'].coords) >= 2:
            if feature_type.lower() == 'crest':
                crest_lines.append(line_info)
            else:
                edge_bowl_lines.append(line_info)

    # 6. Save crest lines
    if crest_lines:
        crest_gdf = gpd.GeoDataFrame(crest_lines, crs=TARGET_CRS)
        crest_gdf.to_file(OUTPUT_CREST_LINES, driver="GeoJSON")
        print(f"✓ Crest lines: {len(crest_lines)} -> {OUTPUT_CREST_LINES}")
    else:
        print("⚠️ No crest lines created")

    # 7. Save edge and bowl lines
    if edge_bowl_lines:
        edge_bowl_gdf = gpd.GeoDataFrame(edge_bowl_lines, crs=TARGET_CRS)
        edge_bowl_gdf.to_file(OUTPUT_EDGE_BOWL_LINES, driver="GeoJSON")
        print(f"✓ Edge/Bowl lines: {len(edge_bowl_lines)} -> {OUTPUT_EDGE_BOWL_LINES}")
    else:
        print("⚠️ No edge/bowl lines created")

    # 8. Process additional files (just transform and save as GeoJSON)
    for shp_path in ADDITIONAL_FILES:
        if not os.path.exists(shp_path):
            print(f"Warning: additional file not found: {shp_path}")
            continue
        
        print(f"Processing additional file: {os.path.basename(shp_path)}")
        gdf = transform_shapefile(shp_path)
        if gdf is not None and not gdf.empty:
            # Create output filename
            base_name = os.path.splitext(os.path.basename(shp_path))[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.geojson")
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"  Saved: {os.path.basename(output_path)} ({len(gdf)} features)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Points: {len(merged_gdf)}")
    print(f"  Crest lines: {len(crest_lines)}")
    print(f"  Edge/Bowl lines: {len(edge_bowl_lines)}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print("="*60)
    print("Done!")


if __name__ == "__main__":
    main()