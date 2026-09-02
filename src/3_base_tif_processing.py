"""
PRE-PROCESSING SCRIPT: Extract true color RGB from GeoTIFFs for Folium overlay
Input: tif\sossusvlei_YYYY_MM_YYYY-MM-DD.tif
Output: Base_tif\sossusvlei_YYYY_MM.png + Base_tif\metadata.json
"""

import os
import json
import numpy as np
from PIL import Image
import rasterio
from pyproj import Transformer
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FOLDER = "data/raw/tif"
OUTPUT_FOLDER = "data/processed/Base_tif"

# Band indices
BAND_RED = 3
BAND_GREEN = 2
BAND_BLUE = 1

# Enhancement parameters
GAMMA = [1.2, 1.2, 1.5]  # Per-band gamma correction
MULTIPLIER = [3, 3, 3]   # Per-band stretch multiplier
BLACK_POINT = [0.01, 0.02, 0.04]  # Per-band black point (haze removal)

# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_rgb_and_bounds(tif_path):
    with rasterio.open(tif_path) as src:
        # Read RGB bands
        red = src.read(BAND_RED).astype(np.float32)
        green = src.read(BAND_GREEN).astype(np.float32)
        blue = src.read(BAND_BLUE).astype(np.float32)
        
        # 1. Normalize each band using min-max (0-1 range)
        # Handle possible all-zero bands
        red_min, red_max = np.nanmin(red), np.nanmax(red)
        green_min, green_max = np.nanmin(green), np.nanmax(green)
        blue_min, blue_max = np.nanmin(blue), np.nanmax(blue)
        
        # Avoid division by zero
        red = np.where((red_max - red_min) > 0, (red - red_min) / (red_max - red_min), 0)
        green = np.where((green_max - green_min) > 0, (green - green_min) / (green_max - green_min), 0)
        blue = np.where((blue_max - blue_min) > 0, (blue - blue_min) / (blue_max - blue_min), 0)
        
        # 2. Apply multiplier (stretch values)
        red = red * MULTIPLIER[0]
        green = green * MULTIPLIER[1]
        blue = blue * MULTIPLIER[2]
        
        # 3. Apply black point correction (remove haze)
        red = np.clip(red, BLACK_POINT[0], 1)
        green = np.clip(green, BLACK_POINT[1], 1)
        blue = np.clip(blue, BLACK_POINT[2], 1)
        
        # 4. Apply gamma correction
        red = np.power(red, 1/GAMMA[0])
        green = np.power(green, 1/GAMMA[1])
        blue = np.power(blue, 1/GAMMA[2])
        
        # Stack and scale to 0-255
        rgb = np.stack([red, green, blue], axis=2) * 255
        rgb_norm = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # Replace NaN with 0 (just in case)
        rgb_norm = np.nan_to_num(rgb_norm, nan=0)
        
        # Get bounds in WGS84
        transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
        left, bottom = transformer.transform(src.bounds.left, src.bounds.bottom)
        right, top = transformer.transform(src.bounds.right, src.bounds.top)
        bounds = {"left": left, "right": right, "bottom": bottom, "top": top}
        
        # Get date from filename
        filename = os.path.basename(tif_path)
        parts = filename.replace('.tif', '').split('_')
        year = int(parts[1])
        month = int(parts[2])
        date_full = parts[3]
        
        return rgb_norm, bounds, year, month, date_full, filename

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    tif_files = sorted(Path(INPUT_FOLDER).glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in '{INPUT_FOLDER}'")
        return
    
    print(f"Found {len(tif_files)} GeoTIFF files")
    all_metadata = {}
    
    for tif_path in tif_files:
        print(f"Processing: {tif_path.name}")
        
        try:
            rgb, bounds, year, month, date_full, filename = extract_rgb_and_bounds(tif_path)
            
            output_name = f"sossusvlei_{year}_{month:02d}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            
            Image.fromarray(rgb, mode='RGB').save(output_path, format='PNG', compress_level=6)
            
            all_metadata[f"{year}_{month:02d}"] = {
                "year": year,
                "month": month,
                "date_full": date_full,
                "png_path": output_name,
                "bounds": bounds,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "source_file": filename
            }
            
            print(f"  Saved: {output_name} ({rgb.shape[1]}x{rgb.shape[0]})")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save metadata
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nDone! Processed {len(all_metadata)} files")

if __name__ == "__main__":
    main()