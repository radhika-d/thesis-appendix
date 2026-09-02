import os
import pandas as pd
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline on a new weather station export. Update paths/
# settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# Folder of monthly weather station CSVs (semicolon-delimited)
INPUT_FOLDER = "data/raw/weather_station_data_csv"
INPUT_GLOB   = os.path.join(INPUT_FOLDER, "*.csv")

DATE_COLUMN     = "Date"
DATE_FORMAT_IN  = "%d %b %Y"   # matches "01 May 2017" style dates in source CSVs
DATE_FORMAT_OUT = "%d %b %Y"   # format used when writing dates back out

# ---- 2. PARAMETERS ----------------------------------------------------------
# Station metadata attached to every row
STATION_LAT = -24.1296
STATION_LON = 15.8947
STATION_ID  = "31201"

# Full date range to reindex against (fills in missing days as NaN rows so
# gaps are visible rather than silently absent)
DATE_RANGE_START = "2017-01-01"
DATE_RANGE_END   = "2026-06-30"

# Columns to retain in the final output (others dropped)
WIND_COLUMNS = [
    "Date", "Wind speed  (vc avg)", "Wind  direction  (vc avg)",
    "Wind  speed (max)", "Wind Dir.(Max wind speed)",
    "latitude", "longitude", "station_id",
]

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = "data/processed"
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "combined_weather_with_location.csv")


# ============================================================================
# SETUP
# ============================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================================
# PROCESSING
# ============================================================================

# Combine all monthly CSVs
all_files = glob.glob(INPUT_GLOB)
combined_df = pd.concat([pd.read_csv(f, delimiter=';') for f in all_files], ignore_index=True)

# Convert date format (handles "01 May 2017")
combined_df[DATE_COLUMN] = pd.to_datetime(combined_df[DATE_COLUMN], format=DATE_FORMAT_IN)

# Create complete date range and merge with existing data
all_dates = pd.date_range(start=DATE_RANGE_START, end=DATE_RANGE_END, freq='D')
complete_df = pd.DataFrame({DATE_COLUMN: all_dates})
complete_df = complete_df.merge(combined_df, on=DATE_COLUMN, how='left')

# Add location columns
complete_df['latitude'] = STATION_LAT
complete_df['longitude'] = STATION_LON
complete_df['station_id'] = STATION_ID

# Keep only wind-related columns
complete_df = complete_df[[col for col in WIND_COLUMNS if col in complete_df.columns]]

# Convert dates back to original "01 May 2017" format
complete_df[DATE_COLUMN] = complete_df[DATE_COLUMN].dt.strftime(DATE_FORMAT_OUT)

# Save using semicolon delimiter to match original format
complete_df.to_csv(OUTPUT_CSV, sep=';', index=False)

print(f"Done! File saved as '{OUTPUT_CSV}'")
print(f"Total dates in file: {len(complete_df)}")
print(f"Wind data present for: {complete_df['Wind speed  (vc avg)'].notna().sum()} dates")