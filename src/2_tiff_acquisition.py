import ee
import requests
import os
import csv
import geopandas as gpd

# ============================================================================
# CONFIGURATION
# ============================================================================
# Reproducibility note: this is the ONLY section that should need editing
# to re-run this pipeline for a new date range or study area. Update paths/
# settings below, nothing else in the script should need to change.

# ---- 1. INPUTS -------------------------------------------------------------
# GEE project (must have Earth Engine access enabled)
EE_PROJECT = 'ee-radhamaheshdhuri'

# GeoJSON files used ONLY to derive the study-area bounding box —
# not otherwise read or modified by this script
GEOJSON_FILES_FOR_BBOX = [
    'data/processed/gnss_processed/Geomorph-SOS1_line-features.geojson',
    'data/processed/gnss_processed/Geomorph-SOS1_point-features.geojson',
    'data/processed/gnss_processed/Geomorph-SOS1_polygon-features.geojson',
    'data/processed/gnss_processed/GNSS_all_points.geojson',
    'data/processed/gnss_processed/GNSS_crest_lines.geojson',
    'data/processed/gnss_processed/GNSS_edge_bowl_lines.geojson',
    'data/raw/reference_lines/star_dune_crsts.geojson',  # manually digitized in QGIS
    # Add more files here
]

# ---- 2. PARAMETERS ----------------------------------------------------------
YEARS = range(2017, 2027)
MONTHS = range(1, 13)
BANDS = ['B2', 'B3', 'B4', 'B8']          # Sentinel-2 SR bands to download
CLOUD_THRESHOLD_PCT = 30                                 # max CLOUDY_PIXEL_PERCENTAGE accepted
BBOX_PADDING_DEG = 0.01                                  # ~1km padding around derived bbox
SCALE_M = 10                                             # output pixel resolution (meters)
COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED'

# ---- 3. OUTPUTS --------------------------------------------------------------
OUTPUT_FOLDER = 'data/raw/tif'
# Individual GeoTIFFs are saved to OUTPUT_FOLDER as:
#   sossusvlei_{year}_{month:02d}_{date}.tif


# ============================================================================
# SETUP
# ============================================================================

ee.Initialize(project=EE_PROJECT)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ============================================================================
# DERIVE STUDY AREA FROM INPUT GEOJSONS
# ============================================================================

all_coords = []

for file_path in GEOJSON_FILES_FOR_BBOX:
    gdf = gpd.read_file(file_path)
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            all_coords.append((geom.x, geom.y))
        elif geom.geom_type == 'LineString':
            all_coords.extend([(c[0], c[1]) for c in geom.coords])
        elif geom.geom_type == 'Polygon':
            all_coords.extend([(c[0], c[1]) for c in geom.exterior.coords])
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                all_coords.extend([(c[0], c[1]) for c in line.coords])
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                all_coords.extend([(c[0], c[1]) for c in poly.exterior.coords])

if all_coords:
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    study_area = ee.Geometry.Rectangle([
        min_lon - BBOX_PADDING_DEG,
        min_lat - BBOX_PADDING_DEG,
        max_lon + BBOX_PADDING_DEG,
        max_lat + BBOX_PADDING_DEG
    ])

    print(f"Study area: {min_lon:.4f} to {max_lon:.4f}, {min_lat:.4f} to {max_lat:.4f}")
else:
    print("No coordinates found!")
    exit()

# ============================================================================
# DOWNLOAD
# ============================================================================

downloaded = 0
failed = 0

for year in YEARS:
    for month in MONTHS:
        print(f"{year}-{month:02d}", end=' ')

        try:
            start = f'{year}-{month:02d}-01'
            end = f'{year}-{month:02d}-28'

            collection = (ee.ImageCollection(COLLECTION_ID)
                          .filterBounds(study_area)
                          .filterDate(start, end)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD_PCT)))

            image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

            if image is None:
                print("None")
                failed += 1
                continue

            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            clouds = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

            url = image.getDownloadURL({
                'region': study_area,
                'bands': BANDS,
                'scale': SCALE_M,
                'format': 'GEOTIFF'
            })

            response = requests.get(url)
            filename = os.path.join(OUTPUT_FOLDER, f'sossusvlei_{year}_{month:02d}_{date}.tif')

            with open(filename, 'wb') as f:
                f.write(response.content)

            print(f"Saved: {date} ({clouds:.1f}%)")
            downloaded += 1

        except Exception as e:
            print(f"Error: {e}")
            failed += 1

print(f"\nDone! Downloaded: {downloaded}, Failed: {failed}")