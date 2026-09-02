"""
============================================================
 DUNE DASHBOARD  –  Streamlit app  (4:1 column layout)
 Aeolian dune monitoring: crest lines, movement, playas,
 wind roses, and GNSS uncertainty (2017-2026, May-August)
============================================================
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io, base64
from shapely.geometry import Point
from folium import plugins
from windrose import WindroseAxes
import os
import json
from PIL import Image
from pyproj import Transformer

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Cartographic Dashboard for Star Dune Dynamics Visualization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# Reproducibility note: this is the ONLY section that should need editing
# to point the dashboard at a new data drop. Update paths/settings below,
# nothing else in the script should need to change.

# ---- 1. DATA PATHS (INPUTS) -------------------------------------------------
DATA_CREST_LINES        = "data/processed/crest_post_process/extended_centerlines.geojson"
DATA_MOVEMENT_POINTS    = "data/processed/crest_post_process/centerline_points.geojson"
DATA_PLAYA_POLYGONS     = "data/processed/merged_playa.geojson"
DATA_WIND_CSV           = "data/processed/combined_weather_with_location.csv"
DATA_UNCERTAINTY_LINES  = "data/processed/uncertainty_analysis/uncertainty_lines_length.geojson"
DATA_GNSS_POINTS        = "data/processed/gnss_processed/GNSS_all_points.geojson"
DATA_GNSS_EDGE_BOWL     = "data/processed/gnss_processed/GNSS_edge_bowl_lines.geojson"
DATA_GNSS_CREST         = "data/processed/gnss_processed/GNSS_crest_lines.geojson"
DATA_GEOMORPH_LINES     = "data/processed/gnss_processed/Geomorph-SOS1_line-features.geojson"
DATA_GEOMORPH_POINTS    = "data/processed/gnss_processed/Geomorph-SOS1_point-features.geojson"
DATA_GEOMORPH_POLYGONS  = "data/processed/gnss_processed/Geomorph-SOS1_polygon-features.geojson"
DATA_HOBO_WIND_CSV      = "data/processed/Weatherstation_SOS_1_West_March_2026.csv"
DATA_BASE_IMAGERY_META  = "data/processed/Base_tif/metadata.json"
BASE_IMAGERY_FOLDER     = "data/processed/Base_tif"

# SOS 1 WEST station location (EPSG:32733 easting/northing)
HOBO_STATION_EASTING  = 533272.70433545333799
HOBO_STATION_NORTHING = 7262367.944419549778104

# ---- 2. PARAMETERS ----------------------------------------------------------
ALL_YEARS = list(range(2017, 2027))

ALL_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES = list(ALL_MONTHS.keys())
WIND_MONTHS = ALL_MONTHS

DAYS_IN_MONTH = {
    "January": 31, "February": 28, "March": 31, "April": 30,
    "May": 31, "June": 30, "July": 31, "August": 31,
    "September": 30, "October": 31, "November": 30, "December": 31,
}

DEFAULT_FOCUS_MONTHS = ["May", "June", "July", "August"]

UNC_GREEN = 2.0
UNC_YELLOW = 6.0
REPRESENTATIVE_MARGIN_OF_ERROR_M = 4.738

WIND_WARN_PCT = 0.70
WIND_HIDE_PCT = 0.30

MAP_CENTER = [-24.76, 15.31]
MAP_ZOOM = 14

# ---- 3. STYLE ----------------------------------------------------------------
MPL_BG = "#ffffff"
MPL_FG = "#050505"
MPL_GRID = "#bebebe"
MPL_ACCENT = "#0065BD"
MPL_ACCENT2 = "#C61826"

def month_abbr(month_name):
    return month_name[:3].upper()

def _dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=MPL_BG)
    ax.set_facecolor(MPL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL_GRID)
    ax.tick_params(colors=MPL_FG, labelsize=7)
    ax.xaxis.label.set_color(MPL_FG)
    ax.yaxis.label.set_color(MPL_FG)
    ax.title.set_color(MPL_ACCENT)
    return fig, ax

# ---- 4. BRANDING --------------------------------------------------------------
LOGO_HEIDELBERG = "assets/heidelberg_logo.svg"   # left
LOGO_TUM        = "assets/tum_logo.svg"          # right
PROJECT_URL     = "https://www.asg.ed.tum.de/rsa/forschung/star-dune-dynamics/"
AUTHOR_NAME     = "Radhika Dhuri"           

# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading base imagery...")
def load_base_imagery_metadata():
    if not os.path.exists(DATA_BASE_IMAGERY_META):
        return {}
    with open(DATA_BASE_IMAGERY_META, 'r') as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading crest lines ...")
def load_crest_lines():
    gdf = gpd.read_file(DATA_CREST_LINES).to_crs("EPSG:4326")
    gdf["date"] = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"] = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month
    if "is_gap_fill" not in gdf.columns:
        gdf["is_gap_fill"] = gdf["type"] == "connection"
    if "length_m" not in gdf.columns:
        gdf["length_m"] = gdf.geometry.length * 111_320
    return gdf

@st.cache_data(show_spinner="Loading movement points ...")
def load_movement_points():
    gdf = gpd.read_file(DATA_MOVEMENT_POINTS).to_crs("EPSG:4326")
    gdf["point_id"] = gdf["point_id"].astype(str)
    date_cols = [c for c in gdf.columns if c.startswith("date_")]
    geom_df = gdf[["point_id", "distance_along_m", "geometry", "orientation_deg"]].drop_duplicates("point_id")
    df_long = gdf[["point_id", "distance_along_m"] + date_cols].melt(
        id_vars=["point_id", "distance_along_m"],
        var_name="date_col", value_name="distance_m",
    ).dropna(subset=["distance_m"])
    df_long["date"] = pd.to_datetime(
        df_long["date_col"].str.replace("date_", "", regex=False).str.replace("_", "-"),
        format="%Y-%m-%d",
    )
    df_long["year"] = df_long["date"].dt.year
    df_long["month"] = df_long["date"].dt.month
    df_long = df_long.merge(geom_df, on=["point_id", "distance_along_m"], how="left")
    return gpd.GeoDataFrame(df_long, geometry="geometry", crs="EPSG:4326")

@st.cache_data(show_spinner="Loading playa (Highest Purity) ...")
def load_playa_polygons():
    gdf = gpd.read_file(DATA_PLAYA_POLYGONS).to_crs("EPSG:4326")
    gdf["date"] = pd.to_datetime(gdf["acquisition_date"])
    gdf["year"] = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month
    if "area_m2" not in gdf.columns:
        gdf["area_m2"] = gdf.geometry.area * (111_320 ** 2)
    return gdf

@st.cache_data(show_spinner="Loading wind data ...")
def load_wind_data():
    df = pd.read_csv(DATA_WIND_CSV, sep=";").rename(columns={
        "Date": "datetime",
        "Wind speed  (vc avg)": "speed_ms",
        "Wind  direction  (vc avg)": "direction",
    })
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d %b %Y")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df[df["month"].isin(WIND_MONTHS.values())]

@st.cache_data(show_spinner="Loading uncertainty lines ...")
def load_uncertainty_lines():
    gdf = gpd.read_file(DATA_UNCERTAINTY_LINES).to_crs("EPSG:4326")
    if "abs_error_m" in gdf.columns:
        gdf = gdf.rename(columns={"abs_error_m": "error_m"})
    if "signed_distance_m" in gdf.columns:
        gdf["gnss_val"] = gdf["signed_distance_m"]
    if "left_or_right" in gdf.columns:
        gdf["detected_val"] = gdf["left_or_right"]
    return gdf

@st.cache_data(show_spinner="Loading GNSS points ...")
def load_gnss_points():
    if not os.path.exists(DATA_GNSS_POINTS):
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(DATA_GNSS_POINTS).to_crs("EPSG:4326")
    except Exception:
        return gpd.GeoDataFrame()

@st.cache_data(show_spinner="Loading GNSS crest/edge/bowl lines ...")
def load_gnss_lines():
    gdf_edge_bowl = gpd.read_file(DATA_GNSS_EDGE_BOWL).to_crs("EPSG:4326")
    gdf_crest = gpd.read_file(DATA_GNSS_CREST).to_crs("EPSG:4326")
    return pd.concat([gdf_edge_bowl, gdf_crest], ignore_index=True)

@st.cache_data(show_spinner="Loading geomorphology layers ...")
def load_geomorph_layers():
    geomorph_data = {}
    geomorph_files = {
        "geomorph_lines": DATA_GEOMORPH_LINES,
        "geomorph_points": DATA_GEOMORPH_POINTS,
        "geomorph_polygons": DATA_GEOMORPH_POLYGONS,
    }
    for key, path in geomorph_files.items():
        if os.path.exists(path):
            try:
                geomorph_data[key] = gpd.read_file(path).to_crs("EPSG:4326")
            except Exception:
                geomorph_data[key] = gpd.GeoDataFrame()
        else:
            geomorph_data[key] = gpd.GeoDataFrame()
    return geomorph_data

@st.cache_data(show_spinner="Loading SOS 1 WEST wind data...")
def load_hobo_wind_data():
    df = pd.read_csv(DATA_HOBO_WIND_CSV, skiprows=1)
    df['datetime'] = pd.to_datetime(df['Datum Zeit, GMT+01:00'], format='%m.%d.%y %I:%M:%S %p')
    df = df.rename(columns={
        'Windrichtung, ø (LGR S/N: 22429151_duplicate, SEN S/N: 22407863, LBL: Wind direction)': 'direction',
        'Windgeschwindigkeit, m/s (LGR S/N: 22429151_duplicate, SEN S/N: 22427869, LBL: Wind speed)': 'speed_ms'
    })
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    return df

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def utm_to_latlon(easting, northing):
    transformer = Transformer.from_crs("EPSG:32733", "EPSG:4326")
    lon, lat = transformer.transform(easting, northing)
    return lon, lat

def get_base_imagery_for_date(metadata, selected_years, selected_months, date_a, date_b, preset):
    """Return the latest base imagery within the selected temporal period"""
    if not metadata:
        return None, None, None

    available_dates = sorted(metadata.keys())
    if not available_dates:
        return None, None, None

    if preset == "Compare":
        target_date = max(date_a, date_b) if date_a and date_b else None
        if target_date:
            target_year = target_date.year
            target_month = target_date.month
        else:
            target_year = 2026
            target_month = 8
    elif preset == "Annual":
        target_year = selected_years[-1] if selected_years else 2026
        target_month = ALL_MONTHS[selected_months[-1]] if selected_months else 8
    elif preset == "Monthly":
        target_year = selected_years[-1] if selected_years else 2026
        target_month = ALL_MONTHS[selected_months[0]] if selected_months else 5
    else:  # Custom
        target_year = max(selected_years) if selected_years else 2026
        target_month = ALL_MONTHS[selected_months[-1]] if selected_months else 8

    # First try exact match
    key = f"{target_year}_{target_month:02d}"
    if key in metadata:
        png_path = os.path.join(BASE_IMAGERY_FOLDER, metadata[key]["png_path"])
        bounds = metadata[key]["bounds"]
        date_full = metadata[key].get("date_full", None)
        return png_path, bounds, date_full

    # Fall back to the latest available date within the selected year range
    if selected_years:
        min_year = min(selected_years)
        max_year = max(selected_years)
    elif date_a and date_b:
        min_year = min(date_a.year, date_b.year)
        max_year = max(date_a.year, date_b.year)
    else:
        min_year = 2017
        max_year = 2026

    valid_keys = [k for k in available_dates if min_year <= int(k.split('_')[0]) <= max_year]

    if valid_keys:
        latest_key = valid_keys[-1]
        png_path = os.path.join(BASE_IMAGERY_FOLDER, metadata[latest_key]["png_path"])
        bounds = metadata[latest_key]["bounds"]
        date_full = metadata[latest_key].get("date_full", None)
        return png_path, bounds, date_full

    # Final fallback: latest available overall
    latest_key = available_dates[-1]
    png_path = os.path.join(BASE_IMAGERY_FOLDER, metadata[latest_key]["png_path"])
    bounds = metadata[latest_key]["bounds"]
    date_full = metadata[latest_key].get("date_full", None)
    return png_path, bounds, date_full

def date_colormap(dates):
    """Create color map for dates using custom palette"""
    timestamps = pd.to_datetime(dates).astype(np.int64)
    norm = plt.Normalize(timestamps.min(), timestamps.max())
    cmap = plt.cm.Blues
    return [mcolors.to_hex(cmap(0.2 + 0.7 * norm(t))) for t in timestamps]

def diverging_color(value, vmin=-10, vmax=10):
    norm = plt.Normalize(vmin, vmax)
    r, g, b, _ = plt.cm.berlin(norm(value))
    return mcolors.to_hex((r, g, b))

def unc_color(error_m):
    if error_m < UNC_GREEN: return "#31A857"
    if error_m < UNC_YELLOW: return "#F7E62C"
    return "#C7400F"

def wind_completeness(wind_df, years=None, months=None, date_a=None, date_b=None):
    if date_a is not None and date_b is not None:
        d0, d1 = sorted([pd.Timestamp(date_a), pd.Timestamp(date_b)])
        sub = wind_df[(wind_df["datetime"] >= d0) & (wind_df["datetime"] <= d1)]
        expected_days = (d1 - d0).days + 1
        if expected_days <= 0:
            return 0.0, sub
        frac = min(sub["direction"].notna().sum() / expected_days, 1.0)
        return frac, sub
    else:
        m_nums = [WIND_MONTHS[m] for m in months if m in WIND_MONTHS]
        sub = wind_df[wind_df["year"].isin(years) & wind_df["month"].isin(m_nums)]
        if sub.empty:
            return 0.0, sub
        expected = sum(DAYS_IN_MONTH[m] for y in years for m in months if m in DAYS_IN_MONTH)
        if expected == 0:
            return 0.0, sub
        frac = min(sub["direction"].notna().sum() / expected, 1.0)
        return frac, sub

def build_wind_rose_image(wind_df):
    fig = plt.figure(figsize=(2.8, 2.8), facecolor=MPL_BG)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_df["direction"], wind_df["speed_ms"],
           normed=True, opening=0.8, edgecolor=MPL_GRID,
           cmap=plt.cm.Reds, bins=np.arange(0, 12, 2))
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, labelsize=8)
    fig.patch.set_alpha(1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=MPL_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def build_simple_wind_rose(wind_df, date_start, date_end):
    sub = wind_df[(wind_df["datetime"] >= pd.Timestamp(date_start)) &
                  (wind_df["datetime"] <= pd.Timestamp(date_end))]
    if sub.empty or sub["direction"].notna().sum() < 5:
        return None, False

    fig = plt.figure(figsize=(2.2, 2.2), facecolor=MPL_BG)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(sub["direction"], sub["speed_ms"],
           normed=True, opening=0.8, edgecolor=MPL_GRID,
           cmap=plt.cm.Reds, bins=np.arange(0, 12, 2))
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_FG, labelsize=6)
    ax.set_title('')
    ax.legend().set_visible(False)
    fig.patch.set_alpha(1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor=MPL_BG)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), True

def get_wind_rose_pairs(crest_gdf, preset, selected_years, selected_months):
    pairs = []
    if crest_gdf.empty:
        return pairs
    dates = sorted(crest_gdf["date"].unique())

    if preset == "Monthly":
        month_num = ALL_MONTHS[selected_months[0]]
        month_dates = [d for d in dates if d.month == month_num]
        for i in range(len(month_dates) - 1):
            pairs.append((month_dates[i], month_dates[i + 1],
                         f"{month_dates[i].year}→{month_dates[i+1].year}"))
    elif preset == "Annual":
        year = selected_years[0]
        year_dates = [d for d in dates if d.year == year]
        for i in range(len(year_dates) - 1):
            pairs.append((year_dates[i], year_dates[i + 1],
                         f"{year_dates[i].strftime('%b')}→{year_dates[i+1].strftime('%b')}"))
    return pairs

def build_gantt_figure(wind_df, years, months):
    fig, ax = _dark_fig(5.6, max(1.8, len(years) * 0.38))
    m_nums = [WIND_MONTHS[m] for m in months if m in WIND_MONTHS]

    ordered_months = sorted(WIND_MONTHS.items(), key=lambda item: item[1])
    x0_by_month, xticks, xlabels = {}, [], []
    cursor = 0
    for m_name, m_num in ordered_months:
        x0_by_month[m_name] = cursor
        xticks.append(cursor + DAYS_IN_MONTH[m_name] / 2)
        xlabels.append(month_abbr(m_name))
        cursor += DAYS_IN_MONTH[m_name] + 2

    for i, year in enumerate(sorted(years)):
        for m_name, m_num in ordered_months:
            if m_num not in m_nums:
                continue
            sub = wind_df[(wind_df.year == year) & (wind_df.month == m_num) & (wind_df["direction"].notna())]
            days = DAYS_IN_MONTH[m_name]
            x0 = x0_by_month[m_name]
            valid = len(sub)
            miss = days - valid
            pct = (valid / days) * 100

            ax.barh(i, valid, left=x0, height=0.55, color="#2D7D46", alpha=0.85)
            if valid > 0:
                ax.text(x0 + days / 2, i, f"{pct:.0f}%", ha="center", va="center", fontsize=6, color="#FFFFFF", fontweight="bold")
            if miss > 0:
                ax.barh(i, miss, left=x0 + valid, height=0.55, color=MPL_ACCENT2, alpha=0.75)

    ax.set_yticks(range(len(sorted(years))))
    ax.set_yticklabels(sorted(years), fontsize=7, color=MPL_FG)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, color=MPL_FG)
    ax.set_title("Wind Coverage", fontsize=12, color=MPL_ACCENT, pad=4)
    ax.grid(axis="x", color=MPL_GRID, linewidth=0.4, linestyle=":")
    fig.tight_layout(pad=0.4)
    return fig

def date_filter(gdf, selected_years, selected_months):
    if gdf.empty or "year" not in gdf.columns:
        return gdf
    m_nums = [ALL_MONTHS[m] for m in selected_months]
    return gdf[gdf["year"].isin(selected_years) & gdf["month"].isin(m_nums)].copy()

# ------------------------------------------------------------------------------
# MAP BUILDER
# ------------------------------------------------------------------------------

def build_map(
    crest_gdf, var_gdf, playa_gdf, unc_gdf,
    wind_b64, wind_completeness_pct,
    show_crests, show_gap_fills,
    show_movement, date_a, date_b,
    show_playa, show_wind, show_uncertainty,
    show_margin_buffer,
    opacity, date_min=None, date_max=None,
    show_base_imagery=True, base_metadata=None,
    selected_years=None, selected_months=None,
    preset="Custom",
    show_gnss_points=False, show_gnss_lines=False,
    show_geomorph_lines=False, show_geomorph_points=False,
    show_geomorph_polygons=False,
    gnss_points_gdf=None, gnss_lines_gdf=None,
    geomorph_data=None,
    hobo_df=None, show_hobo_wind=False,
    custom_center=None, custom_zoom=None,
):
    center = custom_center if custom_center else MAP_CENTER
    zoom = custom_zoom if custom_zoom else MAP_ZOOM
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    plugins.MeasureControl(
        position="bottomright",
        primary_length_unit="meters", secondary_length_unit="kilometers",
        primary_area_unit="sqmeters", secondary_area_unit="hectares",
        active_color=MPL_ACCENT, completed_color="#2D7D46",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OSM Standard",
        overlay=False,
        control=True,
    ).add_to(m)

    # Base Imagery
    base_img_date = None
    if show_base_imagery and base_metadata:
        png_path, bounds, base_img_date = get_base_imagery_for_date(base_metadata, selected_years, selected_months, date_a, date_b, preset)
        if png_path and bounds:
            img = Image.open(png_path)
            img_array = np.array(img)
            folium.raster_layers.ImageOverlay(
                image=img_array,
                bounds=[[bounds["bottom"], bounds["left"]], [bounds["top"], bounds["right"]]],
                opacity=1.0, name="Base Imagery", overlay=True, control=True
            ).add_to(m)

    # 1. UNCERTAINTY LINES
    if show_uncertainty and not unc_gdf.empty:
        for _, row in unc_gdf.iterrows():
            err = row.get("error_m", 0)
            if pd.isna(err):
                continue
            c = unc_color(err)
            coords = list(row.geometry.coords)
            folium.PolyLine(
                locations=[[lat, lon] for lon, lat in coords],
                color=c, weight=5, opacity=opacity,
                tooltip=folium.Tooltip(f"<b>GNSS Displacement</b><br>Error: {err:.2f} m")
            ).add_to(m)

    # 2. MARGIN OF ERROR BUFFER
    if show_margin_buffer and not crest_gdf.empty:
        try:
            buffered = crest_gdf.to_crs("EPSG:32733")
            buffered = buffered.assign(geometry=buffered.geometry.buffer(REPRESENTATIVE_MARGIN_OF_ERROR_M))
            dissolved = buffered.dissolve().to_crs("EPSG:4326")
            folium.GeoJson(
                dissolved.geometry.iloc[0].__geo_interface__,
                style_function=lambda f: {"fillColor": "#C61826", "color": "#C61826", "weight": 0, "fillOpacity": opacity * 0.5},
                tooltip=folium.Tooltip(f"<b>Margin of Error (95% CI)</b><br>±{REPRESENTATIVE_MARGIN_OF_ERROR_M:.2f} m <p> Based on March 2026 Epoch only</p>")
            ).add_to(m)
        except Exception:
            pass

    # 3. PLAYA
    if show_playa and not playa_gdf.empty:
        dates_sorted = sorted(playa_gdf["date"].unique())
        date_color_map = dict(zip([str(d) for d in dates_sorted], date_colormap(dates_sorted)))
        for _, row in playa_gdf.iterrows():
            c = date_color_map.get(str(row["date"]), MPL_GRID)
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, col=c: {"fillColor": col, "color": col, "weight": 1, "fillOpacity": opacity * 0.5},
                tooltip=folium.Tooltip(f"<b>Playa</b><br>{row['date'].date()}")
            ).add_to(m)

    # 4. CREST LINES
    if show_crests and not crest_gdf.empty:
        dates_sorted = sorted(crest_gdf["date"].unique())
        date_color_map = dict(zip([str(d) for d in dates_sorted], date_colormap(dates_sorted)))

        for _, row in crest_gdf.iterrows():
            if row.get("is_gap_fill", False) and not show_gap_fills:
                continue
            c = date_color_map.get(str(row["date"]), MPL_GRID)
            style = {
                "color": c,
                "weight": 2,
                "opacity": opacity * 0.5 if row.get("is_gap_fill", False) else opacity,
                "dashArray": "8 4" if row.get("is_gap_fill", False) else None
            }
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, s=style: s,
                tooltip=folium.Tooltip(
                    f"<b>Crest</b><br>{row['date'].date()}<br>"
                    + (" [gap fill]" if row.get("is_gap_fill") else "")
                ),
            ).add_to(m)

    # 5. MOVEMENT POINTS
    if show_movement and date_a and date_b and not var_gdf.empty:
        def _agg(df, dt):
            sub = df[df["date"] == pd.Timestamp(dt)]
            return sub.groupby("point_id")["geometry"].first(), sub.groupby("point_id")["distance_m"].mean()

        geom_a, dist_a = _agg(var_gdf, date_a)
        geom_b, dist_b = _agg(var_gdf, date_b)
        common = dist_a.index.intersection(dist_b.index)

        orientation_lookup = {}
        if 'orientation_deg' in var_gdf.columns:
            for pid in common:
                orientation_values = var_gdf[var_gdf['point_id'] == pid]['orientation_deg'].unique()
                if len(orientation_values) > 0 and not pd.isna(orientation_values[0]):
                    orientation_lookup[pid] = orientation_values[0]
                else:
                    orientation_lookup[pid] = None

        for pid in common:
            if pid not in orientation_lookup or orientation_lookup[pid] is None:
                continue

            val_a = float(dist_a[pid])
            val_b = float(dist_b[pid])
            geom = geom_b[pid]
            diff = val_b - val_a
            color = diverging_color(diff)
            magnitude = abs(diff)
            size = max(10, min(25, 10 + magnitude * 1.8))

            orientation = orientation_lookup[pid]
            if orientation >= 315 or orientation < 45:
                direction_label = "North"
            elif 45 <= orientation < 135:
                direction_label = "East"
            elif 135 <= orientation < 225:
                direction_label = "South"
            else:
                direction_label = "West"

            arrow_deg = orientation if diff >= 0 else (orientation + 180) % 360

            arrow_svg = f"""
            <div style="width:{size}px;height:{size}px;transform:rotate({arrow_deg-90}deg);opacity:0.7;">
            <svg width="{size}" height="{size}" viewBox="0 0 24 24">
                <line x1="0" y1="12" x2="19" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
                <polygon points="17,8 23,12 17,16" fill="{color}"/>
            </svg>
            </div>"""

            folium.Marker(
                location=[float(geom.y), float(geom.x)],
                icon=folium.DivIcon(icon_size=(size, size), icon_anchor=(size/2, size/2), html=arrow_svg),
                tooltip=folium.Tooltip(
                    f"<b>Crest movement</b><br>"
                    f"Point ID: {pid}<br>"
                    f"Date A: {pd.Timestamp(date_a).strftime('%b %Y')}<br>"
                    f"Date B: {pd.Timestamp(date_b).strftime('%b %Y')}<br>"
                    f"<b>Movement: {magnitude:.2f} m</b><br>"
                ),
            ).add_to(m)

    # 6. GNSS AND GEOMORPHOLOGY LAYERS
    if show_gnss_points and gnss_points_gdf is not None and not gnss_points_gdf.empty:
        type_colors = {
            'p3': '#FF6B6B', 'p4': '#FF6B6B', 'ccp': '#4ECDC4',
            'crest': '#FFD93D', 'edge': '#6C5CE7', 'bowl': '#A8E6CF'
        }
        default_color = '#95A5A6'
        for _, row in gnss_points_gdf.iterrows():
            point_type = row.get('type', 'Unknown')
            color = type_colors.get(point_type.lower(), default_color)
            name = row.get('full_name', row.get('Name', 'Unknown'))
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2, color=color, fill=True, stroke=False,
                fill_color=color, fill_opacity=1, weight=5,
                tooltip=folium.Tooltip(f"<b>GNSS Point</b><br>Name: {name}<br>Type: {point_type}")
            ).add_to(m)

    if show_gnss_lines and gnss_lines_gdf is not None and not gnss_lines_gdf.empty:
        type_colors = {'crest': '#FFD93D', 'CREST': '#FFD93D', 'edge': '#6C5CE7', 'bowl': '#A8E6CF'}
        default_color = '#95A5A6'
        for _, row in gnss_lines_gdf.iterrows():
            coords = list(row.geometry.coords)
            locations = [[coord[1], coord[0]] for coord in coords]
            if len(locations) >= 2:
                folium.PolyLine(
                    locations=locations,
                    color=type_colors.get(row.get('type'), default_color),
                    weight=3, opacity=opacity,
                    tooltip=folium.Tooltip(f"<b>GNSS Line</b><br>Name: {row.get('name', 'Unknown')}")
                ).add_to(m)

    if show_geomorph_lines and geomorph_data is not None and 'geomorph_lines' in geomorph_data:
        gdf_lines = geomorph_data['geomorph_lines']
        if not gdf_lines.empty:
            for _, row in gdf_lines.iterrows():
                coords = list(row.geometry.coords)
                folium.PolyLine(
                    locations=[[lat, lon] for lon, lat in coords],
                    color="#201C19", weight=2, opacity=opacity * 0.7,
                    tooltip=folium.Tooltip(f"<b>Geomorph Line</b><br>ID: {row.get('id', 'Unknown')}")
                ).add_to(m)

    if show_geomorph_points and geomorph_data is not None and 'geomorph_points' in geomorph_data:
        gdf_points = geomorph_data['geomorph_points']
        if not gdf_points.empty:
            for _, row in gdf_points.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4, color="#FD6F17", fill=True,
                    fill_color='#FD6F17', fill_opacity=opacity * 0.7, weight=1,
                    tooltip=folium.Tooltip(f"<b>Geomorph Point</b><br>ID: {row.get('Sample', 'Unknown')}")
                ).add_to(m)

    if show_geomorph_polygons and geomorph_data is not None and 'geomorph_polygons' in geomorph_data:
        gdf_polygons = geomorph_data['geomorph_polygons']
        if not gdf_polygons.empty and 'Feature' in gdf_polygons.columns:
            type_colors = {'Fossil Dune': '#6C5CE7', 'Recent Vlei': "#0CC883"}
            default_color = '#95A5A6'
            for _, row in gdf_polygons.iterrows():
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style={
                        "fillColor": type_colors.get(row.get('Feature'), default_color),
                        "color": type_colors.get(row.get('Feature'), default_color),
                        "weight": 1, "fillOpacity": opacity * 0.3,
                    },
                    tooltip=folium.Tooltip(f"<b>Geomorph Polygon</b><br>ID: {row.get('Feature', 'Unknown')}")
                ).add_to(m)

    # 7. WIND ROSE OVERLAY
    if show_wind and wind_b64 and wind_completeness_pct is not None:
        badge = ""
        if wind_completeness_pct < WIND_WARN_PCT:
            badge = f'<div style="background:#FFF3CD;color:#6B4E00;font-size:9px;padding:2px 5px;border-radius:3px;margin-top:3px;font-weight:700;">! {wind_completeness_pct*100:.0f}% coverage</div>'

        if preset == "Compare":
            date_label = f"{date_a.strftime('%b %Y')} – {date_b.strftime('%b %Y')}"
        elif preset in ["Annual", "Monthly"]:
            if date_a and date_b:
                if date_a.year == date_b.year:
                    date_label = f"{date_a.strftime('%b')}–{date_b.strftime('%b %Y')}"
                else:
                    date_label = f"{date_a.strftime('%b %Y')}–{date_b.strftime('%b %Y')}"
            else:
                date_label = preset
        else:
            date_label = ""

        html = f"""
        <div style="position:fixed;top:10px;right:10px;z-index:9999;
                    background:rgba(235,232,232,0.92);border:1px solid {MPL_GRID};
                    border-radius:8px;padding:7px 8px;text-align:center;opacity:1.0;font-family:sans-serif;">
          <div style="font-size:8px;color:{MPL_FG};font-weight:600;margin-bottom:3px;">{date_label}</div>
          <img src="data:image/png;base64,{wind_b64}" width="150"/>
          <div style="margin-top:4px;font-size:8px;color:{MPL_FG};font-weight:600;">Wind speed (m/s)</div>
          <div style="display:flex;align-items:center;margin-top:2px;gap:2px;">
            <span style="font-size:7px;color:{MPL_FG};">0</span>
            <div style="flex:1;height:6px;background:linear-gradient(to right,
              #FEE5D9, #FCAE91, #FB6A4A, #DE2D26, #A50F15, #67000D);
              border-radius:2px;margin:0 2px;"></div>
            <span style="font-size:7px;color:{MPL_FG};">10+</span>
          </div>
          {badge}
        </div>"""
        m.get_root().html.add_child(folium.Element(html))

    # 7b. SOS 1 WEST WIND ROSE
    if show_hobo_wind and hobo_df is not None and not hobo_df.empty:
        fig = plt.figure(figsize=(1.5, 1.5), facecolor='none')
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(hobo_df["direction"], hobo_df["speed_ms"],
               normed=True, opening=0.8, edgecolor="#050505", linewidth=0.3,
               cmap=plt.cm.Reds, bins=np.arange(0, 12, 2))
        ax.set_facecolor('none')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.legend().set_visible(False)
        ax.set_title('')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.patch.set_alpha(0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', transparent=True)
        plt.close(fig)
        hobo_b64 = base64.b64encode(buf.getvalue()).decode()

        hobo_lat, hobo_lon = utm_to_latlon(HOBO_STATION_EASTING, HOBO_STATION_NORTHING)

        folium.Marker(
            location=[hobo_lat, hobo_lon],
            icon=folium.DivIcon(
                html=f'<div style="transform: translate(-50%, -50%);"><img src="data:image/png;base64,{hobo_b64}" width="55"/></div>',
                icon_size=(55, 55), icon_anchor=(27, 27)
            ),
            tooltip="SOS 1 WEST Weather Station (16 - 20 March 2026)"
        ).add_to(m)

    # 8. LEGEND
    sections = []

    if show_crests and not crest_gdf.empty:
        dates_sorted = sorted(crest_gdf["date"].unique())
        date_color_map = dict(zip([str(d) for d in dates_sorted], date_colormap(dates_sorted)))

        legend_items = []
        display_dates = dates_sorted[:6] if len(dates_sorted) > 6 else dates_sorted

        for date in display_dates:
            color = date_color_map.get(str(date), MPL_GRID)
            date_str = date.strftime("%b %Y")
            legend_items.append(f"""
                <div class="lr">
                    <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4"
                        stroke="{color}" stroke-width="2"/></svg>
                    <span>{date_str}</span>
                </div>
            """)

        legend_items.append(f"""
            <div class="lr">
                <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4"
                    stroke="#C49A6C" stroke-width="1.5" stroke-dasharray="5,3"/></svg>
                <span>Gap fill</span>
            </div>
        """)

        if show_playa:
            legend_items.append("""
                <div class="lr">
                    <svg width="24" height="10"><rect x="0" y="0" width="24" height="10"
                        fill="#C49A6C" opacity="0.5" stroke="#C49A6C" stroke-width="1" rx="2"/></svg>
                    <span>Playa</span>
                </div>
            """)

        early = date_min.strftime("%b %Y") if date_min else "Early"
        late = date_max.strftime("%b %Y") if date_max else "Late"

        sections.append(f"""
        <div class="ls">
        <div class="lt">Crest lines &amp; Playa</div>
        {"".join(legend_items)}
        <div style="display:flex;align-items:center;gap:3px;margin:4px 0 0 0;">
            <span style="font-size:8px;color:#6BAED6;">{early}</span>
            <div style="flex:1;height:6px;background:linear-gradient(to left,
            #08306B, #08519C, #2171B5, #4292C6, #6BAED6, #9ECAE1, #DEEBF7);border-radius:2px;"></div>
            <span style="font-size:8px;color: #08306B;">{late}</span>
        </div>
        </div>""")

    if show_movement:
        sections.append("""
        <div class="ls">
        <div class="lt">Crest Movement (arrows)</div>
            <div style="display:flex;align-items:center;gap:3px;margin:3px 0;width:100%;">
                <span style="font-size:8px;color: #99BAFF;">left</span>
                <div style="flex:1;height:6px;background:linear-gradient(to right,
                #99BAFF,#1A2F6B,#191919,#5C2016,#FFB1A6);border-radius:2px;"></div>
                <span style="font-size:8px;color:#FFB1A6;">right</span>
            </div>
        </div>""")

    if show_uncertainty or show_margin_buffer:
        error_items = []
        if show_uncertainty:
            error_items.append("""
            <div class="lr" style="margin-top:3px;"><span>MEASURED DISPLACEMENT ERROR</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#31A857;"></div><span>&lt; 2 m</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#F7E62C;"></div><span>2-6 m</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#C7400F;"></div><span>&gt; 6 m</span></div>""")
        if show_margin_buffer:
            if show_uncertainty:
                error_items.append('<div style="border-top:1px dashed #D9DADB;margin:4px 0;"></div>')
            error_items.append(f"""
            <div class="lr"><span>MARGIN OF ERROR (95% CI)</span></div>
            <div class="lr">
                <svg width="16" height="10"><rect x="0" y="0" width="16" height="10"
                    fill="#C61826" opacity="0.5" stroke="#C61826" stroke-width="1"/></svg>
                <span>±{REPRESENTATIVE_MARGIN_OF_ERROR_M:.1f} m buffer based on March 2026 Epoch</span>
            </div>""")
        sections.append(f"""
        <div class="ls">
        <div class="lt">ERROR ASSESSMENT</div>
        {"".join(error_items)}
        </div>""")

    if show_gnss_points or show_gnss_lines:
        gnss_items = []
        if show_gnss_points:
            gnss_items.append("""
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#FF6B6B;"></div><span>GNSS Points (P3, P4)</span></div>
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#4ECDC4;"></div><span>GNSS Points (CCP)</span></div>
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#FFD93D;"></div><span>GNSS Points (Crest)</span></div>
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#6C5CE7;"></div><span>GNSS Points (Edge)</span></div>
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#A8E6CF;"></div><span>GNSS Points (Bowl)</span></div>""")
        if show_gnss_lines:
            gnss_items.append("""
            <div class="lr"><div style="width:14px;height:3px;background:#FFD93D;"></div><span>GNSS Crest Line</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#6C5CE7;"></div><span>GNSS Edge Line</span></div>
            <div class="lr"><div style="width:14px;height:3px;background:#A8E6CF;"></div><span>GNSS Bowl Line</span></div>""")
        sections.append(f"""
        <div class="ls">
        <div class="lt">GNSS Field Data</div>
        {"".join(gnss_items)}
        </div>""")

    if show_geomorph_lines or show_geomorph_points or show_geomorph_polygons:
        geomorph_items = []
        if show_geomorph_lines:
            geomorph_items.append("""
            <div class="lr"><div style="width:14px;height:3px;background:#201C19;"></div><span>Erosion fossil dune / vlei deposits</span></div>""")
        if show_geomorph_points:
            geomorph_items.append("""
            <div class="lr"><div style="width:14px;height:14px;border-radius:50%;background:#FD6F17;"></div><span>Sediment Sample Points</span></div>""")
        if show_geomorph_polygons:
            geomorph_items.append("""
            <div class="lr"><div style="width:14px;height:14px;background:#6C5CE7;opacity:0.5;border:1px solid #6C5CE7;"></div><span>Fossil Dune</span></div>
            <div class="lr"><div style="width:14px;height:14px;background:#A8E6CF;opacity:0.5;border:1px solid #A8E6CF;"></div><span>Recent Vlei</span></div>""")
        sections.append(f"""
        <div class="ls">
        <div class="lt">Geomorphology</div>
        {"".join(geomorph_items)}
        </div>""")

    if show_base_imagery and base_img_date:
        sections.append(f"""
        <div class="ls">
            <div class="lt">Base Imagery</div>
            <div class="lr"><span>{base_img_date}</span></div>
        </div>""")

    if show_hobo_wind and hobo_df is not None:
        sections.append("""
        <div class="ls">
            <div class="lt">SOS 1 WEST Weather Station</div>
            <div class="lr"><span>Wind Rose (March 2026)</span></div>
            <div style="display:flex;align-items:center;gap:3px;margin:4px 0 0 0;">
                <span style="font-size:7px;color:#5C3D1E;">0</span>
                <div style="flex:1;height:5px;background:linear-gradient(to right,#FEE5D9, #FCAE91, #FB6A4A, #DE2D26, #A50F15, #67000D);border-radius:2px;"></div>
                <span style="font-size:7px;color:#5C3D1E;">10+</span>
            </div>
        </div>""")

    if sections:
        legend_html = f"""
        <style>
        #dl {{ position:fixed;bottom:50px;left:10px;z-index:9998;font-family:sans-serif;font-size:10px;color:{MPL_FG}; }}
        #db {{ background:rgba(235,232,232,0.92);border:1px solid {MPL_GRID};border-radius:7px;padding:7px 9px;min-width:160px;max-width:200px;box-shadow:0 2px 8px rgba(0,0,0,0.15); }}
        #dt {{ cursor:pointer;user-select:none;font-weight:700;font-size:11px;color:{MPL_ACCENT};display:flex;justify-content:space-between;align-items:center;font-family:sans-serif;letter-spacing:.04em; }}
        #dt:hover {{ color:{MPL_ACCENT2}; }}
        #dc {{ margin-top:6px; }}
        .ls {{ margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid {MPL_GRID}; }}
        .ls:last-child {{ border-bottom:none;margin-bottom:0; }}
        .lt {{ font-weight:700;font-size:9px;color:{MPL_ACCENT};margin-bottom:3px;text-transform:uppercase;letter-spacing:.06em;font-family:sans-serif; }}
        .lr {{ display:flex;align-items:center;gap:5px;margin-bottom:2px;font-size:9px;font-family:sans-serif;color:{MPL_FG}; }}
        </style>
        <div id="dl"><div id="db">
        <div id="dt" onclick="var c=document.getElementById('dc');var a=document.getElementById('da');if(c.style.display==='none'){{c.style.display='block';a.textContent='▼';}}else{{c.style.display='none';a.textContent='►';}}">
            LEGEND <span id="da">▼</span>
        </div>
        <div id="dc">{"".join(sections)}</div>
        </div></div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ------------------------------------------------------------------------------
# CHART HELPERS
# ------------------------------------------------------------------------------

def movement_trend_fig(var_gdf, nearest_pid):
    trend = var_gdf[var_gdf["point_id"] == nearest_pid].sort_values("date")
    fig, ax = _dark_fig(3.6, 1.9)
    mask_winter = trend["date"].dt.month.isin([4, 5, 6, 7, 8, 9])
    mask_summer = trend["date"].dt.month.isin([10, 11, 12, 1, 2, 3])

    ax.scatter(trend.loc[mask_winter, "date"], trend.loc[mask_winter, "distance_m"],
               color="#0065BD", s=5, marker='d', zorder=5, label="Winter (Apr-Sep)")
    ax.scatter(trend.loc[mask_summer, "date"], trend.loc[mask_summer, "distance_m"],
               color="#C61826", s=5, marker='o', zorder=5, label="Summer (Oct-Mar)")
    ax.plot(trend["date"], trend["distance_m"], marker="", color=MPL_GRID, linewidth=1, alpha=0.5)
    ax.axhline(0, color=MPL_GRID, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date", fontsize=7, color=MPL_FG)
    ax.set_ylabel("Distance (m)", fontsize=7, color=MPL_FG)
    ax.set_title(f"Point {nearest_pid}", fontsize=8, color=MPL_ACCENT)
    ax.grid(color=MPL_GRID, linewidth=0.4, linestyle=":")
    ax.legend(fontsize=6, loc="best")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(pad=0.4)
    return fig

# ------------------------------------------------------------------------------
# RENDER FUNCTIONS
# ------------------------------------------------------------------------------

def render_dashboard_layout_1(map_col, right_col):
    if "map_center" not in st.session_state:
        st.session_state.map_center = MAP_CENTER
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = MAP_ZOOM
    if "last_checked_state" not in st.session_state:
        st.session_state.last_checked_state = {}

    def safe_load(fn, label):
        try:
            return fn()
        except Exception as e:
            st.error(f"Could not load {label}: {e}")
            return gpd.GeoDataFrame()

    crest_gdf = safe_load(load_crest_lines, "crest lines")
    var_gdf = safe_load(load_movement_points, "movement points")
    playa_gdf = safe_load(load_playa_polygons, "playa (Highest Purity)")
    unc_gdf = safe_load(load_uncertainty_lines, "uncertainty lines")
    gnss_points_gdf = safe_load(load_gnss_points, "GNSS points")
    gnss_lines_gdf = safe_load(load_gnss_lines, "GNSS lines")
    geomorph_data = safe_load(load_geomorph_layers, "geomorphology layers")

    try:
        wind_df = load_wind_data()
    except Exception as e:
        st.error(f"Could not load wind data: {e}")
        wind_df = pd.DataFrame()

    try:
        hobo_df = load_hobo_wind_data()
    except Exception as e:
        hobo_df = None
        st.warning(f"Could not load SOS 1 WEST data: {e}")

    base_metadata = load_base_imagery_metadata()

    if "dune_names" not in st.session_state:
        st.session_state["dune_names"] = (
            sorted(crest_gdf["dune_name"].dropna().unique())
            if "dune_name" in crest_gdf.columns else []
        )

    date_a = None
    date_b = None
    wind_pct = 0

    with st.sidebar:
        st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin-bottom:8px;">Presets</div>', unsafe_allow_html=True)

        preset = st.radio(
            "Select View Mode",
            ["Compare", "Annual", "Monthly", "Custom"],
            key="b_preset",
            label_visibility="collapsed",
            horizontal=True
        )

        if preset == "Annual":
            c1, c2 = st.columns(2)
            with c1:
                selected_year = st.selectbox("Year", options=sorted(crest_gdf["year"].unique()), key="b_annual_year")
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:#050505;margin-top:20px;">Select Months (2-6)</p>', unsafe_allow_html=True)

            month_range = st.select_slider(
                "Months", options=MONTH_NAMES,
                value=(MONTH_NAMES[4], MONTH_NAMES[7]),
                key="b_annual_month_range", label_visibility="collapsed"
            )
            start_idx = MONTH_NAMES.index(month_range[0])
            end_idx = MONTH_NAMES.index(month_range[1])
            selected_months = MONTH_NAMES[start_idx:end_idx + 1]
            selected_years = [selected_year]

            if len(selected_months) > 6:
                st.warning("Maximum 6 months allowed. Adjusting to 6 months.")
                selected_months = selected_months[:6]

        elif preset == "Monthly":
            c1, c2 = st.columns(2)
            with c1:
                selected_month = st.selectbox(
                    "Month", options=MONTH_NAMES, index=MONTH_NAMES.index("May"),
                    format_func=month_abbr, key="b_monthly_month"
                )
            with c2:
                st.markdown('<p style="font-size:0.7rem;color:#050505;margin-top:20px;">Select Years (2-6)</p>', unsafe_allow_html=True)

            sorted_years = sorted(ALL_YEARS)
            year_range = st.select_slider(
                "Years", options=sorted_years,
                value=(sorted_years[-5], sorted_years[-1]),
                key="b_monthly_year_range", label_visibility="collapsed"
            )
            start_idx = sorted_years.index(year_range[0])
            end_idx = sorted_years.index(year_range[1])
            selected_years = sorted_years[start_idx:end_idx + 1]
            selected_months = [selected_month]

            if len(selected_years) > 6:
                st.warning("Maximum 6 years allowed. Adjusting to 6 years.")
                selected_years = selected_years[:6]

        elif preset == "Compare":
            date_options = sorted(crest_gdf["date"].unique())
            date_strings = [d.strftime("%Y-%m-%d") for d in date_options]

            date_range = st.select_slider(
                "Select Date Range", options=date_strings,
                value=(date_strings[0], date_strings[-1] if len(date_strings) > 1 else date_strings[0]),
                key="b_compare_date_range"
            )

            date_a = pd.to_datetime(date_range[0])
            date_b = pd.to_datetime(date_range[1])

            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;background:#D9DADB;padding:8px 12px;border-radius:4px;border:1px solid #D9DADB;margin-top:4px;font-family:sans-serif;">
                    <span style="font-size:0.9rem;color:#0065BD;font-weight:600;">{date_a.strftime("%b %Y")}</span>
                    <span style="color:#050505;">→</span>
                    <span style="font-size:0.9rem;color:#0065BD;font-weight:600;">{date_b.strftime("%b %Y")}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if date_a == date_b:
                st.info("Select two different dates for comparison.")

            selected_years = list(set([date_a.year, date_b.year]))
            selected_months = list(set([date_a.strftime("%B"), date_b.strftime("%B")]))
            wind_pct, f_wind = wind_completeness(wind_df, date_a=date_a, date_b=date_b)

        else:
            c1, c2 = st.columns(2)
            with c1:
                sorted_years = sorted(crest_gdf["year"].unique()) if not crest_gdf.empty else ALL_YEARS
                year_range = st.select_slider(
                    "Select Year Range",
                    options=sorted_years,
                    value=(sorted_years[0], sorted_years[-1] if len(sorted_years) > 1 else sorted_years[0]),
                    key="b_custom_year_range"
                )
                start_year_idx = sorted_years.index(year_range[0])
                end_year_idx = sorted_years.index(year_range[1])
                selected_years = sorted_years[start_year_idx:end_year_idx + 1]

            with c2:
                month_range = st.select_slider(
                    "Select Month Range",
                    options=MONTH_NAMES,
                    value=(MONTH_NAMES[4], MONTH_NAMES[7]),
                    key="b_custom_month_range"
                )
                start_idx = MONTH_NAMES.index(month_range[0])
                end_idx = MONTH_NAMES.index(month_range[1])
                selected_months = MONTH_NAMES[start_idx:end_idx + 1]

            date_a = None
            date_b = None
            wind_pct = 0
            f_wind = pd.DataFrame()

        st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin:12px 0 8px 0;">Zoom to Feature</div>', unsafe_allow_html=True)
        dune_names = st.session_state.get("dune_names", [])
        DEFAULT_DUNE = "The Star Dune"

        zoom_options = ["All Features"] + dune_names
        default_index = zoom_options.index(DEFAULT_DUNE) if DEFAULT_DUNE in zoom_options else 0

        zoom_to = st.selectbox(
            "Zoom to feature", zoom_options,
            index=default_index, label_visibility="collapsed", key="b_zoom_select"
        )
        st.session_state["zoom_to"] = zoom_to

        st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin:12px 0 8px 0;">Layers</div>', unsafe_allow_html=True)
        with st.expander("Remote Sensing Layers", expanded=True):
            show_crests = st.checkbox("Crest lines", value=True, key="b_show_crests")
            show_gap_fills = st.checkbox("Gap fills", value=True, disabled=not show_crests, key="b_show_gap_fills")
            show_movement = st.checkbox("Crest Movement (arrows)", value=True, key="b_show_movement") if preset == "Compare" else False
            show_playa = st.checkbox("Playa (97th Percentile SI)", value=False, key="b_show_playa")
            show_uncertainty = st.checkbox("Displacement Error Lines (Only March 2026)", value=False, key="b_show_uncertainty")
            show_margin_buffer = st.checkbox("Margin of Error Buffer (95% CI)", value=True, key="b_show_margin_buffer")
            show_base_imagery = st.checkbox("Base Imagery (Sentinel-2)", value=True, key="b_show_base_imagery")

        with st.expander("Wind Data Layers", expanded=False):
            show_wind = st.checkbox("Wind rose overlay (Dieprivier Station ~ 95 km away)", value=True, key="b_show_wind")
            show_hobo_wind = st.checkbox("SOS 1 WEST Weather Station Wind Rose (16 - 20 March 2026)", value=False, key="b_show_hobo_wind")

        with st.expander("In-situ Layers", expanded=False):
            st.markdown('<div style="font-weight:600;font-size:12px;color:#C61826;font-family:sans-serif;margin:4px 0 8px 0;">MARCH 2026</div>', unsafe_allow_html=True)
            show_gnss_points = st.checkbox("GNSS Survey Points", value=False, key="b_show_gnss_points")
            show_gnss_lines = st.checkbox("GNSS Crest/Edge/Bowl Lines", value=False, key="b_show_gnss_lines")
            show_geomorph_lines = st.checkbox("Erosion fossil dune / vlei deposits", value=False, key="b_show_geomorph_lines")
            show_geomorph_points = st.checkbox("Sediment Sample Points", value=False, key="b_show_geomorph_points")
            show_geomorph_polygons = st.checkbox("Recent Vlei / Fossil Dunes", value=False, key="b_show_geomorph_polygons")

        current_state = {
            "show_crests": show_crests, "show_gap_fills": show_gap_fills,
            "show_movement": show_movement, "show_playa": show_playa,
            "show_wind": show_wind, "show_uncertainty": show_uncertainty,
            "show_margin_buffer": show_margin_buffer, "show_base_imagery": show_base_imagery,
            "show_gnss_points": show_gnss_points, "show_gnss_lines": show_gnss_lines,
            "show_geomorph_lines": show_geomorph_lines, "show_geomorph_points": show_geomorph_points,
            "show_geomorph_polygons": show_geomorph_polygons, "show_hobo_wind": show_hobo_wind,
        }
        st.session_state.last_checked_state = current_state

        st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin:12px 0 8px 0;">Opacity</div>', unsafe_allow_html=True)
        opacity = st.slider("Layer opacity", 0.2, 1.0, 0.75, 0.05, label_visibility="collapsed", key="b_opacity_slider")

        if st.button("Refresh Data (Clear Cache)", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if preset == "Compare":
        f_crest = crest_gdf[crest_gdf["date"].isin([date_a, date_b])].copy()
        f_playa = playa_gdf[playa_gdf["date"].isin([date_a, date_b])].copy()
        f_var = var_gdf[var_gdf["date"].isin([date_a, date_b])].copy()
        wind_b64 = None
        if show_wind and not f_wind.empty and wind_pct >= WIND_HIDE_PCT:
            wind_b64 = build_wind_rose_image(f_wind)
        elif show_wind and wind_pct < WIND_HIDE_PCT and not f_wind.empty:
            st.warning("Wind rose hidden: data coverage below 30% for selected period.")

    elif preset in ["Annual", "Monthly"]:
        f_crest = date_filter(crest_gdf, selected_years, selected_months)
        f_playa = date_filter(playa_gdf, selected_years, selected_months)
        f_var = var_gdf.copy()

        wind_b64 = None
        wind_pct = 0
        wind_pct_individual = 0
        date_a = None
        date_b = None

        if show_wind and not f_crest.empty:
            date_a = f_crest["date"].min()
            date_b = f_crest["date"].max()
            wind_sub = wind_df[(wind_df["datetime"] >= pd.Timestamp(date_a)) &
                            (wind_df["datetime"] <= pd.Timestamp(date_b))]
            if not wind_sub.empty and wind_sub["direction"].notna().sum() >= 5:
                wind_b64 = build_wind_rose_image(wind_sub)
                wind_pct, _ = wind_completeness(wind_df, date_a=date_a, date_b=date_b)
            elif not wind_sub.empty:
                st.warning("Wind rose hidden: insufficient wind data for selected period.")

            wind_pct_individual, f_wind = wind_completeness(wind_df, selected_years, selected_months)

    else:
        f_crest = date_filter(crest_gdf, selected_years, selected_months)
        f_playa = date_filter(playa_gdf, selected_years, selected_months)
        f_var = var_gdf.copy()

        wind_b64 = None
        wind_pct = 0
        date_a = None
        date_b = None

    date_min = f_crest["date"].min() if not f_crest.empty and "date" in f_crest.columns else None
    date_max = f_crest["date"].max() if not f_crest.empty and "date" in f_crest.columns else None

    with map_col:
        folium_map = build_map(
            crest_gdf=f_crest, var_gdf=f_var, playa_gdf=f_playa, unc_gdf=unc_gdf,
            wind_b64=wind_b64, wind_completeness_pct=wind_pct,
            show_crests=show_crests, show_gap_fills=show_gap_fills,
            show_movement=show_movement, date_a=date_a, date_b=date_b,
            show_playa=show_playa, show_wind=show_wind, show_uncertainty=show_uncertainty,
            show_margin_buffer=show_margin_buffer, opacity=opacity,
            date_min=date_min, date_max=date_max,
            show_base_imagery=show_base_imagery, base_metadata=base_metadata,
            selected_years=selected_years, selected_months=selected_months, preset=preset,
            show_gnss_points=show_gnss_points, show_gnss_lines=show_gnss_lines,
            show_geomorph_lines=show_geomorph_lines, show_geomorph_points=show_geomorph_points,
            show_geomorph_polygons=show_geomorph_polygons,
            gnss_points_gdf=gnss_points_gdf, gnss_lines_gdf=gnss_lines_gdf,
            geomorph_data=geomorph_data,
            hobo_df=hobo_df, show_hobo_wind=show_hobo_wind,
            custom_center=st.session_state.map_center, custom_zoom=st.session_state.map_zoom,
        )

        zoom_to = st.session_state.get("zoom_to", "All Features")
        if zoom_to != "All Features" and not f_crest.empty and "dune_name" in f_crest.columns:
            geoms = f_crest[f_crest["dune_name"] == zoom_to].geometry
            if not geoms.empty:
                b = geoms.total_bounds
                folium_map.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

        map_data = st_folium(
            folium_map, width="100%", height=600,
            returned_objects=["last_object_clicked"], key=f"b_folium_map_{show_wind}"
        )

        if map_data and "center" in map_data and "zoom" in map_data:
            st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
            st.session_state.map_zoom = map_data["zoom"]

    with right_col:
        if preset in ["Annual", "Monthly"]:
            if f_crest.empty:
                st.caption("No crest data available for wind roses.")
            elif wind_df.empty:
                st.caption("No wind data available.")
            else:
                pairs = get_wind_rose_pairs(f_crest, preset, selected_years, selected_months)
                if not pairs:
                    st.caption("Not enough consecutive dates to show wind roses.")
                else:
                    st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin-bottom:8px;">Wind Conditions Across Consecutive Epochs</div>', unsafe_allow_html=True)
                    cols = st.columns(2)
                    roses_shown = 0
                    for idx, (date_a_pair, date_b_pair, label) in enumerate(pairs):
                        img_b64, has_data = build_simple_wind_rose(wind_df, date_a_pair, date_b_pair)
                        col_idx = idx % 2
                        with cols[col_idx]:
                            st.markdown(f'<div style="text-align:center;font-size:0.6rem;color:#050505;font-weight:600;font-family:sans-serif;">{label}</div>', unsafe_allow_html=True)
                            if has_data and img_b64:
                                st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
                                roses_shown += 1
                            else:
                                st.markdown('<div style="text-align:center;font-size:0.55rem;color:#8B7A6A;padding:10px 0;border:1px dashed #D9DADB;border-radius:4px;font-family:sans-serif;">No wind data</div>', unsafe_allow_html=True)
                        if idx % 2 == 1:
                            cols = st.columns(2)

                    if roses_shown > 0:
                        st.markdown("""
                            <div style="font-size:7px;color:#050505;font-weight:600;text-align:center;font-family:sans-serif;">Wind speed (m/s)</div>
                            <div style="display:flex;align-items:center;gap:4px;padding:0 5px;">
                                <span style="font-size:6px;">0</span>
                                <div style="flex:1;height:5px;background:linear-gradient(to right,#FEE5D9, #FCAE91, #FB6A4A, #DE2D26, #A50F15, #67000D);border-radius:2px;"></div>
                                <span style="font-size:6px;">10+</span>
                            </div>
                        """, unsafe_allow_html=True)

                    if roses_shown == 0:
                        st.caption("No wind data available for any period.")

        elif preset == "Compare":
            st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin-bottom:8px;">Movement Trend</div>', unsafe_allow_html=True)
            if (map_data and map_data.get("last_object_clicked") and show_movement and not var_gdf.empty):
                click = map_data["last_object_clicked"]
                lat, lon = click.get("lat"), click.get("lng")
                if lat and lon:
                    click_pt = Point(lon, lat)
                    var_proj = var_gdf.to_crs("EPSG:3857")
                    click_gdf = gpd.GeoDataFrame(geometry=[click_pt], crs="EPSG:4326").to_crs("EPSG:3857")
                    dists = var_proj.geometry.distance(click_gdf.geometry.iloc[0])
                    nearest_pid = var_gdf.iloc[dists.idxmin()]["point_id"]
                    fig_ts = movement_trend_fig(var_gdf, nearest_pid)
                    st.pyplot(fig_ts, use_container_width=True)
                    plt.close(fig_ts)
                else:
                    st.caption("Click an arrow on the map.")
            else:
                st.caption("Click an arrow on the map. (Only available in Compare preset)")

        st.markdown('<div style="font-weight:700;font-size:14px;color:#0065BD;font-family:sans-serif;margin-bottom:8px;">Wind Coverage</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#050505;font-family:sans-serif;margin-bottom:3px;">Dieprivier station ~95km away</div>', unsafe_allow_html=True)
        if not wind_df.empty:
            try:
                fig_g = build_gantt_figure(wind_df, ALL_YEARS, MONTH_NAMES)
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)
                display_pct = wind_pct if preset == "Compare" else (wind_pct_individual if 'wind_pct_individual' in locals() else 0)
                if display_pct < WIND_WARN_PCT and display_pct > 0:
                    st.markdown(f'<div style="background:#FFF3CD;color:#050505;border:1px solid #C61826;margin:8px 12px;padding:8px 12px;border-radius:4px;font-size:0.8rem;font-weight:bold;font-family:sans-serif;">! {display_pct*100:.0f}% coverage</div>', unsafe_allow_html=True)
            except Exception:
                st.caption("Wind coverage chart unavailable for this selection.")
        else:
            st.caption("No wind data loaded.")

        st.download_button(
            "Download Map (HTML)",
            data=folium_map._repr_html_(),
            file_name="dune_map.html",
            mime="text/html",
            use_container_width=True,
            key="b_download_map"
        )

        if not f_crest.empty:
            st.download_button(
                "Download Crests (CSV)",
                data=f_crest.drop(columns="geometry").to_csv(index=False),
                file_name="crest_lines_filtered.csv",
                mime="text/csv",
                use_container_width=True,
                key="b_download_crests"
            )

# ------------------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------------------

def main():
    title_col, logo_l, logo_r  = st.columns([7, 1, 1], vertical_alignment="center")
    with logo_l:
        st.image(LOGO_HEIDELBERG, width=100)
    with logo_r:
        st.image(LOGO_TUM, width=90)
    with title_col:
        st.markdown('<div style="font-size:2rem;color:#0065BD; font-weight:bold">Cartographic Dashboard for Star Dune Dynamics Visualization</div>', unsafe_allow_html=True)

    st.divider()

    map_col, right_col = st.columns([4, 1.3])
    render_dashboard_layout_1(map_col, right_col)

    st.divider()

    st.markdown(f"""
            <div style="
                        font-size:11px;
                        color:{MPL_FG};
                        font-family:sans-serif;
                        text-align:center;">
                © 2026 {AUTHOR_NAME} · MSc Cartography Thesis Project.<br>
                This work is part of the Star Dune Dynamics Project funded by the 
                German Research Foundation (DFG, project number 551866032).<br> 
                Find more info on the project website:
                <a href="{PROJECT_URL}" target="_blank" style="color:#0065BD;">Star Dune Dynamics</a><br>
                <span style="font-size:10px; color:#6C757D;">
                Data sources: 
                <a href="https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2" 
                target="_blank" 
                style="color:#0065BD;">Copernicus Sentinel-2</a> · 
                <a href="https://www.sasscalweathernet.com/" 
                target="_blank" 
                style="color:#0065BD;">SASSCAL WeatherNet (2020)</a> · 
                <a href="{PROJECT_URL}" 
                target="_blank" 
                style="color:#0065BD;">GNSS Field Data</a>
                </span>
                <br>
                <span style="font-size:9px; color:#6C757D;">
                Contains modified Copernicus Sentinel data 2026 · 
                SASSCAL WeatherNet data used under license · 
                GNSS data collected under DFG project 551866032
                </span>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()