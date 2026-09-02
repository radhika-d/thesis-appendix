# thesis-appendix

Supplementary material for the MSc Cartography thesis *[A Star in Motion: A Cartographic Approach to Remotely Sensed Star Dune and Environment Dynamics]* by Radhika Dhuri, TUM, 2026. This repository contains the code, data, and evaluation materials referenced in the thesis appendix. If you're new here, this file explains what's in the repository and how to use it.

## What this thesis is about

The thesis develops a framework for visualizing star dune dynamics in the Namib Desert (Sossusvlei, Namibia) using an interactive Streamlit dashboard. It processes Sentinel-2 imagery, GNSS field data, and wind station records into crest lines, playa polygons, and error assessments, then presents them on a monthly-navigable map.

## Live Dashboard

[\[Cartographic Dashboard for Star Dune Dynamics Visualization\]](https://cartographic-star-dune-dynamics-thesis.streamlit.app/)

## Repository Content

- **`data/`** — input and output data for the pipeline: raw satellite imagery, field survey files, wind station records, and processed outputs (crest lines, playa polygons, error statistics) that feed the dashboard.
- **`src/`** — the processing pipeline (numbered scripts 1-8) and the dashboard app (`app.py`).
- **`surveys/`** — evaluation materials: the questionnaire, survey responses, and the cleaned meeting transcript.
- **`LICENSE`**, **`requirements.txt`**, **`README.md`** — standard repo files.

## Running the Code

1. Install dependencies: `pip install -r requirements.txt`
2. Requires a Google Earth Engine account for step 2 (imagery download).
3. Run scripts in `src/` in numbered order, 1 through 8. Step 1 is optional if you're not using your own field data.
4. Launch the dashboard: `streamlit run src/app.py`

Each script reads the previous step's output from `data/`, so run them in sequence the first time.

## Pipeline Scripts

| Script | Function |
|---|---|
| `1_field_data_process.py` | Processes raw GNSS field survey points into crest, edge, and bowl lines |
| `2_tiff_acquisition.py` | Downloads monthly Sentinel-2 imagery via Google Earth Engine |
| `3_base_tif_processing.py` | Converts GeoTIFFs to PNG basemap overlays with metadata |
| `4_crest_extraction.py` | Extracts raw crest lines per image (Canny edge detection + centerline vectorization) |
| `5_crest_post_process.py` | Merges dates, gap-fills, builds movement points |
| `6_playa_extraction.py` | Extracts and merges playa polygons per date |
| `7_uncertainty_calculation.py` | Compares GNSS reference lines to detected crests, computes error statistics |
| `8_wind_data_combining.py` | Cleans and merges monthly wind station CSVs |
| `app.py` | Streamlit dashboard rendering all outputs above |

## Data Sources 01

| Dataset | Provider | Date Range | Access |
|---|---|---|---|
| Sentinel-2 Level-2A Surface Reflectance Harmonized (`COPERNICUS/S2_SR_HARMONIZED`), bands B2, B3, B4, B8, 10 m resolution | [European Space Agency](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2), Copernicus Programme, via Google Earth Engine | January 2017 to July 2026, monthly | Open access |
| GNSS field survey data | [Star Dune Dynamics project](https://www.asg.ed.tum.de/rsa/forschung/star-dune-dynamics/) fieldwork | March 2026 | Restricted, held by the project |
| Manually digitized reference lines (Big Mommy Dune, Inverted Y Dune) | Digitized in QGIS from a Sentinel-2 scene | August 2025 | Derived, not independently available |
| Wind data (Dieprivier station) | [SASSCAL WeatherNet (2020)](https://www.sasscalweathernet.com/) | Ongoing record | Used under license |
| Wind data (SOS 1 WEST on-site station) | [Star Dune Dynamics project](https://www.asg.ed.tum.de/rsa/forschung/star-dune-dynamics/) project fieldwork | 16-20 March 2026 | Restricted, held by the project |

## Survey Materials

- `Final_survey_google_form_answers.xlsx` — raw responses from the second expert-evaluation round
- `final_survey_google_form.pdf` — the full questionnaire as distributed
- `initial_meeting_feedback_transcript.pdf` — cleaned transcript of the first expert consultation, participants anonymized

## License

See `LICENSE`.

## Citation

If referencing this work, cite the thesis: Dhuri, R. (2026). *[A Star in Motion: A Cartographic Approach to Remotely Sensed Star Dune and Environment Dynamics]*. MSc Thesis, Technical University of Munich.
