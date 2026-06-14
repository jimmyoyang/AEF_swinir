# -*- coding: utf-8 -*-
"""
Google Colab download reference for Landsat 8 TOA and AlphaEarth Annual.

This script is saved in the data pipeline folder as a reference only. It records
the exact Google Earth Engine export logic that was already run in Colab, so it
is easy to check later when debugging model normalization, band order, raster
dimensions, CRS, and LR/HR scale alignment.

Important notes for later normalization checks:
    - Landsat is exported at 30 m and is used as LR input.
    - AlphaEarth is exported at 10 m and is used as HR target.
    - The intended super-resolution scale factor is 3:
          HR height/width should be about 3x LR height/width.
    - Landsat bands exported here are:
          B1, B2, B3, B4, B5, B6, B7, B10, B11, CloudMask_Soft
      so LR has 10 channels after adding the soft cloud mask.
    - AlphaEarth bands are quantized to uint8 in [0, 254] by `quantize_aef`.
      Training code may later dequantize or normalize these values, so do not
      confuse this export-time quantization with Dataset-time normalization.
    - This is "sample-style" export: same scene geometry, same CRS, fixed scales.
      It does not explicitly lock a shared affine origin with crsTransform.

Colab setup reminder:
    # !pip -q install geemap earthengine-api
"""

import os
import sys
from datetime import datetime

import ee
import geemap
from google.colab import drive
from tqdm.notebook import tqdm


# =============================================================================
# 1. GEE / Google Drive initialization
# =============================================================================

PROJECT_ID = "gee-pro-486908"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

drive.mount("/content/drive", force_remount=True)
print("GEE and Google Drive initialized.")


# =============================================================================
# 2. Export parameters
# =============================================================================

# Landsat WRS-2 scene identifier. The raw file name pattern later uses 132033.
REF_PATH = 132
REF_ROW = 33

# The attached Colab-exported file name mentioned 2018, but the saved Colab
# parameter was 2024. Change only this value when re-running for another year.
YEAR_TO_PROCESS = 2024

# UTM zone used for this path/row region. Both LR and HR exports use this CRS.
FORCED_METER_CRS = "EPSG:32648"

# Google Drive folder names used by Export.image.toDrive. These names should
# correspond to the raw folders consumed later by datapipe/prepare_data_local.py.
GEE_LANDSAT_FOLDER = "raw_landsat"
GEE_ALPHA_FOLDER = "raw_alphaearth"

ALPHAEARTH_ASSET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

# Full AlphaEarth uses A00-A63. The Colab run saved here exported only A53-A63.
# For 64-band training, set this back to:
#     ALPHA_BANDS = [f"A{i:02d}" for i in range(64)]
ALPHA_BANDS = [f"A{i:02d}" for i in range(53, 64)]

# Local Drive folders are created only for human organization in Colab. GEE
# export still writes by Drive folder name, not by this absolute path.
LOCAL_ROOT = "/content/drive/MyDrive/data/Cloud_test/2024"
RAW_LANDSAT_DIR = os.path.join(LOCAL_ROOT, GEE_LANDSAT_FOLDER)
RAW_ALPHA_DIR = os.path.join(LOCAL_ROOT, GEE_ALPHA_FOLDER)
os.makedirs(RAW_LANDSAT_DIR, exist_ok=True)
os.makedirs(RAW_ALPHA_DIR, exist_ok=True)

print("REF_PATH, REF_ROW, YEAR:", REF_PATH, REF_ROW, YEAR_TO_PROCESS)
print("FORCED_METER_CRS:", FORCED_METER_CRS)
print("Landsat folder:", RAW_LANDSAT_DIR)
print("AlphaEarth folder:", RAW_ALPHA_DIR)


# =============================================================================
# 3. Helper functions
# =============================================================================

def get_landsat_collection(path, row, year):
    """Return all Landsat 8 Collection 2 TOA images for one WRS path/row/year."""
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
        .filter(ee.Filter.eq("WRS_PATH", path))
        .filter(ee.Filter.eq("WRS_ROW", row))
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .sort("system:time_start")
    )


def get_ultimate_soft_mask(image):
    """
    Build a soft cloud mask from Landsat QA confidence bits.

    Output range:
        0.0 means low cloud/shadow/cirrus confidence.
        1.0 means highest cloud/shadow/cirrus confidence.

    This band is appended to the 9 Landsat spectral/thermal bands, becoming the
    10th LR channel. Dataset normalization later treats it as part of LR input.
    """
    qa = image.select("QA_PIXEL")

    cloud_conf = qa.rightShift(8).bitwiseAnd(3)
    shadow_conf = qa.rightShift(10).bitwiseAnd(3)
    cirrus_conf = qa.rightShift(14).bitwiseAnd(3)

    max_conf = cloud_conf.max(shadow_conf).max(cirrus_conf)
    return max_conf.toFloat().divide(3.0).rename("CloudMask_Soft")


def quantize_aef(image):
    """
    Quantize AlphaEarth float embeddings to uint8.

    Formula used in this Colab export:
        sign(x) * abs(x) ** (1 / 2.0) * 127.5
        round, clamp to [-127, 127], then add 127

    Result:
        uint8-like values in [0, 254]. Value 127 is the signed zero center.

    This is export-time compression. If training uses `hr_normalization` or
    dequantization parameters, check datapipe/datasets.py as the second stage.
    """
    power = 2.0
    scale = 127.5
    min_v = -127
    max_v = 127

    sign = image.gt(0).toFloat().subtract(image.lt(0).toFloat())
    saturated = image.abs().pow(1.0 / power).multiply(sign)
    snapped = saturated.multiply(scale).round()

    return snapped.clamp(min_v, max_v).add(127).uint8()


# =============================================================================
# 4. Reference Landsat scene and whole-scene geometry
# =============================================================================

l8_coll = get_landsat_collection(REF_PATH, REF_ROW, YEAR_TO_PROCESS)
coll_size = l8_coll.size().getInfo()

if coll_size == 0:
    sys.exit("No Landsat 8 TOA images found. Check path/row/year.")

image_list = l8_coll.toList(coll_size)
first_l8 = ee.Image(image_list.get(0))
ref_date = ee.Date(first_l8.get("system:time_start")).format("YYYY-MM-dd").getInfo()

# Sample-style export: the first Landsat scene geometry is the common region.
tile_geom = first_l8.geometry()

print("Reference scene date:", ref_date)
print(f"Total Landsat scenes in {YEAR_TO_PROCESS}:", coll_size)
print("Using whole-scene geometry from the reference Landsat image.")

tile_bounds = tile_geom.bounds(maxError=1, proj=FORCED_METER_CRS)
print("Tile bounds (UTM):", tile_bounds.coordinates().getInfo())


# =============================================================================
# 5. AlphaEarth annual image
# =============================================================================

alpha_coll = (
    ee.ImageCollection(ALPHAEARTH_ASSET_ID)
    .filterDate(f"{YEAR_TO_PROCESS}-01-01", f"{YEAR_TO_PROCESS + 1}-01-01")
    .filterBounds(tile_geom)
)

alpha_count = alpha_coll.size().getInfo()
print("AlphaEarth image count after filtering:", alpha_count)

if alpha_count == 0:
    sys.exit("No AlphaEarth images found. Check asset id, year, or region.")

alpha_img = alpha_coll.mosaic().select(ALPHA_BANDS)
alpha_quantized = quantize_aef(alpha_img)

print("AlphaEarth exported band count:", alpha_quantized.bandNames().size().getInfo())
print("AlphaEarth exported bands:", ALPHA_BANDS)


# =============================================================================
# 6. Optional map preview
# =============================================================================

Map = geemap.Map()

landsat_preview = first_l8.select(["B4", "B3", "B2"]).clip(tile_geom)

# Preview bands must exist in ALPHA_BANDS. If exporting only A53-A63, use bands
# from that subset. For full A00-A63 export, A01/A31/A63 is also fine.
preview_bands = [ALPHA_BANDS[0], ALPHA_BANDS[len(ALPHA_BANDS) // 2], ALPHA_BANDS[-1]]
alpha_preview = alpha_quantized.select(preview_bands).clip(tile_geom)

center = tile_geom.centroid(maxError=1).transform("EPSG:4326", 1).coordinates().getInfo()
Map.setCenter(center[0], center[1], 8)

Map.addLayer(
    landsat_preview,
    {"bands": ["B4", "B3", "B2"], "min": 0.02, "max": 0.30},
    "Landsat RGB",
)
Map.addLayer(
    alpha_preview,
    {"bands": preview_bands, "min": 0, "max": 255},
    "AlphaEarth preview",
)
Map.addLayer(tile_geom, {}, "Reference tile geometry")

Map


# =============================================================================
# 7. Submit Landsat 30 m exports
# =============================================================================

dates = []
for i in range(coll_size):
    img = ee.Image(image_list.get(i))
    ts = img.get("system:time_start").getInfo()
    dates.append(datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d"))

landsat_tasks = []

for i, date in enumerate(tqdm(dates, desc="Submitting Landsat exports")):
    img = ee.Image(image_list.get(i))

    soft_mask = get_ultimate_soft_mask(img)
    img10 = img.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11"]).addBands(soft_mask)

    dt_str = date.replace("-", "")
    file_name = f"L8_{REF_PATH}{REF_ROW}_{dt_str}_Masked"

    task = ee.batch.Export.image.toDrive(
        image=img10.clip(tile_geom),
        description=file_name,
        folder=GEE_LANDSAT_FOLDER,
        fileNamePrefix=file_name,
        region=tile_geom.getInfo()["coordinates"],
        scale=30,
        crs=FORCED_METER_CRS,
        maxPixels=1e13,
    )
    task.start()
    landsat_tasks.append(task)
    print(f"Started Landsat export: {file_name}")

print(f"Landsat export tasks submitted: {len(landsat_tasks)}")


# =============================================================================
# 8. Submit AlphaEarth 10 m single-band exports
# =============================================================================

alpha_tasks = []

for band_name in tqdm(ALPHA_BANDS, desc="Submitting AlphaEarth exports"):
    alpha_single_band = alpha_quantized.select(band_name)
    file_name_alpha = f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{band_name}"

    task_alpha = ee.batch.Export.image.toDrive(
        image=alpha_single_band.clip(tile_geom),
        description=file_name_alpha,
        folder=GEE_ALPHA_FOLDER,
        fileNamePrefix=file_name_alpha,
        region=tile_geom.getInfo()["coordinates"],
        scale=10,
        crs=FORCED_METER_CRS,
        maxPixels=1e13,
    )
    task_alpha.start()
    alpha_tasks.append(task_alpha)
    print(f"Started AlphaEarth export: {file_name_alpha}")

print(f"AlphaEarth export tasks submitted: {len(alpha_tasks)}")


# =============================================================================
# 9. What to compare during later checks
# =============================================================================

"""
Use this checklist when debugging normalization or data dimensions:

1. Raw Landsat files:
       expected folder: raw_landsat
       expected channels: 10
       expected export scale: 30 m
       expected names: L8_13233_YYYYMMDD_Masked.tif

2. Raw AlphaEarth files:
       expected folder: raw_alphaearth
       expected channels before stacking: one tif per Axx band
       expected export scale: 10 m
       expected names: AlphaEarth_Path132_Row33_reprojected_Axx.tif

3. Prepare-data stage:
       datapipe/prepare_data_local.py stacks AlphaEarth bands and cuts LR/HR
       patches. With LR_PATCH_SIZE=64 and SCALE_FACTOR=3, HR patches should be
       192x192.

4. Dataset normalization stage:
       datapipe/datasets.py and datapipe/datasets_with_lr_norm.py decide how LR
       and HR tensors are mapped to the model range, usually [-1, 1].
"""
