import glob
import rasterio
import numpy as np

hr_band_files = sorted(glob.glob('/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/raw_alphaearth/AlphaEarth_Path132_Row33_reprojected_A*.tif'))

for f in hr_band_files:
    with rasterio.open(f) as src:
        arr = src.read(1)
        print(f"{f}: min={np.min(arr)}, max={np.max(arr)}, dtype={arr.dtype}")