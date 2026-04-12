# scripts/check_zero_hr_ratio.py
from pathlib import Path
import numpy as np
import rasterio

root = Path("data/processed_data")
for split in ["train", "val", "test"]:
    hr_dir = root / split / "HR"
    files = sorted(hr_dir.glob("*.tif"))
    zero_cnt = 0
    for p in files:
        with rasterio.open(p) as ds:
            arr = ds.read()
        if np.max(arr) == 0:
            zero_cnt += 1
    ratio = zero_cnt / max(len(files), 1)
    print(f"{split}: total={len(files)} zero_hr={zero_cnt} ratio={ratio:.4%}")