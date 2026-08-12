import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

FEATURE_ROOT = Path("features/splits")

# Find all .npz files in your feature tree
npz_files = list(FEATURE_ROOT.rglob("*.npz"))
print(f"Found {len(npz_files)} .npz archives to unpack.")

for npz_path in tqdm(npz_files):
    npy_path = npz_path.with_suffix(".npy")
    if not npy_path.exists():
        # Load fully into RAM once
        data = np.load(npz_path)["features"]
        # Save as raw, uncompressed binary
        np.save(npy_path, data)

print("Done. You can now delete the .npz files to save storage if desired.")
