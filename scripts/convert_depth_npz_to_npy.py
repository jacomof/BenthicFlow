"""Migrate compressed depths.npz to uncompressed .npy for memory mapping."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

# Updated to match your exact directory structure
DEPTH_ROOT = Path("depth")


def migrate_all() -> None:
    # Matches depth/Campaign/Deployment/depths.npz
    npz_files = list(DEPTH_ROOT.glob("*/*/depths.npz"))
    print(
        f"Found {len(npz_files)} depth archives to migrate in {DEPTH_ROOT.resolve()}."
    )

    for npz_path in tqdm(npz_files):
        try:
            with np.load(npz_path) as z:
                depths = z["depths"]
                keys = z["keys"]

            npy_path = npz_path.with_name("depths.npy")
            keys_path = npz_path.with_name("depths_keys.npy")

            # Save as flat, uncompressed binaries
            np.save(npy_path, depths)
            np.save(keys_path, keys)

            # Optional: Uncomment below to delete the old .npz files once you confirm the .npy files are created
            # npz_path.unlink()
        except Exception as e:
            print(f"Failed on archive {npz_path}: {e}")


if __name__ == "__main__":
    migrate_all()
