"""Reusable loader for exposure-normalized RGB + cached depth maps."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms

from REEF.reef_io import depth_paths, normalized_image_path

NATIVE_SIZE = 518


def resize_shorter_then_center_crop(img: Image.Image, size: int = NATIVE_SIZE) -> Image.Image:
    """Resize shorter side to size, then center-crop to size x size."""
    return transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
        ]
    )(img)


class RGBDImageLoader:
    """Load exposure-normalized RGB and cached Depth-Anything-V2 maps."""

    def __init__(self, campaign: str, deployment: str, native_size: int = NATIVE_SIZE, metric=False, processed=False):
        self.campaign = campaign
        self.deployment = deployment
        self.native_size = native_size
        self.label = "metric" if metric else "processed" if processed else None

        print(f"Initializing RGBDImageLoader for {campaign}/{deployment} with label={self.label}...")

        npy_path, keys_path = depth_paths(campaign, deployment, self.label)
        if not npy_path.exists() or not keys_path.exists():
            raise FileNotFoundError(
                f"Missing depth cache for {campaign}/{deployment}: {npy_path}"
            )

        self._depths = np.load(npy_path, mmap_mode="r")
        self._keys = np.load(keys_path)
        self._key_to_idx = {str(k): i for i, k in enumerate(self._keys)}

    def list_keys(self) -> list[str]:
        return [str(k) for k in self._keys]

    def key_from_index(self, index: int) -> str:
        if index < 0 or index >= len(self._keys):
            raise IndexError(f"Index {index} out of range [0, {len(self._keys) - 1}]")
        return str(self._keys[index])

    def load(self, key: str, align_rgb: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._key_to_idx:
            raise KeyError(f"Key not found in depth cache: {key}")

        rgb_path = normalized_image_path(self.campaign, self.deployment, key)
        with Image.open(rgb_path) as img:
            img = img.convert("RGB")
            if align_rgb:
                img = resize_shorter_then_center_crop(img, self.native_size)
            rgb = np.asarray(img, dtype=np.uint8)

        idx = self._key_to_idx[key]
        depth = np.asarray(self._depths[idx], dtype=np.float32)

        if depth.shape != (self.native_size, self.native_size):
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize(
                (self.native_size, self.native_size), resample=Image.BILINEAR
            )
            depth = np.asarray(depth_img, dtype=np.float32)

        return rgb, depth
