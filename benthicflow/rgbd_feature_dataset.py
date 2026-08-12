"""Torch dataset for exposure-normalized RGB, cached depth, and cached features."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .reef_io import feature_path
from .rgbd_loader import NATIVE_SIZE, RGBDImageLoader


class RGBDFeatureDataset(Dataset):
    """Load RGB, depth, and cached features keyed by image id."""

    def __init__(
        self,
        campaign: str,
        deployment: str,
        native_size: int = NATIVE_SIZE,
        metric: bool = False,
        processed: bool = False,
        align_rgb: bool = True,
        as_tensor: bool = True,
        rgb_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        depth_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feature_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.campaign = campaign
        self.deployment = deployment
        self.align_rgb = align_rgb
        self.as_tensor = as_tensor
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.feature_transform = feature_transform

        self._loader = RGBDImageLoader(
            campaign,
            deployment,
            native_size=native_size,
            metric=metric,
            processed=processed,
        )

        feats_path = feature_path(campaign, deployment)
        if not feats_path.exists():
            raise FileNotFoundError(
                f"Missing feature cache for {campaign}/{deployment}: {feats_path}"
            )

        feat_npz = np.load(feats_path)
        self._features = feat_npz["features"]
        feat_keys = feat_npz["keys"].astype(str)
        self._feat_key_to_idx = {str(k): i for i, k in enumerate(feat_keys)}

        depth_keys = self._loader.list_keys()
        self._keys = [k for k in depth_keys if k in self._feat_key_to_idx]
        if not self._keys:
            raise SystemExit(
                "No overlapping keys between depth cache and feature cache."
            )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(
        self, index: int
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
        | Tuple[np.ndarray, np.ndarray, np.ndarray, str]
    ):
        key = self._keys[index]
        rgb, depth = self._loader.load(key, align_rgb=self.align_rgb)
        feat = np.asarray(self._features[self._feat_key_to_idx[key]], dtype=np.float32)

        if not self.as_tensor:
            return rgb, depth, feat, key

        rgb_t = torch.from_numpy(rgb)
        depth_t = torch.from_numpy(depth)
        feat_t = torch.from_numpy(feat)

        if self.rgb_transform is not None:
            rgb_t = self.rgb_transform(rgb_t)
        if self.depth_transform is not None:
            depth_t = self.depth_transform(depth_t)
        if self.feature_transform is not None:
            feat_t = self.feature_transform(feat_t)

        return rgb_t, depth_t, feat_t, key
