#!/usr/bin/env python3
"""Generate a colored point cloud from an RGB image and a depth map.
This is a minimal RGBD -> point cloud utility intended to provide XYZ + RGB
for 3DGS or downstream reconstruction.
Examples:
python generate_3d.py --rgb img.jpg --depth depth.png --out out.ply
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from benthicflow import RGBDImageLoader, resize_shorter_then_center_crop
from benthicflow.reef_io import depth_paths


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to read RGB images. Install pillow."
        ) from exc

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = resize_shorter_then_center_crop(img)
        rgb = np.asarray(img, dtype=np.uint8)
    return rgb


def _load_depth(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to read depth images. Install pillow."
        ) from exc

    with Image.open(path) as img:
        depth = np.asarray(img)

    if depth.ndim != 2:
        raise ValueError(
            f"Depth image must be single-channel; got shape {depth.shape}."
        )

    if depth.dtype == np.uint8:
        raise ValueError(
            "Depth image appears to be 8-bit; provide a 16-bit or float depth map."
        )

    return depth.astype(np.float32)


def _load_depth_numpy(
    campaign: str, deployment: str, key: str, label: str
) -> np.ndarray:

    npy_path, keys_path = depth_paths(campaign, deployment, label)
    depths = np.load(npy_path, mmap_mode="r")
    keys = np.load(keys_path)
    print(f"Searching for key '{key}' in depth cache...")
    print(
        f"Loaded depth cache for {campaign}/{deployment} with label={label}: {len(keys)} entries."
    )
    _key_to_idx = {str(k): i for i, k in enumerate(keys)}
    if key not in _key_to_idx:
        raise KeyError(f"Key not found in depth cache: {key}")
    return depths[_key_to_idx[key]]


def _infer_intrinsics(
    width: int,
    height: int,
    focal_length_mm: float = 14.0,
    sensor_width_mm: float = 17.3,
) -> Tuple[float, float, float, float]:

    # Adjust focal length for underwater refraction index (~1.333)
    focal_length_underwater = focal_length_mm * 1.333

    fx = focal_length_underwater * (width / sensor_width_mm)
    fy = fx

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


import math


def _infer_intrinsics_v2(
    width: int,
    height: int,
    hfov_degrees: float = 42.0,  # Defaulting to a common Sirius in-water HFOV profile
    is_wet_fov: bool = True,
) -> Tuple[float, float, float, float]:

    # 1. If the FOV provided is a dry/air value, apply Snell's law to get the wet FOV
    if not is_wet_fov:
        # Refraction index of water ~1.333
        half_fov_air_rad = math.radians(hfov_degrees / 2.0)
        half_fov_water_rad = math.asin(math.sin(half_fov_air_rad) / 1.333)
        hfov_radians = half_fov_water_rad * 2.0
    else:
        # If it's already the wet FOV, simply convert degrees to radians
        hfov_radians = math.radians(hfov_degrees)

    # 2. Compute pixel focal length using the pinhole model equation: fx = width / (2 * tan(HFOV / 2))
    fx = width / (2.0 * math.tan(hfov_radians / 2.0))
    fy = fx  # Assuming square pixels, which is standard for machine vision

    # 3. Compute the principal point (optical center of the image frame)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    return fx, fy, cx, cy


def process_depth(depth_m: np.ndarray, min_depth=3.0, max_depth=4.5) -> np.ndarray:

    depth_norm = (depth_m - np.min(depth_m)) / (
        np.max(depth_m) - np.min(depth_m) + 1e-8
    )

    epsilon = 0.5
    linear_depth = 1.0 / (depth_norm + epsilon)

    linear_depth = (linear_depth - np.min(linear_depth)) / (
        np.max(linear_depth) - np.min(linear_depth)
    )

    return linear_depth * (max_depth - min_depth) + min_depth


def process_depth_v2(depth_m: np.ndarray, min_depth=3.0, max_depth=4.5) -> np.ndarray:
    # DepthAnything outputs high values for CLOSE objects, low values for FAR objects (disparity).
    # To turn this into true depth, we invert the raw values smoothly before scaling.

    # 1. Prevent divide-by-zero by adding a tiny float stabilizer
    raw_disparity = depth_m + 0.01

    # 2. Invert smoothly
    depth_inv = 1.0 / raw_disparity

    # 3. Min-Max normalize the inverted depth map
    depth_min = np.min(depth_inv)
    depth_max = np.max(depth_inv)
    depth_norm = (depth_inv - depth_min) / (depth_max - depth_min + 1e-6)

    # 4. Map to target metric meters
    return depth_norm * (max_depth - min_depth) + min_depth


def _normalize_points_unit_cube(
    points: np.ndarray, centered: bool = True
) -> np.ndarray:
    if points.size == 0:
        return points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.where((maxs - mins) > 1e-6, (maxs - mins), 1.0)
    normalized = (points - mins) / span
    if centered:
        normalized = normalized - 0.5
    return normalized


def print_depth_stats(depth: np.ndarray) -> None:
    valid = depth > 0.0
    print(f"Depth map shape: {depth.shape}, dtype: {depth.dtype}")
    if np.any(valid):
        print(
            f"Depth stats: min={float(depth[valid].min()):.3f} max={float(depth[valid].max()):.3f} "
            f"mean={float(depth[valid].mean()):.3f} std={float(depth[valid].std()):.3f}"
        )
    else:
        print("Depth stats: no valid depth values found.")


def _rgbd_to_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    depth_max: float | None,
    y_up: bool,
    metric: bool = False,
    epsilon: float = 0.2,
    target_min: float | None = 3.0,
    target_max: float | None = 4.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"RGB and depth sizes differ: {rgb.shape[:2]} vs {depth.shape[:2]}"
        )

    print_depth_stats(depth)

    # if not metric, normalize to 0-1 range and invert depth (assuming DepthAnything-style inverse depth)
    if not metric:
        depth_m = process_depth(depth)
    # if metric, use meters directly and optionally apply depth max threshold, do not normalize
    # otherwise metric range is lost!!
    else:
        depth_m = depth / float(depth_scale)
        if depth_max is not None:
            depth_m = np.where(depth_m <= depth_max, depth_m, 0.0)

    h, w = depth_m.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_m
    valid = z > 0.0

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    if y_up:
        y = -y

    points = np.stack([x, y, z], axis=-1)
    points = points[valid]
    # points = _normalize_points_unit_cube(points, centered=True)

    colors = rgb.reshape(-1, 3)[valid.reshape(-1)]

    return points.astype(np.float32), colors.astype(np.uint8), depth_m


def _points_to_mesh(
    points: np.ndarray, colors: np.ndarray
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Open3D expects float colors in [0, 1].
    colors_f = (colors.astype(np.float32) / 255.0).clip(0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors_f)

    print("Filtering noise...")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)

    print("Estimating surface normals for meshing...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0])
    )

    print("Generating Poisson Surface Mesh...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # crop mesh to remove artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    return mesh


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors length mismatch.")

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {points.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def _save_reference_images(
    out_dir: Path,
    base_name: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_max: float | None,
    epsilon: float = 0.05,
) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to save reference images. Install pillow."
        ) from exc

    rgb_path = out_dir / f"{base_name}_rgb.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)

    depth_m = depth / float(depth_scale)
    if depth_max is not None:
        depth_m = np.where(depth_m <= depth_max, depth_m, 0.0)
    valid = depth_m > 0.0
    if np.any(valid):
        min_d = float(depth_m[valid].min())
        max_d = float(depth_m[valid].max())
        span = max(max_d - min_d, epsilon)
        depth_norm = depth_m.copy()
        depth_norm[valid] = (depth_norm[valid] - min_d) / span
        depth_norm[~valid] = 0.0
        depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_m, dtype=np.uint8)

    depth_path = out_dir / f"{base_name}_depth.png"
    Image.fromarray(depth_vis, mode="L").save(depth_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a colored point cloud (PLY) from RGB + depth."
    )
    parser.add_argument("--rgb", type=Path, help="Path to RGB image.")
    parser.add_argument(
        "--campaign", type=str, help="Campaign name for cached RGB/depth."
    )
    parser.add_argument(
        "--deployment", type=str, help="Deployment name for cached RGB/depth."
    )
    parser.add_argument("--key", type=str, help="Image key within the deployment.")
    parser.add_argument(
        "--index",
        type=int,
        help="Image index within depths.npy (use --list-keys to inspect keys).",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List depth keys for a campaign/deployment and exit.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of cached images to process (uses first N keys).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PLY file path.",
    )
    parser.add_argument("--fx", type=float, help="Focal length x in pixels.")
    parser.add_argument("--fy", type=float, help="Focal length y in pixels.")
    parser.add_argument("--cx", type=float, help="Principal point x in pixels.")
    parser.add_argument("--cy", type=float, help="Principal point y in pixels.")
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=60.0,
        help="Horizontal field of view in degrees for default intrinsics.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Depth scale to convert to meters (e.g., 1000 for mm).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Optional max depth in meters; farther points are discarded.",
    )
    parser.add_argument(
        "--y-up",
        action="store_true",
        help="Flip Y to make Y axis up instead of down.",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Use metric depth cache (if available) instead of normalized.",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Use post-processed 'depth' output from the pipeline instead of 'predicted_depth'. Only applies to cached depth extraction, not raw --depth input.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Small value to prevent division by zero when normalizing depth.",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=None,
        help="Maximum depth in meters for target visualization.",
    )
    parser.add_argument(
        "--target-min",
        type=float,
        default=None,
        help="Minimum depth in meters for target visualization.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print(f"Arguments: {args}")

    use_cache = any(
        [
            args.campaign,
            args.deployment,
            args.key,
            args.index is not None,
            args.list_keys,
        ]
    )

    depth_scale = 1.0 if args.depth_scale is None else args.depth_scale
    if use_cache:
        if not args.campaign or not args.deployment:
            raise SystemExit(
                "--campaign and --deployment are required for cached RGB/depth."
            )
        loader = RGBDImageLoader(
            args.campaign, args.deployment, metric=args.metric, processed=args.processed
        )
        if args.list_keys:
            for k in loader.list_keys():
                print(k)
            return
        if args.count < 1:
            raise SystemExit("--count must be >= 1.")
        if args.key is not None or args.index is not None:
            if args.count != 1:
                raise SystemExit("--count cannot be used with --key or --index.")
            keys = [
                args.key if args.key is not None else loader.key_from_index(args.index)
            ]
        else:
            keys = loader.list_keys()[: args.count]
        if not keys:
            raise SystemExit("No cached keys found for the requested deployment.")
    else:
        if args.rgb is None:
            raise SystemExit("Provide --rgb and --depth, or use cached inputs.")
        rgb_path = Path(args.rgb)
        rgb = _load_rgb(rgb_path)
        campaign = rgb_path.parent.parent.name
        deployment = rgb_path.parent.name
        key = rgb_path.stem
        label = None
        if args.metric:
            label = "metric"
        elif args.processed:
            label = "processed"
        depth = _load_depth_numpy(campaign, deployment, key, label)

    if use_cache:
        if args.out.suffix:
            out_dir = args.out.parent
            if len(keys) > 1:
                raise SystemExit("When --count > 1, --out must be a directory path.")
            out_path_single = args.out
        else:
            out_dir = args.out
            out_path_single = None
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            rgb, depth = loader.load(key)
            print(
                f"Processing key '{key}' with RGB shape {rgb.shape} and depth shape {depth.shape}..."
            )
            if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
                fx, fy, cx, cy = _infer_intrinsics_v2(rgb.shape[1], rgb.shape[0])
            else:
                fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

            points, colors, depth = _rgbd_to_points(
                rgb=rgb,
                depth=depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale=depth_scale,
                depth_max=args.depth_max,
                y_up=args.y_up,
                metric=args.metric,
            )

            mesh = _points_to_mesh(points, colors)

            out_path = out_path_single or (out_dir / f"{key}_pointcloud.ply")
            out_path_mesh = out_path_single or (out_dir / f"{key}_mesh.ply")
            print(f"Saving Point Cloud to {out_path}...")
            _write_ply(out_path, points, colors)
            print(f"Saving solid mesh to {out_path_mesh}...")
            o3d.io.write_triangle_mesh(out_path_mesh, mesh)
            _save_reference_images(
                out_dir=out_dir,
                base_name=out_path.stem,
                rgb=rgb,
                depth=depth,
                depth_scale=depth_scale,
                depth_max=args.depth_max,
            )

            print(
                f"Wrote {points.shape[0]} points to {os.fspath(out_path)} "
                f"(fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f})."
            )
    else:
        if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
            fx, fy, cx, cy = _infer_intrinsics_v2(rgb.shape[1], rgb.shape[0])
        else:
            fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

        points, colors, depth = _rgbd_to_points(
            rgb=rgb,
            depth=depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
            depth_max=args.depth_max,
            y_up=args.y_up,
            metric=args.metric,
        )

        out_path = Path(args.out)
        base_name = out_path.stem
        pointcloud_path = out_path / f"{args.rgb.stem}.ply"

        out_path.mkdir(parents=True, exist_ok=True)
        _write_ply(pointcloud_path, points, colors)
        _save_reference_images(
            out_dir=out_path,
            base_name=base_name,
            rgb=rgb,
            depth=depth,
            depth_scale=depth_scale,
            depth_max=args.depth_max,
        )

        print(
            f"Wrote {points.shape[0]} points to {os.fspath(pointcloud_path)} "
            f"(fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f})."
        )


if __name__ == "__main__":
    main()
