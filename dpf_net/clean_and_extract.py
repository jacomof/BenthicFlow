import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from dataset_UIEB import UIEB_Dataset
from Depth_Anything_V2_main.depth_anything_v2.dpt import DepthAnythingV2
from DPEM import DPEM_model
from loss import Totaloss
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader

import DPF_Net
from benthicflow.reef_io import DATA_ROOT, DEPTH_ROOT

start_time = str(datetime.now())[0:19].replace(" ", "-")


def flat_depth_paths(
    campaign: str, deployment: str, label: str | None = None
) -> tuple[Path, Path]:
    """Returns paths for the raw depth array and the corresponding keys."""
    suffix = f"_{label}" if label else ""
    depth_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}.npy"
    keys_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}_keys.npy"
    print(f"Depth path: {depth_npy}, Keys path: {keys_npy}")
    return depth_npy, keys_npy


def generate_src_image_path(campaign: str, deployment: str, key: str) -> Path:
    """Path to one source JPEG."""
    return str(DATA_ROOT / campaign / deployment / "images" / f"{key}.jpg")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_image_path",
        type=str,
        help="path to the folder of images",
        default="./UIEB/test/raw",
    )

    parser.add_argument(
        "--load_DPF_Net",
        type=str,
        help="path of a pretrained DPF-Net to use",
        default="./checkpoint/DPF-Net.pth",
    )

    parser.add_argument(
        "--load_DPEM",
        type=str,
        help="path of a pretrained DPF-Net to use",
        default="./checkpoint/DPEM_finetune.pth",
    )

    parser.add_argument(
        "--depth_anything_folder",
        type=str,
        help="path of a pretrained depth_anything to use",
        default="./Depth_Anything_V2_main",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="select the device to run the models on",
        default="cuda",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="path to save the output images",
        default="./cleaned_images",
    )

    parser.add_argument(
        "--batch_size", type=int, help="batch size to use for inference", default=1
    )

    parser.add_argument(
        "--split_path",
        type=Path,
        help="path to the data splits directory",
        default="./data_split",
    )

    parser.add_argument(
        "--debug", action="store_true", help="enable debug mode for verbose output"
    )

    parser.add_argument(
        "--campaign",
        type=str,
        default=None,
        help="specific campaign to process (overrides split_path)",
    )

    return parser.parse_args()


def test(args):
    depth_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    encoder = "vits"
    depth_anything = DepthAnythingV2(**depth_configs[encoder])
    depth_anything.load_state_dict(
        torch.load(
            f"{args.depth_anything_folder}/depth_anything_v2_{encoder}.pth",
            map_location=args.device,
        )
    )
    depth_anything = depth_anything.to(args.device).eval()

    model = DPF_Net.TotalNetwork(args.device).eval()
    model.load_state_dict(torch.load(args.load_DPF_Net, map_location=args.device))
    dpem = DPEM_model.MainNet(
        device=args.device, imgSize=256, depth_anything_dir=args.depth_anything_folder
    ).eval()
    dpem.load_state_dict(torch.load(args.load_DPEM, map_location=args.device))

    split_csvs = Path(args.split_path).glob("*.csv")

    df = pd.concat((pd.read_csv(csv) for csv in split_csvs), ignore_index=True)

    if args.campaign is not None:
        df = df[df["campaign"] == args.campaign]

    groups = df.groupby(["campaign", "deployment"])

    for (campaign, deployment), group in groups:
        d_path, k_path = flat_depth_paths(campaign, deployment)

        if d_path.exists() and k_path.exists():
            print(f"Skipping {campaign}/{deployment} (depth files already exist)")
            continue

        print(f"Processing campaign: {campaign}, deployment: {deployment}")

        img_list = [
            generate_src_image_path(campaign, deployment, key)
            for key in group["key"].tolist()
        ]

        dataset = UIEB_Dataset(
            img_list,
            campaign,
            deployment,
            depthanything=depth_anything,
            device=args.device,
            isTrain=False,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_keys = []
        all_depths = []

        print("Test start")
        with torch.no_grad():
            for batch_idx, (
                data_raw,
                data_depth,
                BL,
                file_name,
                dst_path,
                key,
            ) in enumerate(dataloader):

                x_B, x_beta_D, x_beta_B, x_d = dpem(data_raw, BL)
                replicated_x_B = (
                    x_B.unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
                )
                replicated_x_beta_D = (
                    x_beta_D.unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
                )
                replicated_x_beta_B = (
                    x_beta_B.unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, data_raw.shape[2], data_raw.shape[3])
                )
                channel_replica1 = x_d[:, 0:1, :, :]
                channel_replica2 = x_d[:, 0:1, :, :]
                replicated_x_d = torch.cat(
                    (x_d, channel_replica1, channel_replica2), dim=1
                )

                outputs = model(
                    data_raw,
                    replicated_x_B,
                    replicated_x_d,
                    replicated_x_beta_D,
                    replicated_x_beta_B,
                )

                for i in range(data_raw.shape[0]):
                    curr_dst_path = Path(dst_path[i])

                    if args.debug:
                        print(f"Processing image: {file_name[i]}")
                        print(f"Raw image shape: {data_raw[i].shape}")
                        print(f"Depth map shape: {data_depth[i].shape}")
                        print(f"Output image shape: {outputs[i].shape}")
                        print(f"Saving cleaned image to {curr_dst_path}")

                    if not curr_dst_path.parent.exists():
                        curr_dst_path.parent.mkdir(parents=True, exist_ok=True)

                    enc_img = (
                        (outputs[i] * 255).to("cpu", dtype=torch.uint8).permute(1, 2, 0)
                    )
                    img_save = Image.fromarray(enc_img.numpy())
                    if args.debug:
                        print(f"Saving image to {curr_dst_path}")
                    img_save.save(curr_dst_path)
                    depth_np = x_d[i].to("cpu", dtype=torch.float32).detach().numpy()
                    all_depths.append(depth_np)

                    all_keys.append(key[i])

                    depth_vis = depth_np.squeeze()
                    depth_vis = (depth_vis - depth_vis.min()) / (
                        depth_vis.max() - depth_vis.min() + 1e-8
                    )
                    depth_vis = (depth_vis * 255).astype(np.uint8)

        d_path.parent.mkdir(parents=True, exist_ok=True)

        depths = np.stack(all_depths, axis=0)
        keys = np.array(all_keys, dtype=str)

        np.save(d_path, depths)
        np.save(k_path, keys)

        print(f"\nDepth maps and keys saved to {d_path} and {k_path}")
        print(f"Finished processing campaign: {campaign}, deployment: {deployment}\n")

    print("\nTest completed")


if __name__ == "__main__":
    args = parse_args()
    test(args)
