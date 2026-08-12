"""GPU exposure normalization via Kornia CLAHE on LAB luminance.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torchvision
from kornia.enhance import equalize_clahe, equalize
from kornia.color import rgb_to_lab, lab_to_rgb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benthicflow import DATA_NORM_ROOT
from benthicflow.reef_io import iter_deployments, load_manifest, normalized_image_path
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)
JPEG_QUALITY     = 92


# ----------------------------- dataset -------------------------------------

class JpegBytesDataset(Dataset):
    """Yields raw JPEG bytes + destination paths.

    We deliberately do NOT decode on CPU — decoding happens on GPU in batches
    via torchvision.io.decode_jpeg.
    """

    def __init__(self, jobs):
        self.jobs = jobs   # list of (src_path, dst_path)

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        src, dst = self.jobs[i]
        try:
            data = torch.frombuffer(src.read_bytes(), dtype=torch.uint8)
            return data, str(dst), True
        except Exception:
            return torch.zeros(0, dtype=torch.uint8), str(dst), False


def collate(batch):
    # We keep variable-length JPEG byte tensors as a Python list — they get
    # batched on GPU after decode.
    bytes_list, dst_paths, valid_flags = zip(*batch)
    return list(bytes_list), list(dst_paths), list(valid_flags)


# ----------------------------- GPU pipeline -------------------------------

@torch.inference_mode()
def normalize_batch_gpu(jpeg_bytes_list, global_equalize=False):
    """Decode JPEGs on GPU, run CLAHE on L channel, return RGB uint8 CPU tensors."""
    imgs = torchvision.io.decode_jpeg(
        jpeg_bytes_list,
        mode=torchvision.io.ImageReadMode.RGB,
        device=DEVICE,
    )

    out_imgs = [None] * len(imgs)
    by_shape = {}

    for i, img in enumerate(imgs):
        by_shape.setdefault(tuple(img.shape), []).append((i, img))

    for shape, items in by_shape.items():
        idxs = [i for i, _ in items]

        batch = torch.stack([img for _, img in items], dim=0).float().div_(255.0)

        lab = rgb_to_lab(batch)
        L = lab[:, 0:1] / 100.0

        if global_equalize:
            L_eq = equalize(L) * 100.0
        else:
            L_eq = equalize_clahe(
                L,
                clip_limit=CLAHE_CLIP_LIMIT,
                grid_size=CLAHE_TILE_GRID,
            ) * 100.0

        lab[:, 0:1] = L_eq
        rgb = lab_to_rgb(lab).clamp_(0, 1)
        rgb_u8 = (rgb * 255.0).round().to(torch.uint8)

        # Critical: move final images to CPU before returning/submitting writes.
        rgb_u8_cpu = rgb_u8.cpu()

        for k, src_idx in enumerate(idxs):
            # clone prevents each slice from keeping the whole CPU batch storage alive
            out_imgs[src_idx] = rgb_u8_cpu[k].clone()

        del batch, lab, L, L_eq, rgb, rgb_u8, rgb_u8_cpu

    del imgs
    return out_imgs


# ----------------------------- writer ------------------------------------

def encode_and_write(img_chw_u8_cpu, dst_path):
    """JPEG-encode on CPU and write."""
    arr = img_chw_u8_cpu.permute(1, 2, 0).numpy()  # HWC RGB uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {dst_path}...")
    cv2.imwrite(str(dst_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


# ----------------------------- driver ------------------------------------

def collect_jobs(campaigns):
    if Path("normalize_jobs.pkl").exists():
        with open("normalize_jobs.pkl", "rb") as f:
            jobs = pickle.load(f)
        print(f"Loaded {len(jobs)} jobs from normalize_jobs.pkl")
        return jobs
    jobs = []
    for campaign, deployment, manifest, image_dir in list(iter_deployments(campaigns))[:1]:
        print(f"Collecting {campaign}/{deployment}...")
        df = load_manifest(manifest, image_dir)
        for _, row in df.iterrows():
            src = row["img_path"]
            dst = normalized_image_path(campaign, deployment, row["key"])
            if dst.exists() and dst.stat().st_size > 0:
                continue
            jobs.append((src, dst))
    
    with open("normalize_jobs.pkl", "wb") as f:
        pickle.dump(jobs, f)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaigns", nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=8,
                    help="Images per GPU batch.")
    ap.add_argument("--read-workers", type=int, default=8,
                    help="DataLoader workers for reading JPEG bytes.")
    ap.add_argument("--write-workers", type=int, default=8,
                    help="Thread pool size for JPEG encoding/writing.")
    ap.add_argument("--max-pending-writes", type=int, default=None,
                    help="Maximum queued CPU write jobs. Defaults to 4x write-workers.")
    ap.add_argument("--global-equalize", action="store_true",
                    help="Apply global histogram equalization instead of CLAHE.")
    args = ap.parse_args()

    DATA_NORM_ROOT.mkdir(exist_ok=True)
    jobs = collect_jobs(args.campaigns)
    if not jobs:
        sys.exit("Nothing to do — all images already normalized or no images found.")

    max_pending_writes = (
        args.max_pending_writes
        if args.max_pending_writes is not None
        else args.write_workers * 4
    )

    print(f"Normalizing {len(jobs):,} images on {DEVICE} "
          f"(batch={args.batch}, read_workers={args.read_workers}, "
          f"write_workers={args.write_workers}, "
          f"max_pending_writes={max_pending_writes})")

    ds = JpegBytesDataset(jobs)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.read_workers,
        collate_fn=collate,
    )

    n_ok, n_fail = 0, 0
    pending_writes = set()

    with ThreadPoolExecutor(max_workers=args.write_workers) as write_pool:
        for bytes_list, dst_paths, valid_flags in tqdm(dl, total=len(dl)):
            good_idx = [
                i for i, v in enumerate(valid_flags)
                if v and len(bytes_list[i]) > 0
            ]

            if not good_idx:
                n_fail += len(bytes_list)
                continue

            good_bytes = [bytes_list[i] for i in good_idx]
            good_dsts = [dst_paths[i] for i in good_idx]

            try:
                with torch.inference_mode():
                    out_imgs = normalize_batch_gpu(good_bytes, global_equalize=args.global_equalize)

                # Critical fix:
                # move each output to CPU before submitting to the writer.
                # Otherwise pending write futures keep CUDA tensors alive.
                for img_cpu, dst in zip(out_imgs, good_dsts):
                    fut = write_pool.submit(encode_and_write, img_cpu, dst)
                    pending_writes.add(fut)

                    # Bound the queue so RAM/VRAM cannot grow forever.
                    if len(pending_writes) >= max_pending_writes:
                        done, pending_writes = wait(
                            pending_writes,
                            return_when=FIRST_COMPLETED,
                        )

                        for f in done:
                            f.result()

                n_ok += len(out_imgs)

                # Release references to GPU tensors from this batch.
                del out_imgs

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nbatch failed ({e}), falling back to CPU for this batch")
                n_fail += len(good_bytes)

        # Drain remaining writes and surface possible writer errors.
        for f in pending_writes:
            f.result()

    print(f"\nDone. ok={n_ok}  fail={n_fail}")

if __name__ == "__main__":
    main()
