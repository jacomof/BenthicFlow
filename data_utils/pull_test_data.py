"""Download test data assets for evaluation.

Retrieves evaluation datasets and deployment manifests from remote scratch storage.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://squidle.org/api"
TOKEN = os.environ.get("SQUIDLE_TOKEN")  # optional; higher rate limits
HEAD = {"X-auth-token": TOKEN} if TOKEN else {}

DEFAULT_TEST_CSV = Path(__file__).resolve().parents[1] / "data_split" / "test.csv"
DEFAULT_ROOT = Path(os.environ.get("REEF_SCRATCH_ROOT", ".")) / "data_test"
PLATFORM_KEY = "IMOS AUV Sirius"


# ---------------------------------------------------------------------------
# Squidle+ API helpers (self-contained; mirror discover_campaigns.py)
# ---------------------------------------------------------------------------
def get(url, params=None, retries=5):
    """GET with exponential backoff. Raises on final failure."""
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEAD, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2**k)


def platform_id(key=PLATFORM_KEY):
    q = {"filters": [{"name": "key", "op": "eq", "val": key}]}
    objs = get(f"{API}/platform", {"q": json.dumps(q)})["objects"]
    return objs[0]["id"] if objs else None


def campaign_id_for_key(camp_key):
    """Resolve a campaign key -> id (exact match, then case-insensitive fallback)."""
    for op, val in (("eq", camp_key), ("ilike", f"%{camp_key}%")):
        q = {"filters": [{"name": "key", "op": op, "val": val}]}
        objs = get(f"{API}/campaign", {"q": json.dumps(q), "results_per_page": 50})[
            "objects"
        ]
        exact = [o for o in objs if o["key"] == camp_key]
        if exact:
            return exact[0]["id"]
        if op == "ilike" and len(objs) == 1:
            return objs[0]["id"]
    return None


def deployment_id_for_key(camp_id, dep_key, pid):
    """Resolve a deployment key -> id within a campaign (with then without platform)."""
    for use_platform in (True, False):
        filters = [
            {"name": "campaign_id", "op": "eq", "val": camp_id},
            {"name": "key", "op": "eq", "val": dep_key},
        ]
        if use_platform and pid is not None:
            filters.append({"name": "platform_id", "op": "eq", "val": pid})
        objs = get(
            f"{API}/deployment",
            {"q": json.dumps({"filters": filters}), "results_per_page": 10},
        )["objects"]
        if objs:
            return objs[0]["id"]
    return None


def media_for_deployment(dep_id):
    """All media items (with pose) for a deployment, keyed by media key."""
    by_key, page = {}, 1
    while True:
        j = get(
            f"{API}/media",
            {
                "q": json.dumps(
                    {"filters": [{"name": "deployment_id", "op": "eq", "val": dep_id}]}
                ),
                "page": page,
                "results_per_page": 500,
            },
        )
        for m in j["objects"]:
            by_key[m["key"]] = m
        if page >= j["total_pages"]:
            break
        page += 1
    return by_key


def download_one(url, dst, max_attempts=3):
    """Download one image with retries; skip if it already exists and is non-empty."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            return True
        except Exception as e:
            if dst.exists():
                dst.unlink()
            if attempt == max_attempts - 1:
                print(f"FAIL {url}: {e}")
                return False
    return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_test_index(test_csv: Path):
    """test.csv -> {(campaign, deployment): set(keys)}."""
    wanted = defaultdict(set)
    with open(test_csv, newline="") as f:
        for row in csv.DictReader(f):
            wanted[(row["campaign"], row["deployment"])].add(row["key"])
    return wanted


def process_deployment(
    camp, dep, keys, pid, root: Path, workers: int, thumbnails: bool, dry_run: bool
):
    out_dir = root / camp / dep
    img_dir = out_dir / "images"

    camp_id = campaign_id_for_key(camp)
    if camp_id is None:
        print(f"[skip] {camp}/{dep}: campaign not found on Squidle+")
        return 0, 0, len(keys)
    dep_id = deployment_id_for_key(camp_id, dep, pid)
    if dep_id is None:
        print(f"[skip] {camp}/{dep}: deployment not found (campaign_id={camp_id})")
        return 0, 0, len(keys)

    media = media_for_deployment(dep_id)
    url_field = "path_best_thm" if thumbnails else "path_best"

    rows, missing = [], []
    for k in sorted(keys):
        m = media.get(k)
        if m is None or not m.get(url_field):
            missing.append(k)
            continue
        pose = m.get("pose") or {}
        rows.append(
            {
                "media_id": m["id"],
                "key": m["key"],
                "url": m[url_field],
                "lat": pose.get("lat"),
                "lon": pose.get("lon"),
                "alt": pose.get("alt"),
                "depth": pose.get("dep"),
                "timestamp": m.get("timestamp"),
            }
        )

    print(
        f"{camp}/{dep}: {len(keys)} wanted, matched {len(rows)}, missing {len(missing)} "
        f"(deployment has {len(media)} media)"
    )
    if dry_run:
        return len(rows), 0, len(missing)

    img_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "media_id",
                "key",
                "url",
                "lat",
                "lon",
                "alt",
                "depth",
                "timestamp",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(download_one, r["url"], img_dir / f"{r['key']}.jpg") for r in rows
        ]
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc=f"{camp}/{dep}", leave=False
        ):
            ok += bool(fut.result())
    print(f"  downloaded {ok}/{len(rows)} -> {img_dir}")
    return len(rows), ok, len(missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Destination root; images go to <root>/<campaign>/<deployment>/images/.",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--thumbnails",
        action="store_true",
        help="Pull ~5KB thumbnails instead of full-res (~500KB) images.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve deployments and report match counts without downloading.",
    )
    args = ap.parse_args()

    wanted = load_test_index(args.test_csv)
    total = sum(len(v) for v in wanted.values())
    print(f"test.csv: {len(wanted)} deployment(s), {total} images -> {args.root}\n")

    pid = platform_id()
    grand = defaultdict(int)
    for (camp, dep), keys in sorted(wanted.items()):
        n, ok, miss = process_deployment(
            camp, dep, keys, pid, args.root, args.workers, args.thumbnails, args.dry_run
        )
        grand["matched"] += n
        grand["downloaded"] += ok
        grand["missing"] += miss

    print(
        f"\nDONE. matched={grand['matched']}  downloaded={grand['downloaded']}  "
        f"missing={grand['missing']}  (of {total} test images)"
    )
    if grand["missing"]:
        print(
            "Some test keys were not found on Squidle+ (deployment contents may have "
            "changed). Re-run to resume; missing keys are reported per deployment above."
        )


if __name__ == "__main__":
    main()
