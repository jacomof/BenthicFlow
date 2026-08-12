"""
Pull all Sirius AUV imagery for the public DreamSea-equivalent datasets
(Scott Reef + Batemans, 2009–2015) from Squidle+.

Outputs:
  data/<campaign>/<deployment>/images/*.jpg   raw images
  data/<campaign>/<deployment>/manifest.csv   pose metadata
"""

import os, csv, json, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

API   = "https://squidle.org/api"
ROOT  = Path(os.environ.get("REEF_DATA_SCRATCH_ROOT", "data"))
TOKEN = os.environ.get("SQUIDLE_TOKEN")           # optional; higher rate limits
HEAD  = {"X-auth-token": TOKEN} if TOKEN else {}

# ---- DreamSea-equivalent public subset (Sirius, 2009–2015) ----------------
# (campaign_key, campaign_id) — IDs from discover_campaigns.py
CAMPAIGNS = [
    ("ScottReef200907", 20),
    ("ScottReef201108", 21),
    ("ScottReef201503", 22),
    ("Batemans201011",   3),
    ("Batemans201211",   4),
    ("Batemans201411",   5),
]
PLATFORM_ID    = 1          # IMOS AUV Sirius. Set to None to keep all platforms.
USE_THUMBNAILS = False       # True ≈ 5 KB/img, False = full-res ≈ 500 KB/img.
MAX_WORKERS    = 16
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
            time.sleep(2 ** k)


def deployments_for_campaign(camp_id):
    """List deployments under a campaign, optionally filtered to one platform."""
    filters = [{"name": "campaign_id", "op": "eq", "val": camp_id}]
    if PLATFORM_ID is not None:
        filters.append({"name": "platform_id", "op": "eq", "val": PLATFORM_ID})
    q = {"filters": filters}

    out, page = [], 1
    while True:
        j = get(f"{API}/deployment",
                {"q": json.dumps(q), "page": page, "results_per_page": 200})
        out += j["objects"]
        if page >= j["total_pages"]:
            break
        page += 1
    return out


def media_for_deployment(dep_id):
    """List all media items for a deployment, with pose."""
    q = {"filters": [{"name": "deployment_id", "op": "eq", "val": dep_id}]}
    out, page = [], 1
    while True:
        j = get(f"{API}/media",
                {"q": json.dumps(q), "page": page, "results_per_page": 500})
        out += j["objects"]
        if page >= j["total_pages"]:
            break
        page += 1
    return out


def download_one(url, dst):
    """Download a single image. Skips if file already exists and is non-empty."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"FAIL {url}: {e}")
        if dst.exists():
            dst.unlink()
        return False


def process_deployment(camp_key, dep):
    """Download all media for one deployment and write a manifest CSV."""
    dep_key = dep["key"]
    out_dir = ROOT / camp_key / dep_key
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    media = media_for_deployment(dep["id"])
    if not media:
        print(f"  (empty) {camp_key}/{dep_key}")
        return

    rows = []
    for m in media:
        pose = m.get("pose") or {}
        rows.append({
            "media_id":  m["id"],
            "key":       m["key"],
            "url":       m["path_best_thm"] if USE_THUMBNAILS else m["path_best"],
            "lat":       pose.get("lat"),
            "lon":       pose.get("lon"),
            "alt":       pose.get("alt"),
            "depth":     pose.get("dep"),
            "timestamp": m.get("timestamp"),
        })

    # Write manifest first so metadata survives even if downloads fail
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Parallel image download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(download_one, r["url"], img_dir / f"{r['key']}.jpg")
                for r in rows]
        for _ in tqdm(as_completed(futs), total=len(futs),
                      desc=f"{camp_key}/{dep_key}", leave=False):
            pass


def main():
    ROOT.mkdir(exist_ok=True)
    for camp_key, camp_id in CAMPAIGNS:
        deps = deployments_for_campaign(camp_id)
        print(f"\n{camp_key} (id={camp_id}): {len(deps)} Sirius deployment(s)")
        for dep in deps:
            process_deployment(camp_key, dep)


if __name__ == "__main__":
    main()