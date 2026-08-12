"""
Pull all Sirius AUV imagery for the public DreamSea-equivalent datasets
(Scott Reef + Batemans, 2009–2015) from Squidle+.

Outputs:
  data/<campaign>/<deployment>/images/*.jpg   raw images
  data/<campaign>/<deployment>/manifest.csv   pose metadata
"""

import os, csv, json, time, requests, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from discover_campaigns import get_platform_id, list_campaigns

API   = "https://squidle.org/api"
ROOT  = Path(os.environ.get("REEF_DATA_SCRATCH_ROOT", "data"))
TOKEN = os.environ.get("SQUIDLE_TOKEN")           # optional; higher rate limits
HEAD  = {"X-auth-token": TOKEN} if TOKEN else {}


KEYWORDS = [
    # -- DreamSea-equivalent subset (Sirius, 2009–2015) ---
    "Scott", 
    #"Batemans",
    # # --- New South Wales (Temperate Reefs) ---
    # "Stephens",  # Catches Port Stephens
    # "Sydney",    # Catches Sydney Harbour/offshore lines
    # "Jervis",    # Catches Jervis Bay
    # "Solitary",  # Catches Solitary Islands

    # # --- Western Australia ---
    # "Rottnest",  # Catches Rottnest Island
    # "Abrolhos",  # Catches Houtman Abrolhos
    # "Jurien",    # Catches Jurien Bay
    # "Ningaloo",  # Catches Ningaloo Reef

    # # --- Tasmania & Queensland ---
    # "Tasman",    # Catches Tasman Peninsula / Tasmania
    # "Lizard",    # Catches Lizard Island
    # "Heron",      # Catches Heron Island
    # "SEQueensland",

    # -- Hawaii 
    #"Hawaii",     # Catches Hawaii deployments

    # # Great Barrier Reef
    # "GBR",        # Catches some Great Barrier Reef deployments



    # # -- Other / unknown --
    # "PS2",

]
PLATFORM_ID = get_platform_id("IMOS AUV Sirius")  # Set to None to keep all platforms.
USE_THUMBNAILS = False  # True ≈ 5 KB/img, False = full-res ≈ 500 KB/img.
MAX_IMAGES_PER_CAMPAIGN = 100_000
MAX_WORKERS = 16
# ---------------------------------------------------------------------------


def discover_campaigns():
    campaigns = []
    seen = set()
    for kw in KEYWORDS:
        for c in list_campaigns(kw):
            key = c.get("key")
            if not key or key in seen:
                continue
            seen.add(key)
            campaigns.append((key, c["id"]))
    campaigns.sort(key=lambda x: x[0])
    return campaigns


def deployment_already_downloaded(dep_dir: Path) -> bool:
    """Check if all images for a deployment are downloaded and non-empty."""
    manifest_path = dep_dir / "manifest.csv"
    print(f"Checking if deployment {dep_dir} is already downloaded...")
    if not manifest_path.exists():
        print(f"Not downloaded (manifest missing) {dep_dir}")
        return False, 0

    img_dir = dep_dir / "images"
    if not img_dir.exists():
        print(f"Not downloaded (images missing) {dep_dir}")
        return False, 0

    # Count lines in manifest (rows, excluding header)
    with open(manifest_path, 'r') as f:
        manifest_lines = sum(1 for _ in f) - 1

    # Quick check: number of images must match
    downloaded_count = sum(1 for img in img_dir.iterdir()
                          if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if downloaded_count != manifest_lines:
        print(f"Manifest mismatch for {dep_dir}. Expected {manifest_lines}, got {downloaded_count}")
        return False, 0

    # Detailed check: verify all images are non-empty
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = img_dir / f"{row['key']}.jpg"
            if not img_path.exists() or img_path.stat().st_size == 0:
                return False, 0

    print(f"  (already downloaded) {dep_dir}")
    return True, downloaded_count


def cleanup_campaign_to_limit(camp_key: str) -> None:
    """Delete deployments if campaign exceeds image limit. Deletes in alphanumeric order."""
    camp_dir = ROOT / camp_key
    if not camp_dir.exists():
        return

    # Get all deployments with manifests, sorted alphanumerically
    deployments = []
    for dep_dir in sorted(camp_dir.iterdir()):
        if not dep_dir.is_dir():
            continue
        manifest_path = dep_dir / "manifest.csv"
        if manifest_path.exists():
            img_dir = dep_dir / "images"
            if img_dir.exists():
                img_count = sum(1 for img in img_dir.iterdir()
                               if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"})
                deployments.append((dep_dir.name, dep_dir, img_count))

    # Calculate total images
    total_images = sum(count for _, _, count in deployments)

    # Delete deployments from the front if over limit
    if total_images > MAX_IMAGES_PER_CAMPAIGN:
        for dep_name, dep_path, dep_count in sorted(deployments, reverse=True):
            if total_images <= MAX_IMAGES_PER_CAMPAIGN:
                break
            # Check if removing this deployment would bring us under the limit
            if total_images - dep_count < MAX_IMAGES_PER_CAMPAIGN:
                # Would go under limit, so don't remove this one
                break
            # Remove deployment
            print(f"Campaign {camp_key} exceeds limit ({total_images} images). Deleting deployment {dep_name} ({dep_count} images)...")
            shutil.rmtree(dep_path)
            total_images -= dep_count
            print(f"  Deleted deployment {camp_key}/{dep_name} ({dep_count} images). Campaign total: {total_images} images.")


def campaign_already_downloaded(camp_key: str) -> bool:
    camp_dir = ROOT / camp_key
    print(f"Checking if campaign {camp_key} is already downloaded...")
    if not camp_dir.exists():
        return False

    found_any_manifest = False

    for dep_dir in camp_dir.iterdir():
        if not dep_dir.is_dir():
            continue

        if (dep_dir / "manifest.csv").exists():
            found_any_manifest = True
            already_downloaded, _ = deployment_already_downloaded(dep_dir)
            if not already_downloaded:
                return False

    # If campaign is complete, clean up if it exceeds the image limit
    if found_any_manifest:
        print(f"  (already downloaded) {camp_key}")

    return found_any_manifest


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


def download_one(url, dst, max_attempts=3):
    """Download a single image with retries. Skips if file already exists and is non-empty."""
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


def process_deployment(camp_key, dep):
    """Download all media for one deployment and write a manifest CSV."""

    print(f"Processing deployment {camp_key}/{dep['key']}...")

    dep_key = dep["key"]
    out_dir = ROOT / camp_key / dep_key

    already_downloaded, downloaded_count = deployment_already_downloaded(out_dir)
    if already_downloaded:
        print(f"  (already downloaded) {camp_key}/{dep_key}")
        return downloaded_count

    print(f"  Downloading {camp_key}/{dep_key}...")
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    media = media_for_deployment(dep["id"])
    if not media:
        print(f"  (empty) {camp_key}/{dep_key}")
        return 0

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
    
    print(f"  Finished {camp_key}/{dep_key}: {len(rows)} images downloaded.")
    return len(rows)


def main():
    ROOT.mkdir(exist_ok=True)
    campaigns = discover_campaigns()
    
    for camp_key, camp_id in campaigns:

        # cleanup_campaign_to_limit(camp_key)
        # if campaign_already_downloaded(camp_key):
        #     print(f"\n{camp_key}: already downloaded. Skipping.")
        #     continue
        # Clean up existing deployments if campaign exceeds limit before downloading new ones
        
        current_campaign_count = 0
        deps = deployments_for_campaign(camp_id)
        print(f"\n{camp_key} (id={camp_id}): {len(deps)} Sirius deployment(s)")
        if not deps:
            continue
        if camp_key != "ScottReef201503":
            print(f"  Skipping {camp_key} (not ScottReef201503).")
            continue
        for dep in deps:
            if dep['key'] not in ["r20150330_225013_09_scott_grids_deep_auv2", 
                           "r20150331_050931_10_scott_long_leg_auv2",
                           "r20150331_231619_11_scott_repeat_large_200907_25_auv5"]:
                
                print(f"  Skipping deployment {dep['key']} (not in selected list).")
                continue
            current_campaign_count += process_deployment(camp_key, dep)
            if current_campaign_count >= MAX_IMAGES_PER_CAMPAIGN:
                print(f"Reached max images for campaign {camp_key}. Stopping further downloads.")
                break



if __name__ == "__main__":
    main()