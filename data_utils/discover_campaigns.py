"""Discover Sirius AUV campaigns matching keywords on Squidle+."""

import json
import os
import time

import requests

API = "https://squidle.org/api"
TOKEN = os.environ.get("SQUIDLE_TOKEN")
HEAD = {"X-auth-token": TOKEN} if TOKEN else {}

KEYWORDS = [
    # -- DreamSea-equivalent subset (Sirius, 2009–2015) ---
    "Scott",
    "Batemans",
    # --- New South Wales (Temperate Reefs) ---
    "Stephens",  # Catches Port Stephens
    "Sydney",  # Catches Sydney Harbour/offshore lines
    "Jervis",  # Catches Jervis Bay
    "Solitary",  # Catches Solitary Islands
    # --- Western Australia ---
    "Rottnest",  # Catches Rottnest Island
    "Abrolhos",  # Catches Houtman Abrolhos
    "Jurien",  # Catches Jurien Bay
    "Ningaloo",  # Catches Ningaloo Reef
    # --- Tasmania & Queensland ---
    "Tasman",  # Catches Tasman Peninsula / Tasmania
    "Lizard",  # Catches Lizard Island
    "Heron",  # Catches Heron Island
    "SEQueensland",
    # -- Hawaii
    "Hawaii",  # Catches Hawaii deployments (non-reef, but good for testing)
    # Great Barrier Reef
    "GBR",  # Catches some Great Barrier Reef deployments
    # -- Other / unknown --
    "PS2",
]


def list_campaigns(keyword):
    """Find campaigns whose key contains the keyword (case-insensitive)."""
    q = {"filters": [{"name": "key", "op": "ilike", "val": f"%{keyword}%"}]}
    r = requests.get(
        f"{API}/campaign", params={"q": json.dumps(q), "results_per_page": 200}
    )
    r.raise_for_status()
    print(f"Found {len(r.json()['objects'])} campaigns matching '{keyword}'")
    return r.json()["objects"]


def get_platform_id(platform_key="IMOS AUV Sirius"):
    """Look up the platform ID for a given platform key."""
    q = {"filters": [{"name": "key", "op": "eq", "val": platform_key}]}
    r = requests.get(f"{API}/platform", params={"q": json.dumps(q)})
    r.raise_for_status()
    objs = r.json()["objects"]
    return objs[0]["id"] if objs else None


def list_all_platforms():
    r = requests.get(f"{API}/platform", params={"results_per_page": 100})
    for p in r.json().get("objects", []):
        print(f"ID: {p['id']} | Name: {p['name']}")


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


PLATFORM_ID = get_platform_id("IMOS AUV Sirius")  # Set to None to keep all platforms.


def deployments_for_campaign(camp_id):
    """List deployments under a campaign, optionally filtered to one platform."""
    filters = [{"name": "campaign_id", "op": "eq", "val": camp_id}]
    if PLATFORM_ID is not None:
        filters.append({"name": "platform_id", "op": "eq", "val": PLATFORM_ID})
    q = {"filters": filters}

    out, page = [], 1
    while True:
        j = get(
            f"{API}/deployment",
            {"q": json.dumps(q), "page": page, "results_per_page": 200},
        )
        out += j["objects"]
        if page >= j["total_pages"]:
            break
        page += 1
    return out


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


if __name__ == "__main__":
    # Run this once to see the ecosystem layout!
    list_all_platforms()
    pid = get_platform_id("IMOS AUV Sirius")
    print(f"IMOS AUV Sirius platform_id = {pid}\n")

    for camp_key, camp_id in discover_campaigns():
        deps = deployments_for_campaign(camp_id)
        print(f"\n{camp_key} (id={camp_id}): {len(deps)} Sirius deployment(s)")

    for kw in ["Scott", "Batemans"]:
        print(f"=== {kw} ===")
        for c in list_campaigns(kw):
            print(f"  id={c['id']:>4}  key={c['key']:<40}  name={c.get('name','')}")
        print()
