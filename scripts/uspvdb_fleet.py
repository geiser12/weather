"""USPVDB (US Large-Scale Solar PV Database) fetch + cache, mirrors uswtdb_fleet.py."""
import json
import time
from pathlib import Path

import requests

USPVDB_URL = "https://energy.usgs.gov/arcgis/rest/services/Hosted/uspvdbDyn/FeatureServer/0/query"
CACHE_PATH = Path("solar_fleet_tx.json")
CACHE_MAX_AGE_DAYS = 90  # USPVDB updates far less often than quarterly

FIELDS = ["p_name", "p_state", "p_county", "xlong", "ylat", "p_cap_ac", "p_cap_dc", "p_area", "eia_id", "p_year"]


def _query_tx_solar():
    params = {
        "where": "p_state='TX'",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "f": "json",
    }
    r = requests.get(USPVDB_URL, params=params, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    out = []
    for f in feats:
        a = f.get("attributes", {})
        out.append({
            "name": a.get("p_name") or "Solar Farm",
            "county": a.get("p_county") or "",
            "lat": a.get("ylat"),
            "lon": a.get("xlong"),
            "capacity_mw": a.get("p_cap_ac") or 0.0,   # AC nameplate, matches ERCOT MW convention
        })
    return out


def fetch_tx_solar_fleet(force=False):
    if not force and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days < CACHE_MAX_AGE_DAYS:
            return json.loads(CACHE_PATH.read_text())
    farms = _query_tx_solar()
    CACHE_PATH.write_text(json.dumps(farms))
    return farms


def demo_solar_fleet():
    return [
        {"name": "Demo Solar 1", "county": "pecos", "lat": 31.0, "lon": -103.0, "capacity_mw": 200.0},
        {"name": "Demo Solar 2", "county": "andrews", "lat": 32.3, "lon": -102.5, "capacity_mw": 150.0},
    ]