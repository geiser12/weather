#!/usr/bin/env python3
"""
uswtdb_fleet.py
================
Fetches Texas wind-farm locations + capacities from the USGS/LBNL/ACP
U.S. Wind Turbine Database (USWTDB) ArcGIS REST service, and aggregates
individual turbines up to project ("wind farm") level: {name, lat, lon,
capacity_mw, turbine_count}.

No API key needed - USWTDB is a fully public ArcGIS FeatureServer/MapServer.
    Service: https://energy.usgs.gov/arcgis/rest/services/uswtdb/uswtdbDyn/MapServer/0
    Docs:    https://energy.usgs.gov/uswtdb/api-doc/
    Field definitions: https://www.sciencebase.gov/catalog/item/57bdfbc9e4b03fd6b7df5ff9

We aggregate to project level (not individual turbine) because our forecast
grid is ~50-80km resolution - sampling every turbine in a farm separately
adds no information, just compute. Turbines are grouped by (project name,
county) to avoid accidentally merging two different projects that happen
to share a generic name in different parts of the state.

Results are cached locally (wind_fleet_tx.json) since USWTDB updates only
quarterly - no need to hit the API on every run.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import requests

USWTDB_QUERY_URL = "https://energy.usgs.gov/api/uswtdb/v1/turbines"
CACHE_PATH = Path("wind_fleet_tx.json")
CACHE_MAX_AGE_DAYS = 30          # USWTDB is refreshed quarterly
PAGE_SIZE = 5000                  # service MaxRecordCount

FIELDS = [
    "case_id", "eia_id", "t_state", "t_county", "p_name", "p_year",
    "p_tnum", "p_cap", "t_manu", "t_model", "t_cap", "t_hh", "t_rd",
    "xlong", "ylat",
]

# USWTDB uses -9999 (and variants) as a "not available" sentinel.
_MISSING = {-9999, -9999.0, "-9999", None}


def _clean(v):
    return None if v in _MISSING else v


def _fetch_page(session: requests.Session, offset: int) -> list[dict]:
    params = {
        "t_state": "eq.TX",
        "select": ",".join(FIELDS),
        "limit": PAGE_SIZE,
        "offset": offset,
        "order": "case_id",
    }

    r = session.get(USWTDB_QUERY_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"USWTDB API error: {payload['error']}")

    return payload if isinstance(payload, list) else []



def _fetch_all_turbines() -> list[dict]:
    session = requests.Session()
    rows = []
    offset = 0

    print("Fetching TX turbines from USWTDB (energy.usgs.gov)...")

    while True:
        page = _fetch_page(session, offset)

        if not page:
            break

        rows.extend(page)
        print(f"  fetched {len(rows)} turbines so far...")

        if len(page) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return rows


def _aggregate_to_farms(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for row in rows:
        lat, lon = _clean(row.get("ylat")), _clean(row.get("xlong"))
        cap_kw = _clean(row.get("t_cap"))
        if lat is None or lon is None or cap_kw is None:
            continue  # can't place or size this turbine - skip it
        name = row.get("p_name") or f"Unnamed ({row.get('t_county', 'TX')})"
        county = row.get("t_county") or "?"
        key = (name, county)
        g = groups.setdefault(key, {
            "name": name, "county": county,
            "lat_sum": 0.0, "lon_sum": 0.0, "capacity_mw": 0.0,
            "turbine_count": 0, "hub_heights": [], "year": row.get("p_year"),
        })
        g["lat_sum"] += float(lat)
        g["lon_sum"] += float(lon)
        g["capacity_mw"] += float(cap_kw) / 1000.0
        g["turbine_count"] += 1
        hh = _clean(row.get("t_hh"))
        if hh is not None:
            g["hub_heights"].append(float(hh))

    farms = []
    for g in groups.values():
        n = g["turbine_count"]
        farms.append({
            "name": g["name"],
            "county": g["county"],
            "lat": g["lat_sum"] / n,
            "lon": g["lon_sum"] / n,
            "capacity_mw": round(g["capacity_mw"], 2),
            "turbine_count": n,
            "avg_hub_height_m": round(sum(g["hub_heights"]) / len(g["hub_heights"]), 1) if g["hub_heights"] else None,
            "online_year": g["year"] if g["year"] not in _MISSING else None,
        })
    farms.sort(key=lambda f: -f["capacity_mw"])
    return farms


def fetch_tx_wind_fleet(force: bool = False) -> list[dict]:
    """Returns a list of TX wind farms: {name, county, lat, lon, capacity_mw,
    turbine_count, avg_hub_height_m, online_year}, aggregated from USWTDB.
    Cached locally for CACHE_MAX_AGE_DAYS since USWTDB updates quarterly."""
    if not force and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days < CACHE_MAX_AGE_DAYS:
            print(f"  using cached wind fleet ({CACHE_PATH}, {age_days:.1f}d old)")
            return json.loads(CACHE_PATH.read_text())["farms"]

    rows = _fetch_all_turbines()
    if not rows:
        raise RuntimeError(
            "USWTDB returned zero TX turbines - check network access to "
            "energy.usgs.gov, or that the service schema hasn't changed."
        )
    farms = _aggregate_to_farms(rows)
    total_mw = sum(f["capacity_mw"] for f in farms)
    total_turbines = sum(f["turbine_count"] for f in farms)
    print(f"  {len(farms)} TX wind farms, {total_turbines} turbines, "
          f"{total_mw:,.0f} MW total nameplate capacity")

    CACHE_PATH.write_text(json.dumps(
        {"fetched_at": time.time(), "source": "USWTDB", "farms": farms}, indent=2
    ))
    return farms


def demo_fleet() -> list[dict]:
    """A handful of real, roughly-located big TX wind farms, for offline/demo
    testing (TX_DEMO=1) without hitting the USWTDB API. Capacities are
    approximate - do not use for anything beyond exercising the pipeline."""
    return [
        {"name": "Roscoe Wind Complex", "county": "Nolan", "lat": 32.45, "lon": -100.55,
         "capacity_mw": 781.5, "turbine_count": 627, "avg_hub_height_m": 80.0, "online_year": 2009},
        {"name": "Horse Hollow Wind Energy Center", "county": "Taylor", "lat": 32.28, "lon": -99.85,
         "capacity_mw": 735.5, "turbine_count": 421, "avg_hub_height_m": 78.0, "online_year": 2006},
        {"name": "Los Vientos Wind Farm", "county": "Starr", "lat": 26.55, "lon": -98.35,
         "capacity_mw": 912.0, "turbine_count": 400, "avg_hub_height_m": 90.0, "online_year": 2016},
        {"name": "Panhandle Wind", "county": "Carson", "lat": 35.45, "lon": -101.35,
         "capacity_mw": 400.0, "turbine_count": 133, "avg_hub_height_m": 90.0, "online_year": 2014},
        {"name": "Sherbino Wind Farm", "county": "Pecos", "lat": 30.90, "lon": -102.55,
         "capacity_mw": 300.0, "turbine_count": 150, "avg_hub_height_m": 80.0, "online_year": 2012},
        {"name": "Karankawa Wind Farm", "county": "Refugio", "lat": 28.35, "lon": -96.90,
         "capacity_mw": 300.0, "turbine_count": 90, "avg_hub_height_m": 90.0, "online_year": 2020},
    ]


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    fleet = fetch_tx_wind_fleet(force=force)
    for f in fleet[:15]:
        print(f"  {f['name']:<35} {f['capacity_mw']:>7.1f} MW  "
              f"({f['turbine_count']:>4} turbines, {f['county']} Co.)")