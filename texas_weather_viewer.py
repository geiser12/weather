#!/usr/bin/env python3
"""
Texas Multi-Model Weather Viewer – Ventusky-style WebGL edition
===============================================================
All settings are configured below – no command-line arguments needed.
Outputs (in OUTPUT_DIR):  index.html  +  grid_data.js   (works from file://)
Needs:  numpy pandas scipy requests openmeteo_requests requests_cache retry_requests
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from scripts.uswtdb_fleet import fetch_tx_wind_fleet, demo_fleet
from scripts.wind_power_model import fleet_mw_timeseries, fleet_total_capacity_mw

# ============================================================================
# SETTINGS – edit these
# ============================================================================

SPACING = 0.75                  # grid spacing in degrees (0.5 recommended; 0.75/1.0 = fewer API calls)
FORECAST_DAYS = 2              # number of days to fetch
BATCH_SIZE = 100               # points per API request (lower = safer against rate limits)
OUTPUT_DIR = "tx_model_viewer"
SLEEP_BETWEEN_BATCHES = 1.8    # seconds between successful API calls
FETCH_DEADLINE_SECONDS = 600

# Per-batch retries for timeouts / rate limits / transient network errors.
FETCH_MAX_ATTEMPTS = 6
# Minimum fraction of grid points that must have data after fetch, else fail.
FETCH_MIN_COVERAGE = 0.90

# Highs/lows: how many top / bottom points to mark, and the minimum lat/lon
# separation (degrees) between picks so they don't cluster on one bump.
HL_COUNT = 3
HL_MIN_SEP_DEG = 0.9

# Base map (Natural Earth land + Census state borders) is downloaded once and
# cached next to the script so repeated runs are fast / offline-friendly.
BASEMAP_CACHE = Path(".texas_basemap_cache.json")
BASEMAP_MAX_AGE_DAYS = 30
BASEMAP_PAD_DEG = 1.0          # geometry is clipped to the map bounds + this pad

# Synthetic data instead of the API (for testing the viewer offline).
DEMO_MODE = os.environ.get("TX_DEMO", "0") == "1"

# Wind MW Forecast tab: force a fresh USWTDB fleet fetch instead of using the
# local cache (wind_fleet_tx.json). USWTDB updates quarterly, so this rarely
# needs to be true - re-run with --refresh-fleet or set the env var below.
WIND_FLEET_FORCE_REFRESH = os.environ.get("TX_REFRESH_FLEET", "0") == "1"

# ============================================================================

TEXAS_BOUNDS = {"min_lat": 25.7, "max_lat": 36.6, "min_lon": -106.7, "max_lon": -93.4}
TIMEZONE = "America/Chicago"

MODELS = ["ecmwf_ifs", "gfs_hrrr", "ncep_nam_conus", "gfs_global"]
MODEL_LABELS = {
    "ecmwf_ifs": "ECMWF",
    "gfs_hrrr": "HRRR",
    "ncep_nam_conus": "NAM",
    "gfs_global": "GFS",
}

# Exactly 10 hourly variables (Open-Meteo counts >10 as extra API calls).
FETCH_VARS = [
    "temperature_2m", "apparent_temperature", "cloud_cover",
    "precipitation_probability", "precipitation", "wind_speed_80m",
    "wind_direction_80m", "relative_humidity_2m", "shortwave_radiation",
    "pressure_msl",
]

# Layers shown in the "Variable" dropdown (some are derived from FETCH_VARS).
LAYERS = [
    "temperature_2m", "apparent_temperature", "cloud_cover",
    "precipitation_probability", "precipitation", "precip_3h",
    "wind_speed_80m", "relative_humidity_2m", "shortwave_radiation",
    "pressure_msl",
]
# Everything shipped to the browser (layers + wind components for the flow).
ENC_VARS = LAYERS + ["wind_u", "wind_v"]

# ── Colour scales ────────────────────────────────────────────────────────
# (value, "#rrggbb" or "#rrggbbaa"). Colours are interpolated between stops
# in premultiplied RGBA, so a transparent first stop fades in cleanly.

TEMP_STOPS = [
    (-20, "#6a3d9a"), (0, "#3b4cc0"), (15, "#3a7bd5"), (25, "#38a8e8"),
    (32, "#7fd4f0"), (40, "#72e0c8"), (50, "#7fdc86"), (60, "#c2e86a"),
    (70, "#f7e45c"), (80, "#fbb03b"), (90, "#f47b2a"), (100, "#e0392a"),
    (110, "#a81d5c"), (120, "#6a1470"),
]
PRECIP_STOPS = [
    (0.0, "#7fc4ff00"), (0.05, "#8fcaff40"), (0.2, "#6db6ff99"),
    (0.5, "#3d8bffcc"), (1.0, "#2a5fe0"), (2.5, "#2bbf5a"), (5.0, "#f2e83a"),
    (10.0, "#f59a23"), (20.0, "#e0341f"), (40.0, "#a3126f"), (80.0, "#6a0dad"),
]
PRECIP_3H_STOPS = [(round(v * 2.5, 3), c) for v, c in PRECIP_STOPS]
POP_STOPS = [
    (0, "#6ec6ff00"), (10, "#6ec6ff00"), (25, "#6ec6ff66"), (40, "#3d8bff99"),
    (60, "#2a5fe0cc"), (80, "#5b2fc9ee"), (100, "#7a1fa2"),
]
CLOUD_STOPS = [
    (0, "#f4f7fa00"), (10, "#f4f7fa00"), (30, "#f4f7fa70"), (60, "#dfe6eeb8"),
    (80, "#b8c4d2e0"), (100, "#8e9db0f2"),
]
WIND_STOPS = [
    (0, "#4a6fe322"), (5, "#4aa3e3"), (10, "#3ec9a7"), (15, "#6ed35b"),
    (20, "#c9e04a"), (25, "#f5c43a"), (35, "#f28a2b"), (45, "#e0402a"),
    (60, "#a3126f"), (80, "#6a0dad"),
]
RH_STOPS = [
    (0, "#b5763a"), (20, "#d9b46a"), (35, "#e9e2a0"), (50, "#b6e3a8"),
    (65, "#7fd0c0"), (80, "#4aa3e3"), (100, "#2a4fc0"),
]
SOLAR_STOPS = [
    (0, "#ffe9a000"), (50, "#ffe9a055"), (200, "#ffd35c"), (400, "#ffb03a"),
    (600, "#f5822a"), (800, "#e0502a"), (1000, "#b81f3a"), (1200, "#7a1050"),
]
PRESSURE_STOPS = [
    (960, "#5b2c83"), (980, "#3a6fd0"), (995, "#4ab0e0"), (1005, "#9fe0d0"),
    (1013, "#f2f2c0"), (1020, "#f7d38a"), (1030, "#f0995a"), (1040, "#d0503a"),
]

# label, unit, decimals shown, stops, (quantisation lo, hi), LUT power
# (power < 1 gives the low end of the scale more colour resolution)
VAR_META = {
    "temperature_2m":            ("Temperature",          "°F",   0, TEMP_STOPS,      (-60, 140),   1.0),
    "apparent_temperature":      ("Feels Like",           "°F",   0, TEMP_STOPS,      (-80, 150),   1.0),
    "cloud_cover":               ("Cloud Cover",          "%",    0, CLOUD_STOPS,     (0, 100),     1.0),
    "precipitation_probability": ("Precip Probability",   "%",    0, POP_STOPS,       (0, 100),     1.0),
    "precipitation":             ("Precipitation (1 h)",  "mm",   1, PRECIP_STOPS,    (0, 200),     0.5),
    "precip_3h":                 ("Precipitation (3 h)",  "mm",   1, PRECIP_3H_STOPS, (0, 400),     0.5),
    "wind_speed_80m":            ("Wind Speed (80m)",           "mph",  0, WIND_STOPS,      (0, 150),     1.0),
    "relative_humidity_2m":      ("Relative Humidity",    "%",    0, RH_STOPS,        (0, 100),     1.0),
    "shortwave_radiation":       ("Solar Radiation",      "W/m²", 0, SOLAR_STOPS,     (0, 1500),    1.0),
    "pressure_msl":              ("Pressure (MSL)",       "hPa",  0, PRESSURE_STOPS,  (850, 1100),  1.0),
}
QUANT = {v: VAR_META[v][4] for v in VAR_META}
QUANT["wind_u"] = (-150.0, 150.0)
QUANT["wind_v"] = (-150.0, 150.0)

# Full labels: name + value. CITY_LABEL_OFFSETS pulls border cities' text
# inward (the value is still sampled at the true city coordinates).
CITIES = [
    ("El Paso", 31.7619, -106.4850),
    ("Amarillo", 35.2220, -101.8313),
    ("Lubbock", 33.5779, -101.8552),
    ("Midland", 31.9973, -102.0779),
    ("Wichita Falls", 33.9137, -98.4934),
    ("FW", 32.7555, -97.3308),
    ("Dallas", 32.7767, -96.7970),
    ("Waco", 31.5493, -97.1467),
    ("Austin", 30.2672, -97.7431),
    ("San Antonio", 29.4241, -98.4936),
    ("Houston", 29.7604, -95.3698),
    ("Corpus Christi", 27.8006, -97.3964),
    ("Laredo", 27.5036, -99.5076),
    ("Brownsville", 25.9017, -97.4975),
    ("Beaumont", 30.0802, -94.1266),
]
# Value-only markers (no city name drawn).
VALUE_ONLY_CITIES = [
    ("Fredericksburg", 30.2752, -98.8720),
    ("Killeen", 31.1171, -97.7278),
    ("Tyler", 32.3513, -95.3011),
    ("San Angelo", 31.4638, -100.4370),
]
CITY_LABEL_OFFSETS = {
    "El Paso": (0.15, 0.35),
    "Brownsville": (0.0, 0.35),
    "Beaumont": (-0.25, 0.15),
    "Laredo": (0.25, 0.15),
    "Corpus Christi": (0.0, 0.25),
    "Amarillo": (0.0, -0.15),
}

TIGER_URL = ("https://tigerweb.geo.census.gov/arcgis/rest/services/"
             "TIGERweb/State_County/MapServer/15/query")
NE_LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_50m_land.geojson")


# ── Grid / fetch helpers ──────────────────────────────────────────────────

def generate_grid(bounds, spacing):
    # linspace (not arange) so the last row/column lands exactly on max_lat /
    # max_lon. The browser relies on the grid spanning the bounds exactly.
    n_lat = int(np.ceil((bounds["max_lat"] - bounds["min_lat"]) / spacing)) + 1
    n_lon = int(np.ceil((bounds["max_lon"] - bounds["min_lon"]) / spacing)) + 1
    lats = np.linspace(bounds["min_lat"], bounds["max_lat"], n_lat)
    lons = np.linspace(bounds["min_lon"], bounds["max_lon"], n_lon)
    points = [(round(float(la), 4), round(float(lo), 4)) for la in lats for lo in lons]
    return points, lats, lons


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = (
        "timeout", "timed out", "timeoutreached", "rate", "limit",
        "429", "503", "502", "504", "connection", "reset", "broken pipe",
        "temporarily", "unavailable", "stream",
    )
    return any(n in msg for n in needles)


def fetch_all(points, models, variables, forecast_days, batch_size):
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".texas_weather_cache", expire_after=1800)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.4)
    client = openmeteo_requests.Client(session=retry_session)

    n_models = len(models)
    per_point = {m: {v: [None] * len(points) for v in variables} for m in models}
    timestamps_ref = None
    n_batches = math.ceil(len(points) / batch_size)
    t_fetch0 = time.time()
    failed_batches: list[int] = []

    def _remaining():
        return FETCH_DEADLINE_SECONDS - (time.time() - t_fetch0)

    def _check_deadline(where: str):
        left = _remaining()
        if left <= 0:
            raise RuntimeError(
                f"Open-Meteo fetch deadline exceeded ({FETCH_DEADLINE_SECONDS}s) at {where}. "
                "Aborting so the workflow does not hang on rate limits / empty API."
            )
        return left

    print(f"Fetching Open-Meteo ({n_batches} batches, deadline {FETCH_DEADLINE_SECONDS}s, "
          f"up to {FETCH_MAX_ATTEMPTS} attempts/batch)…")

    for batch_idx, batch in enumerate(chunked(points, batch_size)):
        _check_deadline(f"before batch {batch_idx+1}/{n_batches}")

        params = {
            "latitude": ",".join(str(p[0]) for p in batch),
            "longitude": ",".join(str(p[1]) for p in batch),
            "hourly": variables,
            "models": models,
            "timezone": TIMEZONE,
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "mm",
            "forecast_days": forecast_days,
        }

        responses = None
        last_err = None
        for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):
            _check_deadline(f"batch {batch_idx+1} attempt {attempt}")
            try:
                responses = client.weather_api(
                    "https://api.open-meteo.com/v1/forecast", params=params
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if not _is_retryable_error(e):
                    print(f"  [error] batch {batch_idx+1}/{n_batches} non-retryable: {e}")
                    break

                if "limit" in msg or "rate" in msg or "429" in msg:
                    wait = min(90 + attempt * 30, 180)
                else:
                    wait = min(5 * attempt, 45)

                left = _remaining()
                if wait > left:
                    raise RuntimeError(
                        f"Retryable error on batch {batch_idx+1}/{n_batches} "
                        f"(attempt {attempt}/{FETCH_MAX_ATTEMPTS}): {e}; "
                        f"need to wait {wait}s but only {left:.0f}s left in "
                        f"{FETCH_DEADLINE_SECONDS}s deadline. Aborting."
                    )
                print(
                    f"  [retry] batch {batch_idx+1}/{n_batches} "
                    f"attempt {attempt}/{FETCH_MAX_ATTEMPTS}: {e} "
                    f"— waiting {wait}s ({left:.0f}s left in deadline)"
                )
                time.sleep(wait)

        if responses is None:
            failed_batches.append(batch_idx + 1)
            print(
                f"  [FAILED] batch {batch_idx+1}/{n_batches} after "
                f"{FETCH_MAX_ATTEMPTS} attempts"
                + (f" (last: {last_err})" if last_err else "")
            )
            continue

        for local_idx in range(len(batch)):
            gidx = batch_idx * batch_size + local_idx
            for midx, model in enumerate(models):
                ridx = local_idx * n_models + midx
                if ridx >= len(responses):
                    continue
                try:
                    hourly = responses[ridx].Hourly()
                    idx = pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left",
                    ).tz_convert(responses[ridx].Timezone().decode()).tz_localize(None)
                    if timestamps_ref is None or len(idx) > len(timestamps_ref):
                        timestamps_ref = idx
                    for j, var in enumerate(variables):
                        vals = hourly.Variables(j).ValuesAsNumpy()
                        per_point[model][var][gidx] = pd.Series(vals, index=idx)
                except Exception:
                    pass

        print(f"  fetched batch {batch_idx+1}/{n_batches} ({_remaining():.0f}s left in deadline)")
        sleep_for = min(SLEEP_BETWEEN_BATCHES, max(0.0, _remaining() - 1.0))
        if sleep_for > 0:
            time.sleep(sleep_for)

    if timestamps_ref is None:
        raise RuntimeError("No data returned from Open-Meteo")

    sample_model, sample_var = models[0], variables[0]
    filled = sum(1 for s in per_point[sample_model][sample_var] if s is not None)
    coverage = filled / max(len(points), 1)
    print(
        f"  fetch finished in {time.time() - t_fetch0:.0f}s — "
        f"coverage {filled}/{len(points)} points ({coverage:.0%})"
    )
    if failed_batches:
        print(f"  failed batches: {failed_batches}")
    if coverage < FETCH_MIN_COVERAGE:
        raise RuntimeError(
            f"Open-Meteo coverage too low ({coverage:.0%} < {FETCH_MIN_COVERAGE:.0%}); "
            f"failed batches {failed_batches}. Not publishing incomplete maps."
        )
    return per_point, timestamps_ref


# ── Array building / cleaning ─────────────────────────────────────────────

def build_arrays(lats, lons, models, variables, per_point, timestamps_ref):
    """per_point -> {model: {var: float32 array (T, nlat, nlon)}}"""
    nlat, nlon, T = len(lats), len(lons), len(timestamps_ref)
    nan_series = pd.Series(np.nan, index=timestamps_ref)
    arrs = {m: {} for m in models}
    for m in models:
        for v in variables:
            cols = [s if s is not None else nan_series for s in per_point[m][v]]
            df = pd.concat(cols, axis=1).reindex(timestamps_ref)
            arrs[m][v] = df.to_numpy(dtype=np.float32).reshape(T, nlat, nlon)
    return arrs


def trailing_sum(p, window):
    """Trailing N-hour sum along axis 0 (partial windows at the start)."""
    fin = np.isfinite(p)
    c = np.cumsum(np.where(fin, p, 0.0), axis=0, dtype=np.float64)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return np.where(fin, out, np.nan).astype(np.float32)


def add_derived(arrs, models):
    for m in models:
        a = arrs[m]
        a["precip_3h"] = trailing_sum(a["precipitation"], 3)
        spd = a["wind_speed_80m"]
        rad = np.deg2rad(a["wind_direction_80m"])
        # meteorological convention: direction the wind blows FROM
        a["wind_u"] = (-spd * np.sin(rad)).astype(np.float32)
        a["wind_v"] = (-spd * np.cos(rad)).astype(np.float32)


def _spatial_fill(a):
    """Nearest-valid-cell fill, slice by slice (a: T, nlat, nlon)."""
    from scipy.ndimage import distance_transform_edt
    out = a.copy()
    for t in range(out.shape[0]):
        sl = out[t]
        bad = ~np.isfinite(sl)
        if not bad.any() or bad.all():
            continue
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        out[t] = sl[tuple(idx)]
    return out


def clean_arrays(arrs, models, variables):
    """Fill gaps so the GPU never sees NaN, and report what is truly missing.

    * A (model, var) with almost no finite data is reported as *missing*; the
      viewer shows "not provided by this model" instead of a fake map.
    * Isolated gaps borrow the multi-model mean, then the nearest valid cell.
    """
    missing = {m: [] for m in models}
    for var in variables:
        ok = []
        for m in models:
            if np.isfinite(arrs[m][var]).mean() < 0.02:
                missing[m].append(var)
                arrs[m][var] = np.zeros_like(arrs[m][var])
            else:
                ok.append(m)
        if not ok:
            continue
        fallback = None
        if len(ok) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fallback = np.nanmean(np.stack([arrs[m][var] for m in ok]), axis=0)
        for m in ok:
            a = arrs[m][var]
            if not np.isfinite(a).all():
                if fallback is not None:
                    a = np.where(np.isfinite(a), a, fallback)
                if not np.isfinite(a).all():
                    a = _spatial_fill(a)
                a = np.nan_to_num(a, nan=0.0)
            arrs[m][var] = a.astype(np.float32)
    return missing



def _scrape_latest_ercot_stwpf():
    """Scrape the newest ERCOT NP4-742-CD/STWPF publication using the same
    IceDocListJsonWS -> mirDownload flow used elsewhere in the ERCOT stack."""
    report_id = 14787
    try:
        list_url = (
            "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
            f"?reportTypeId={report_id}&_={int(time.time() * 1000)}"
        )
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })
        r = session.get(list_url, timeout=30)
        r.raise_for_status()
        docs = pd.json_normalize(
            r.json(),
            record_path=["ListDocsByRptTypeRes", "DocumentList"]
        )
        if docs.empty:
            return pd.DataFrame()
        docs["DocUrl"] = (
            "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId="
            + docs["Document.DocID"].astype(str)
        )
        docs["PublishDate"] = pd.to_datetime(
            docs["Document.PublishDate"], format="mixed", errors="coerce"
        )
        docs = docs[docs["Document.ConstructedName"].astype(str).str.endswith("_csv.zip")]
        now_ct = datetime.now(ZoneInfo(TIMEZONE))
        today_docs = docs[docs["PublishDate"].dt.date == now_ct.date()]
        if not today_docs.empty:
            docs = today_docs
        else:
            print("  [warn] no STWPF publication dated today; using newest available publication")
        doc_url = docs.sort_values("PublishDate", ascending=False)["DocUrl"].iloc[0]

        z = session.get(doc_url, timeout=60)
        z.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                return pd.DataFrame()
            df = pd.read_csv(io.BytesIO(zf.read(csv_files[0])))

        # Keep the parser tolerant to the exact capitalization used by the
        # published CSV while requiring the three fields we actually need.
        cols = {str(c).strip().upper(): c for c in df.columns}
        required = ["DELIVERY_DATE", "HOUR_ENDING", "STWPF_SYSTEM_WIDE"]
        if not all(c in cols for c in required):
            print(f"  [warn] ERCOT STWPF report missing columns; found: {list(df.columns)}")
            return pd.DataFrame()

        out = df[[cols[c] for c in required]].copy()
        out.columns = ["Delivery Date", "Hour Ending", "STWPF"]
        out["Delivery Date"] = pd.to_datetime(
            out["Delivery Date"], format="mixed", errors="coerce"
        )
        he = (
            out["Hour Ending"].astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        out["Hour Ending"] = pd.to_numeric(he, errors="coerce")
        out = out.dropna(subset=["Delivery Date", "Hour Ending", "STWPF"])
        out["Hour Ending"] = out["Hour Ending"].astype(int)
        out["Target Time"] = (
            out["Delivery Date"].dt.tz_localize(None)
            + pd.to_timedelta(out["Hour Ending"], unit="h")
        )
        out["STWPF"] = pd.to_numeric(out["STWPF"], errors="coerce")
        out = out.dropna(subset=["STWPF"])
        out = out.sort_values("Target Time").drop_duplicates("Target Time", keep="last")
        print(
            f"  ERCOT STWPF: {len(out)} hourly points, "
            f"{out['Target Time'].min()} -> {out['Target Time'].max()}"
        )
        return out[["Target Time", "STWPF"]].reset_index(drop=True)
    except Exception as e:
        print(f"  [warn] ERCOT STWPF scrape failed: {e}")
        return pd.DataFrame()


def _wind_farm_points(fleet):
    """Convert cached TX wind-farm list into map-ready farm points."""
    if fleet is None:
        return []

    # wind_fleet_tx.json stores farms as a list of dictionaries
    if isinstance(fleet, list):
        farms = fleet
    elif isinstance(fleet, dict):
        # Handle either {"farms": [...]} or a direct farm dictionary
        farms = fleet.get("farms", [])
        if isinstance(farms, dict):
            farms = list(farms.values())
    elif isinstance(fleet, pd.DataFrame):
        farms = fleet.to_dict("records")
    else:
        return []

    points = []

    for farm in farms:
        if not isinstance(farm, dict):
            continue

        # Case-insensitive field lookup
        lookup = {
            str(k).strip().lower(): v
            for k, v in farm.items()
        }

        def get_value(*names):
            for name in names:
                if name.lower() in lookup:
                    return lookup[name.lower()]
            return None

        lat = get_value("lat", "latitude")
        lon = get_value("lon", "longitude", "lng")

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        capacity = get_value(
            "capacity_mw",
            "capacity",
            "mw",
            "nameplate_capacity_mw",
        )

        try:
            capacity = float(capacity) if capacity is not None else 0.0
        except (TypeError, ValueError):
            capacity = 0.0

        name = get_value(
            "name",
            "project_name",
            "farm_name",
            "facility_name",
        )

        county = get_value("county")

        points.append({
            "name": str(name) if name is not None else "Wind Farm",
            "county": str(county) if county is not None else "",
            "lat": lat,
            "lon": lon,
            "capacity_mw": capacity,
        })

    return points

def build_wind_forecast_payload(arrs, lats, lons, timestamps, models):
    """Build the main-page wind MW chart using the Open-Meteo window and
    the latest ERCOT STWPF forecast, with all hours aligned to HE labels."""
    print("Wind MW Forecast: loading TX wind fleet (USWTDB)...")

    try:
        fleet = demo_fleet() if DEMO_MODE else fetch_tx_wind_fleet(
            force=WIND_FLEET_FORCE_REFRESH
        )
    except Exception as e:
        print(f"  [warn] could not load wind fleet ({e}); wind chart will be empty")
        return None

    total_cap = fleet_total_capacity_mw(fleet)

    # Open-Meteo is the authoritative forecast window
    weather_dt = pd.DatetimeIndex(pd.to_datetime(timestamps))
    weather_times = weather_dt.strftime("%Y-%m-%dT%H:%M").tolist()
    weather_start = weather_dt.min()
    weather_end = weather_dt.max()

    series = {}

    for m in models:
        wind_field = arrs[m]["wind_speed_80m"]
        mw = fleet_mw_timeseries(wind_field, lats, lons, fleet)
        series[MODEL_LABELS.get(m, m)] = [
            round(float(x), 1) for x in mw
        ]

    # ERCOT STWPF
    ercot = pd.DataFrame()

    if not DEMO_MODE:
        ercot = _scrape_latest_ercot_stwpf()

    ercot_times = []
    ercot_values = []

    if not ercot.empty:
        ercot = ercot.copy()
        ercot["Target Time"] = pd.to_datetime(
            ercot["Target Time"], errors="coerce"
        )
        ercot["STWPF"] = pd.to_numeric(
            ercot["STWPF"], errors="coerce"
        )
        ercot = ercot.dropna(subset=["Target Time", "STWPF"])

        # ERCOT STWPF labels: Hour Ending N is stored as clock time N:00
        # (HE1→01:00, HE21→21:00, HE24→00:00 next day).
        # Open-Meteo / map HE convention uses the *start* of the hour
        # (HE21 → 20:00). Shift ERCOT back 1 hour so series align on the
        # same HE label on the chart.
        ercot["Target Time"] = ercot["Target Time"] - pd.Timedelta(hours=1)

        # Clip ERCOT to the exact Open-Meteo forecast window.
        before_clip = len(ercot)

        ercot = ercot[
            ercot["Target Time"].between(
                weather_start,
                weather_end,
                inclusive="both",
            )
        ].copy()

        print(
            f"  ERCOT STWPF clipped to Open-Meteo window: "
            f"{len(ercot)}/{before_clip} hours"
        )

        if not ercot.empty:
            print(
                f"    window: "
                f"{ercot['Target Time'].min()} -> "
                f"{ercot['Target Time'].max()}"
            )

            ercot_times = ercot["Target Time"].dt.strftime(
                "%Y-%m-%dT%H:%M"
            ).tolist()

            ercot_values = [
                round(float(x), 1)
                for x in ercot["STWPF"]
            ]

    # Use ONLY the Open-Meteo hours as the chart time axis.
    # ERCOT is now clipped to this same window.
    all_times = weather_times

    # Align every weather model to the common weather-time axis
    for name in list(series):
        lookup = dict(zip(weather_times, series[name]))
        series[name] = [lookup.get(t) for t in all_times]

    # Align ERCOT to the same weather-time axis
    if ercot_times:
        lookup = dict(zip(ercot_times, ercot_values))
        series["ERCOT STWPF"] = [
            lookup.get(t) for t in all_times
        ]

    farm_points = _wind_farm_points(fleet)

    print(
        f"  wind forecast: {len(all_times)} hours, "
        f"{len(farm_points)} farms, "
        f"{total_cap:,.0f} MW fleet capacity"
    )

    return {
        "times": all_times,
        "series": series,
        "fleetCapacityMw": round(total_cap, 1),
        "fleetFarmCount": len(farm_points),
        "windFarms": farm_points,
        "ercotAvailable": bool(ercot_times),
        "weatherHours": len(weather_times),
        "note": (
            "Weather-model lines are model-derived estimates using the USWTDB TX "
            "wind-farm inventory and a generic power curve. ERCOT STWPF is the "
            "published system-wide wind forecast scraped from the latest ERCOT "
            "NP4-742-CD publication."
        ),
    }

def quantize(a, lo, hi):
    q = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 65535.0
    return np.rint(q).astype("<u2")


def export_grid_data(out_dir, arrs, models, enc_vars):
    """One base64 blob, layout: [model][var][time][lat][lon] as uint16 LE."""
    chunks = []
    for m in models:
        for v in enc_vars:
            lo, hi = QUANT[v]
            chunks.append(np.ascontiguousarray(quantize(arrs[m][v], lo, hi)).tobytes())
    blob = b"".join(chunks)
    path = out_dir / "grid_data.js"
    path.write_text('window.GRID_B64="' + base64.b64encode(blob).decode("ascii") + '";',
                    encoding="ascii")
    print(f"  Wrote {path.name} ({path.stat().st_size / 1048576:.1f} MB)")


# ── Base map (Natural Earth land + Census state borders) ─────────────────

def _flat_ring(coords):
    flat, last = [], None
    for c in coords:
        p = (round(float(c[0]), 3), round(float(c[1]), 3))
        if p != last:
            flat.extend(p)
            last = p
    return flat


def _geometry_rings(geom):
    if not geom:
        return []
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return []
    return [ring for poly in polys for ring in poly]


def _clip_ring(ring, xmin, ymin, xmax, ymax):
    """Sutherland–Hodgman clip of one ring to a rectangle."""
    def clip_edge(pts, inside, intersect):
        out = []
        if not pts:
            return out
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def ix(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def iy(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    pts = [(p[0], p[1]) for p in ring]
    pts = clip_edge(pts, lambda p: p[0] >= xmin, ix(xmin))
    pts = clip_edge(pts, lambda p: p[0] <= xmax, ix(xmax))
    pts = clip_edge(pts, lambda p: p[1] >= ymin, iy(ymin))
    pts = clip_edge(pts, lambda p: p[1] <= ymax, iy(ymax))
    return pts


def _clipped_flat_rings(rings, window):
    xmin, ymin, xmax, ymax = window
    out = []
    for ring in rings:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if max(xs) < xmin or min(xs) > xmax or max(ys) < ymin or min(ys) > ymax:
            continue
        clipped = _clip_ring(ring, xmin, ymin, xmax, ymax)
        if len(clipped) >= 3:
            out.append(_flat_ring(clipped))
    return out


def _tiger_features(where, offset=None):
    params = {"where": where, "outFields": "STATE", "returnGeometry": "true",
              "f": "geojson", "outSR": "4326"}
    if offset:
        params["maxAllowableOffset"] = offset
    r = requests.get(TIGER_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["features"]


def load_basemap():
    b = TEXAS_BOUNDS
    key = {"bounds": b, "pad": BASEMAP_PAD_DEG, "v": 1}
    if BASEMAP_CACHE.exists():
        try:
            cached = json.loads(BASEMAP_CACHE.read_text())
            age_days = (time.time() - BASEMAP_CACHE.stat().st_mtime) / 86400
            if cached.get("key") == key and age_days < BASEMAP_MAX_AGE_DAYS:
                print("  base map: using cache")
                return cached["data"]
        except Exception:
            pass

    window = (b["min_lon"] - BASEMAP_PAD_DEG, b["min_lat"] - BASEMAP_PAD_DEG,
              b["max_lon"] + BASEMAP_PAD_DEG, b["max_lat"] + BASEMAP_PAD_DEG)
    all_ok = True

    try:
        feats = _tiger_features("STATE='48'", offset=0.004)
        texas = [_flat_ring(r) for f in feats for r in _geometry_rings(f["geometry"])]
        if not texas:
            raise ValueError("empty Texas geometry")
    except Exception as e:
        print(f"[warn] Texas boundary download failed: {e} — using bounding box")
        all_ok = False
        texas = [[b["min_lon"], b["min_lat"], b["max_lon"], b["min_lat"],
                  b["max_lon"], b["max_lat"], b["min_lon"], b["max_lat"],
                  b["min_lon"], b["min_lat"]]]

    try:
        feats = _tiger_features("STATE IN ('40','35','22','05')", offset=0.01)
        rings = [r for f in feats for r in _geometry_rings(f["geometry"])]
        states = _clipped_flat_rings(rings, window)
    except Exception as e:
        print(f"[warn] Neighbour-state download failed: {e}")
        all_ok = False
        states = []

    try:
        r = requests.get(NE_LAND_URL, timeout=90)
        r.raise_for_status()
        rings = [ring for f in r.json()["features"] for ring in _geometry_rings(f["geometry"])]
        land = _clipped_flat_rings(rings, window)
    except Exception as e:
        print(f"[warn] Land polygons download failed: {e} — plain background")
        all_ok = False
        land = []

    data = {"texas": texas, "states": states, "land": land}
    if all_ok:
        try:
            BASEMAP_CACHE.write_text(json.dumps({"key": key, "data": data}))
        except Exception:
            pass
    return data


def points_in_rings(px, py, rings):
    """Vectorised even-odd point-in-polygon over many rings."""
    inside = np.zeros(px.shape, dtype=bool)
    for ring in rings:
        x0, y0 = ring[:, 0], ring[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        for xa, ya, xb, yb in zip(x0, y0, x1, y1):
            if ya == yb:
                continue
            cond = (ya > py) != (yb > py)
            if not cond.any():
                continue
            xint = xa + (py - ya) * (xb - xa) / (yb - ya)
            inside ^= cond & (px < xint)
    return inside


def compute_inside_mask(lats, lons, texas_rings, spacing):
    """Grid cells inside Texas (used for H/L). A small tolerance keeps coastal
    and border cells whose centre grazes the outline."""
    LON, LAT = np.meshgrid(lons, lats)
    px, py = LON.ravel(), LAT.ravel()
    r = 0.35 * max(spacing, 0.5)
    offs = [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)]
    ax = np.concatenate([px + dx for dx, _ in offs])
    ay = np.concatenate([py + dy for _, dy in offs])
    inside = points_in_rings(ax, ay, texas_rings).reshape(len(offs), -1).any(axis=0)
    return inside.reshape(LON.shape)


# ── Synthetic data (TX_DEMO=1) ────────────────────────────────────────────

def make_demo_data(lats, lons, models):
    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    times = pd.date_range(start, periods=24 * FORECAST_DAYS, freq="h")
    T = len(times)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    hrs = np.arange(T)
    tt = hrs[:, None, None].astype(float)
    h = (hrs % 24)[:, None, None].astype(float)
    out = {m: {} for m in models}
    for mi, m in enumerate(models):
        ph = 0.35 * mi
        temp = 82 - 2.0 * (LA - 25.7) + 5 * np.sin((LO + 100) / 2.2 + ph) \
            + 14 * np.sin((h - 9) / 24 * 2 * np.pi)
        cloud = np.clip(45 + 45 * np.sin(LO / 1.7 + LA / 2.3 + tt / 7 + ph), 0, 100)
        precip = np.maximum(0, np.sin(LO * 1.1 + tt * 0.12 + ph) * np.cos(LA * 1.3 - tt * 0.05) - 0.55) * 14
        pop = np.clip(precip * 35 + 1.5 * np.abs(np.sin(LO * 3 + LA * 2)), 0, 100)
        spd = np.clip(9 + 7 * np.sin(LA * 0.9 + tt / 6 + ph) + 4 * np.cos(LO * 0.7), 0, None)
        wdir = (190 + 70 * np.sin(LO / 2.5 + ph) + 40 * np.cos(LA / 2 + tt / 9)) % 360
        rh = np.clip(65 + 25 * np.cos(LO / 2 + ph) - 15 * np.sin((h - 15) / 24 * 2 * np.pi), 5, 100)
        solar = np.maximum(0, 950 * np.sin(np.pi * (h - 6.5) / 13)) * (1 - cloud / 130)
        pres = 1014 + 7 * np.sin(LO / 3 + tt / 24 + ph) + 5 * np.cos(LA / 2.5)
        app = temp + 0.05 * (rh - 50) - 0.1 * spd
        d = {
            "temperature_2m": temp, "apparent_temperature": app, "cloud_cover": cloud,
            "precipitation_probability": pop, "precipitation": precip,
            "wind_speed_80m": spd, "wind_direction_80m": wdir,
            "relative_humidity_2m": rh, "shortwave_radiation": solar, "pressure_msl": pres,
        }
        for k, v in d.items():
            out[m][k] = np.broadcast_to(v, (T, len(lats), len(lons))).astype(np.float32).copy()
    # Exercise the "model doesn't provide this variable" path.
    out[models[1]]["precipitation_probability"][:] = np.nan
    return out, times


# ── HTML viewer ───────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Texas Multi-Model Weather Viewer</title>
<style>
:root {
  --bg: #0b0f14; --panel: #111821; --border: #263342;
  --text: #e8edf2; --muted: #93a0ae; --accent: #5ba7ff;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
             font-family: "Segoe UI", system-ui, Arial, sans-serif; overflow: hidden; }
body { display: flex; flex-direction: column; }
header {
  flex: none; height: 52px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px; background: var(--panel); border-bottom: 1px solid var(--border);
}
.title { font-size: 17px; font-weight: 650; }
.subtitle { font-size: 11px; color: var(--muted); margin-top: 2px; }
.tabbar { display: flex; gap: 4px; }
.tabbar button {
  background: transparent; border: 1px solid transparent; color: var(--muted);
  border-radius: 6px; padding: 7px 13px; font-size: 12.5px; font-weight: 600;
  cursor: pointer; font-family: inherit;
}
.tabbar button:hover { color: var(--text); }
.tabbar button.active { background: #1a2430; border-color: #334253; color: var(--text); }

#mapsView { flex: 1; min-height: 0; display: flex; flex-direction: column; }
#windDock { display: none; flex: 0 0 35%; min-height: 170px; background: var(--panel);
  border-top: 1px solid var(--border); position: relative; flex-direction: column; padding: 7px 18px 4px; }
#windDock.on { display: flex; }
#windDock.expanded {
  position: fixed; inset: 0; z-index: 80; flex: none !important;
  min-height: 0; padding: 14px 20px 12px;
  border: 0; background: var(--bg);
  box-shadow: 0 0 0 1px var(--border);
}
#windDock.expanded #windChartWrap { min-height: 0; flex: 1; }
#windDock.expanded .wind-note { white-space: normal; overflow: visible; text-overflow: unset; max-width: 70%; }
.wind-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 20px; }
.wind-head h2 { font-size: 13px; margin: 0; font-weight: 650; }
.wind-head .meta { font-size: 10.5px; color: var(--muted); }
.wind-head-right { display: flex; align-items: center; gap: 10px; }
#windExpandBtn {
  background: #1a2430; color: var(--text); border: 1px solid #334253;
  border-radius: 5px; padding: 4px 10px; font-size: 11px; font-weight: 600;
  cursor: pointer; font-family: inherit; white-space: nowrap;
}
#windExpandBtn:hover { border-color: var(--accent); color: var(--accent); }
.wind-note { font-size: 10px; color: var(--muted); line-height: 1.35; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#windChartWrap { position: relative; flex: 1; min-height: 120px; }
#windChart { position: absolute; inset: 0; width: 100%; height: 100%; }
#windChartTip { display: none; position: absolute; z-index: 5; pointer-events: none;
  background: rgba(17,24,33,.96); border: 1px solid var(--border); border-radius: 6px;
  padding: 7px 9px; font-size: 10.5px; white-space: nowrap; box-shadow: 0 6px 18px rgba(0,0,0,.35); }
.wind-legend { display: flex; gap: 7px 14px; flex-wrap: wrap; align-items: center; min-height: 22px; }
.wind-legend .sw { display: inline-flex; align-items: center; gap: 5px; font-size: 10.5px; color: var(--text); border: 0;
  background: transparent; padding: 2px 0; cursor: pointer; font-family: inherit; }
.wind-legend .sw.off { opacity: .35; }
.wind-legend .dot { width: 9px; height: 3px; border-radius: 2px; display: inline-block; }
.controls {
  flex: none; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px 14px;
  padding: 8px 18px; background: #151e29; border-bottom: 1px solid var(--border);
}
.controls > div { display: flex; flex-direction: column; gap: 3px; }
label.lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
select, button {
  background: #1a2430; color: var(--text); border: 1px solid #334253;
  border-radius: 5px; padding: 6px 11px; font-size: 12px; cursor: pointer; font-family: inherit;
}
button:hover, select:hover { border-color: var(--accent); }
select:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
#modelSelects { display: flex !important; flex-direction: row !important; align-items: flex-end; gap: 10px; }
.model-select-wrap { display: flex; flex-direction: column; gap: 3px; }
.model-select-wrap.hidden { display: none; }
.model-select-wrap select:disabled { opacity: 0.7; cursor: not-allowed; }
.toggles { display: flex !important; flex-direction: row !important; gap: 12px; align-items: center; height: 30px; }
.toggles label { font-size: 12px; color: var(--text); display: flex; align-items: center; gap: 5px; cursor: pointer; white-space: nowrap; }
.toggles input { accent-color: var(--accent); margin: 0; }
#opacityRange { width: 90px; accent-color: var(--accent); }
#probeHint { font-size: 11px; color: var(--muted); align-self: center; }

#maps {
  flex: 1; min-height: 0; display: grid; gap: 3px; background: #1c252f; padding: 3px;
}
#maps.layout-1 { grid-template-columns: minmax(0,1fr); grid-template-rows: minmax(0,1fr); }
#maps.layout-2 { grid-template-columns: repeat(2, minmax(0,1fr)); grid-template-rows: minmax(0,1fr); }
#maps.layout-3 { grid-template-columns: repeat(3, minmax(0,1fr)); grid-template-rows: minmax(0,1fr); }
#maps.layout-4 { grid-template-columns: repeat(2, minmax(0,1fr)); grid-template-rows: repeat(2, minmax(0,1fr)); }
.panel {
  position: relative; background: #0d131a; overflow: hidden; min-width: 0; min-height: 0;
  display: flex; align-items: center; justify-content: center;
}
.panel.hidden { display: none; }
.mapbox { position: relative; flex: none; cursor: crosshair; }
.mapbox canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; display: block; }

.legend {
  flex: none; height: 44px; padding: 4px 18px 0; background: var(--panel);
  border-top: 1px solid var(--border); display: flex; align-items: center; gap: 14px;
}
#legendLabel { font-size: 12px; color: var(--muted); min-width: 150px; white-space: nowrap; }
#legend { flex: 1; height: 36px; max-width: 900px; display: block; }
.timeline {
  flex: none; height: 78px; padding: 8px 18px; background: var(--panel);
  border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 4px;
}
#timeLabel { font-size: 13px; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--accent); margin: 0; display: block; }
.hour-ticks {
  display: flex; justify-content: space-between; color: var(--muted);
  font-size: 9px; margin-top: 2px; padding: 0 10px 0 14px; box-sizing: border-box;
}
.hour-ticks span { min-width: 2.4em; text-align: center; }
#probe {
  display: none; position: fixed; z-index: 50; min-width: 210px; max-width: 290px;
  background: rgba(17, 24, 33, 0.96); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; pointer-events: none;
  box-shadow: 0 8px 28px rgba(0,0,0,0.45); font-size: 12px;
}

#windDock.on + .legend { }
@media (max-height: 720px) and (min-width: 769px) {
  #windDock { flex-basis: 22%; min-height: 145px; }
}

/* ── Mobile / narrow screens ─────────────────────────────────────── */
@media (max-width: 768px) {
  html, body {
    overflow: auto;                 /* allow vertical scroll */
    height: auto;
    min-height: 100%;
  }
  body {
    display: block;                 /* drop the flex column that was locking heights */
  }

  header {
    height: auto;
    padding: 10px 12px;
    flex-wrap: wrap;
    gap: 4px;
  }
  .title { font-size: 15px; }
  .subtitle { font-size: 10px; }

  .controls {
    padding: 8px 12px;
    gap: 8px 10px;
    /* keep flex-wrap; just give it room to grow */
  }
  .controls > div {
    min-width: 0;
  }
  /* Make the four model selects stack nicer */
  #modelSelects {
    flex-wrap: wrap;
    width: 100%;
  }
  .model-select-wrap {
    flex: 1 1 45%;
    min-width: 120px;
  }
  .toggles {
    flex-wrap: wrap;
    height: auto;
    row-gap: 6px;
  }
  #opacityRange { width: 70px; }

  /* Maps area – give it a sensible minimum height so it doesn't collapse */
  #maps {
    min-height: 52vh;
    height: 52vh;
  }
  #windDock.on { min-height: 190px; flex-basis: 26%; }
  #maps.layout-2,
  #maps.layout-3,
  #maps.layout-4 {
    grid-template-columns: 1fr;     /* single column on phones */
    grid-template-rows: repeat(auto-fit, minmax(180px, 1fr));
  }
  /* When user picks 1-panel it already looks good */

  .legend {
    height: auto;
    padding: 6px 12px;
    flex-wrap: wrap;
    gap: 6px;
  }
  #legendLabel { min-width: 0; font-size: 11px; }
  #legend { max-width: none; width: 100%; height: 32px; }

  .timeline {
    height: auto;
    padding: 10px 12px 14px;
    position: sticky;
    bottom: 0;
    z-index: 20;
    background: var(--panel);
    border-top: 1px solid var(--border);
    box-shadow: 0 -4px 16px rgba(0,0,0,0.35);
  }
  #timeLabel { font-size: 12px; }
  .hour-ticks { font-size: 8px; padding: 0 6px; }

  /* Probe needs a bit more room on small screens */
  #probe {
    max-width: min(290px, 92vw);
    font-size: 11px;
  }

  /* Hide the least-critical desktop-only hint */
  #probeHint { display: none; }
}

/* Extra-small phones */
@media (max-width: 420px) {
  .controls {
    gap: 6px 8px;
  }
  select, button {
    padding: 5px 8px;
    font-size: 11px;
  }
  .model-select-wrap {
    flex: 1 1 100%;
  }
  #maps {
    min-height: 46vh;
    height: 46vh;
  }
  #windDock.on { min-height: 190px; flex-basis: 28%; }
}
#probe .probe-title { font-weight: 650; font-size: 12px; margin-bottom: 2px; color: var(--accent); }
#probe .probe-loc { color: var(--muted); font-size: 10px; margin-bottom: 8px; font-variant-numeric: tabular-nums; }
#probe table { width: 100%; border-collapse: collapse; }
#probe td { padding: 2px 0; }
#probe td.k { color: var(--muted); padding-right: 10px; }
#probe td.v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
#fatal { display: none; position: fixed; inset: 0; z-index: 100; background: rgba(11,15,20,.96);
        align-items: center; justify-content: center; text-align: center; padding: 30px; font-size: 14px; line-height: 1.6; }
</style>
</head>
<body>
<header>
  <div>
    <div class="title">Texas Multi-Model Weather Viewer</div>
    <div class="subtitle" id="subtitle"></div>
  </div>
  <div id="lastUpdated" style="font-size:11px; color:var(--muted);"></div>
</header>

<div id="mapsView">
<div class="controls">
  <div><label class="lbl" for="varSelect">Variable</label><select id="varSelect"></select></div>
  <div>
    <label class="lbl" for="layoutSelect">Layout</label>
    <select id="layoutSelect">
      <option value="1">1 panel</option>
      <option value="2">2 panels</option>
      <option value="3" selected>3 panels</option>
      <option value="4">4 panels</option>
    </select>
  </div>
  <div id="modelSelects">
    <div class="model-select-wrap" id="modelWrap0"><label class="lbl" for="modelSelect0">Model 1</label><select id="modelSelect0"></select></div>
    <div class="model-select-wrap" id="modelWrap1"><label class="lbl" for="modelSelect1">Model 2</label><select id="modelSelect1"></select></div>
    <div class="model-select-wrap" id="modelWrap2"><label class="lbl" for="modelSelect2">Model 3</label><select id="modelSelect2"></select></div>
    <div class="model-select-wrap" id="modelWrap3"><label class="lbl" for="modelSelect3">Model 4</label><select id="modelSelect3"></select></div>
  </div>
  <div>
    <label class="lbl" for="outsideSelect">Outside Texas</label>
    <select id="outsideSelect">
      <option value="show">Show</option>
      <option value="dim" selected>Dim</option>
      <option value="hide">Hide</option>
    </select>
  </div>
  <div class="toggles">
    <label><input type="checkbox" id="chkFlow" checked> Wind flow</label>
    <label><input type="checkbox" id="chkIso"> Isobars</label>
    <label><input type="checkbox" id="chkIsot"> Isotherms</label>
    <label><input type="checkbox" id="chkCities" checked> Values</label>
    <label><input type="checkbox" id="chkHL" checked> H/L</label>
    <label><input type="checkbox" id="chkWindFarms" checked> Wind Farms</label>
    <label><input type="checkbox" id="chkWindMW"> Show Wind Forecast MW</label>
  </div>
  <div><label class="lbl" for="opacityRange">Layer opacity</label><input type="range" id="opacityRange" min="20" max="100" value="100"></div>
  <div>
    <label class="lbl" for="speedSelect">Speed</label>
    <select id="speedSelect">
      <option value="0.7">Slow</option>
      <option value="1.4" selected>Normal</option>
      <option value="2.8">Fast</option>
    </select>
  </div>
  <div><label class="lbl">&nbsp;</label><button id="playBtn">▶ Play</button></div>
  <span id="probeHint">Click a map to probe values</span>
</div>

<div id="maps" class="layout-4">
  <div class="panel" id="panel0"><div class="mapbox"><canvas class="c-base"></canvas><canvas class="c-gl"></canvas><canvas class="c-flow"></canvas><canvas class="c-over"></canvas></div></div>
  <div class="panel" id="panel1"><div class="mapbox"><canvas class="c-base"></canvas><canvas class="c-gl"></canvas><canvas class="c-flow"></canvas><canvas class="c-over"></canvas></div></div>
  <div class="panel" id="panel2"><div class="mapbox"><canvas class="c-base"></canvas><canvas class="c-gl"></canvas><canvas class="c-flow"></canvas><canvas class="c-over"></canvas></div></div>
  <div class="panel" id="panel3"><div class="mapbox"><canvas class="c-base"></canvas><canvas class="c-gl"></canvas><canvas class="c-flow"></canvas><canvas class="c-over"></canvas></div></div>
</div>

<div id="windDock">
  <div class="wind-head">
    <h2>Wind MW Forecast</h2>
    <div class="wind-head-right">
      <span class="meta" id="windMeta"></span>
      <button type="button" id="windExpandBtn" title="Expand chart (Esc to exit)">Expand</button>
    </div>
  </div>
  <div class="wind-note" id="windNote"></div>
  <div id="windChartWrap">
    <canvas id="windChart"></canvas>
    <div id="windChartTip"></div>
  </div>
  <div class="wind-legend" id="windLegend"></div>
</div>

<div class="legend"><span id="legendLabel"></span><canvas id="legend"></canvas></div>

<div class="timeline">
  <span id="timeLabel"></span>
  <div>
    <input type="range" id="hourSlider" min="0" max="1" value="0" step="1" aria-label="Hour">
    <div class="hour-ticks" id="hourTicks">
      <span>HE 1</span><span>HE 7</span><span>HE 13</span><span>HE 19</span><span>HE 24</span>
    </div>
  </div>
</div>
</div>

<div id="probe"></div>
<div id="fatal"></div>

<script src="grid_data.js"></script>
<script>
(function () {
"use strict";
const DATA = __DATA_JSON__;

/* ── constants & data ─────────────────────────────────────────────── */
const B = DATA.bounds;
const DLON = B.max_lon - B.min_lon, DLAT = B.max_lat - B.min_lat;
const ASPECT = (DLON * Math.cos((B.min_lat + B.max_lat) / 2 * Math.PI / 180)) / DLAT;
const NLAT = DATA.nlat, NLON = DATA.nlon, NCELL = NLAT * NLON;
const T = DATA.times.length;
const MODELS = DATA.models, DVARS = DATA.dataVars, VARS = DATA.variables;
const META = DATA.varMeta, QUANT = DATA.quant, BASE = DATA.base;
const MASK = Uint8Array.from(DATA.mask);
const DVIDX = {}; DVARS.forEach((v, i) => DVIDX[v] = i);
const FONT = '"Segoe UI", system-ui, Arial, sans-serif';

function fatal(msg) {
  const el = document.getElementById("fatal");
  el.innerHTML = "<div>" + msg + "</div>"; el.style.display = "flex";
}
if (!window.GRID_B64) { fatal("Could not load <b>grid_data.js</b>. Keep it in the same folder as index.html."); return; }

let U16;
(function decode() {
  const s = atob(window.GRID_B64);
  const u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  U16 = new Uint16Array(u8.buffer);
  window.GRID_B64 = null;
})();

const $ = id => document.getElementById(id);
const varSelect = $("varSelect"), layoutSelect = $("layoutSelect");
const hourSlider = $("hourSlider"), timeLabel = $("timeLabel"), playBtn = $("playBtn");
const hourTicks = $("hourTicks");
const probeEl = $("probe"), mapsEl = $("maps"), legendCv = $("legend"), legendLabel = $("legendLabel");
const speedSelect = $("speedSelect"), outsideSelect = $("outsideSelect"), opacityRange = $("opacityRange");
const chkWindFarms = $("chkWindFarms"), chkWindMW = $("chkWindMW"), windDock = $("windDock"), windChart = $("windChart");
const windChartWrap = $("windChartWrap"), windChartTip = $("windChartTip"), windLegend = $("windLegend");
const windMeta = $("windMeta"), windNote = $("windNote"), windExpandBtn = $("windExpandBtn");
const modelSelects = [0,1,2,3].map(i => $("modelSelect" + i));
const modelWraps = [0,1,2,3].map(i => $("modelWrap" + i));

/* timeline */
const dates = [], dayIdx = {}, DATE_OF = [], HOUR_OF = [];
DATA.times.forEach((s, t) => {
  const d = s.slice(0, 10);
  if (!(d in dayIdx)) { dayIdx[d] = []; dates.push(d); }
  dayIdx[d].push(t); DATE_OF.push(d); HOUR_OF.push(+s.slice(11, 13));
});

/* state */
let currentVar = "wind_speed_80m", tPos = 0, layoutCount = 4;
let panelModels = MODELS.slice();
let playing = false, rafId = null, lastFrame = 0, lastOverlayT = -99, hoursPerSec = 1.4;
const opts = { flow: true, iso: false, isot: false, cities: true, hl: true, windFarms: true, outside: "dim", opacity: 1 };

/* ── data access ──────────────────────────────────────────────────── */
function isMissing(model, v) { return (DATA.missing[model] || []).indexOf(v) >= 0; }

function getField(model, v, t, out) {
  const block = (MODELS.indexOf(model) * DVARS.length + DVIDX[v]) * T * NCELL;
  const t0 = Math.max(0, Math.min(T - 1, Math.floor(t)));
  const t1 = Math.min(T - 1, t0 + 1);
  const w = Math.max(0, Math.min(1, t - t0));
  const lo = QUANT[v][0], sc = (QUANT[v][1] - QUANT[v][0]) / 65535;
  const a = block + t0 * NCELL, b = block + t1 * NCELL;
  const f = out || new Float32Array(NCELL);
  for (let i = 0; i < NCELL; i++) {
    const qa = U16[a + i], qb = U16[b + i];
    f[i] = lo + (qa + (qb - qa) * w) * sc;
  }
  return f;
}

function sample(f, lon, lat) {
  const gx = (lon - B.min_lon) / DLON * (NLON - 1), gy = (lat - B.min_lat) / DLAT * (NLAT - 1);
  let i0 = Math.max(0, Math.min(NLON - 2, Math.floor(gx))), j0 = Math.max(0, Math.min(NLAT - 2, Math.floor(gy)));
  const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
  const a = j0 * NLON + i0;
  return (f[a] * (1 - fx) + f[a + 1] * fx) * (1 - fy) + (f[a + NLON] * (1 - fx) + f[a + NLON + 1] * fx) * fy;
}

function sampleQ(model, v, t, lon, lat) {
  if (isMissing(model, v)) return null;
  const base = (MODELS.indexOf(model) * DVARS.length + DVIDX[v]) * T * NCELL + t * NCELL;
  const gx = (lon - B.min_lon) / DLON * (NLON - 1), gy = (lat - B.min_lat) / DLAT * (NLAT - 1);
  let i0 = Math.max(0, Math.min(NLON - 2, Math.floor(gx))), j0 = Math.max(0, Math.min(NLAT - 2, Math.floor(gy)));
  const fx = Math.max(0, Math.min(1, gx - i0)), fy = Math.max(0, Math.min(1, gy - j0));
  const a = base + j0 * NLON + i0;
  const q = (U16[a] * (1 - fx) + U16[a + 1] * fx) * (1 - fy) + (U16[a + NLON] * (1 - fx) + U16[a + NLON + 1] * fx) * fy;
  return QUANT[v][0] + q * (QUANT[v][1] - QUANT[v][0]) / 65535;
}

function fmtV(v, d) {
  let s = v.toFixed(d);
  if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
  return s;
}
function fmtStop(v) { return (Math.abs(v) >= 10 || Number.isInteger(v)) ? String(Math.round(v * 10) / 10) : String(+v.toFixed(2)); }
function compass(deg) {
  return ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"][Math.round(deg / 22.5) % 16];
}

/* ── colour LUTs ──────────────────────────────────────────────────── */
const LUT_N = 2048;
const lutCache = {};
function hexRGBA(h) {
  h = h.replace("#", "");
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16),
          h.length >= 8 ? parseInt(h.slice(6,8),16) / 255 : 1];
}
function cssColor(h) { const c = hexRGBA(h); return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + c[3].toFixed(3) + ")"; }
function buildLut(v) {
  if (lutCache[v]) return lutCache[v];
  const stops = META[v].stops.map(s => [s[0], hexRGBA(s[1])]);
  const vmin = stops[0][0], vmax = stops[stops.length - 1][0], pw = META[v].pow || 1;
  const u8 = new Uint8Array(LUT_N * 4);
  let seg = 0;
  for (let i = 0; i < LUT_N; i++) {
    const val = vmin + (vmax - vmin) * Math.pow(i / (LUT_N - 1), 1 / pw);
    while (seg < stops.length - 2 && val > stops[seg + 1][0]) seg++;
    const s0 = stops[seg], s1 = stops[seg + 1];
    const f = Math.max(0, Math.min(1, (val - s0[0]) / (s1[0] - s0[0])));
    const a0 = s0[1][3], a1 = s1[1][3];
    for (let k = 0; k < 3; k++) {          /* premultiplied interpolation */
      u8[i * 4 + k] = Math.round(s0[1][k] * a0 + (s1[1][k] * a1 - s0[1][k] * a0) * f);
    }
    u8[i * 4 + 3] = Math.round((a0 + (a1 - a0) * f) * 255);
  }
  return lutCache[v] = { u8: u8, vmin: vmin, vmax: vmax, pow: pw };
}

/* ── geometry helpers ─────────────────────────────────────────────── */
function tracePath(ctx, rings, X, Y) {
  for (const r of rings) {
    ctx.moveTo(X(r[0]), Y(r[1]));
    for (let i = 2; i < r.length; i += 2) ctx.lineTo(X(r[i]), Y(r[i + 1]));
    ctx.closePath();
  }
}
function haloText(ctx, txt, x, y, font, fill, halo) {
  ctx.font = font; ctx.lineJoin = "round"; ctx.lineWidth = 3;
  ctx.strokeStyle = halo || "rgba(255,255,255,0.88)"; ctx.strokeText(txt, x, y);
  ctx.fillStyle = fill; ctx.fillText(txt, x, y);
}

/* Catmull-Rom upsample, clamped to the local range (no ringing) */
function upsampleCR(f, K) {
  const NX = (NLON - 1) * K + 1, NY = (NLAT - 1) * K + 1, out = new Float32Array(NX * NY);
  const at = (i, j) => f[Math.max(0, Math.min(NLAT - 1, j)) * NLON + Math.max(0, Math.min(NLON - 1, i))];
  const w = t => { const t2 = t * t, t3 = t2 * t;
    return [-0.5*t3 + t2 - 0.5*t, 1.5*t3 - 2.5*t2 + 1, -1.5*t3 + 2*t2 + 0.5*t, 0.5*t3 - 0.5*t2]; };
  for (let oy = 0; oy < NY; oy++) {
    const gy = oy / K; let j0 = Math.min(NLAT - 2, Math.floor(gy)); const wy = w(gy - j0);
    for (let ox = 0; ox < NX; ox++) {
      const gx = ox / K; let i0 = Math.min(NLON - 2, Math.floor(gx)); const wx = w(gx - i0);
      let v = 0;
      for (let dj = -1; dj <= 2; dj++) {
        let r = 0;
        for (let di = -1; di <= 2; di++) r += wx[di + 1] * at(i0 + di, j0 + dj);
        v += wy[dj + 1] * r;
      }
      const a = at(i0, j0), b = at(i0 + 1, j0), c = at(i0, j0 + 1), d = at(i0 + 1, j0 + 1);
      out[oy * NX + ox] = Math.max(Math.min(a, b, c, d), Math.min(Math.max(a, b, c, d), v));
    }
  }
  return { g: out, NX: NX, NY: NY };
}

/* marching squares -> flat [x1,y1,x2,y2,...] in panel pixels */
function contourSegs(g, NX, NY, lv, W, H) {
  const out = [], sx = W / (NX - 1), sy = H / (NY - 1);
  const P = (x0, y0, v0, x1, y1, v1) => { const t = (lv - v0) / (v1 - v0);
    return [(x0 + (x1 - x0) * t) * sx, H - (y0 + (y1 - y0) * t) * sy]; };
  for (let j = 0; j < NY - 1; j++) for (let i = 0; i < NX - 1; i++) {
    const v00 = g[j*NX+i], v10 = g[j*NX+i+1], v11 = g[(j+1)*NX+i+1], v01 = g[(j+1)*NX+i];
    const idx = (v00 > lv ? 1 : 0) | (v10 > lv ? 2 : 0) | (v11 > lv ? 4 : 0) | (v01 > lv ? 8 : 0);
    if (idx === 0 || idx === 15) continue;
    const eB = () => P(i, j, v00, i+1, j, v10), eR = () => P(i+1, j, v10, i+1, j+1, v11);
    const eT = () => P(i, j+1, v01, i+1, j+1, v11), eL = () => P(i, j, v00, i, j+1, v01);
    const seg = (a, b) => out.push(a[0], a[1], b[0], b[1]);
    switch (idx) {
      case 1: case 14: seg(eL(), eB()); break;
      case 2: case 13: seg(eB(), eR()); break;
      case 3: case 12: seg(eL(), eR()); break;
      case 4: case 11: seg(eR(), eT()); break;
      case 5: seg(eL(), eB()); seg(eR(), eT()); break;
      case 6: case 9: seg(eB(), eT()); break;
      case 7: case 8: seg(eL(), eT()); break;
      case 10: seg(eL(), eT()); seg(eB(), eR()); break;
    }
  }
  return out;
}

function drawIsolines(ctx, f, step, labelEvery, W, H, placed) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < NCELL; i++) { if (f[i] < mn) mn = f[i]; if (f[i] > mx) mx = f[i]; }
  const up = upsampleCR(f, 6);
  const levels = [];
  for (let lv = Math.ceil(mn / step) * step; lv <= mx; lv += step) levels.push(lv);
  const all = [];
  for (const lv of levels) all.push([lv, contourSegs(up.g, up.NX, up.NY, lv, W, H)]);
  ctx.lineCap = "round";
  for (const pass of [0, 1]) {
    ctx.beginPath();
    for (const [, s] of all) for (let i = 0; i < s.length; i += 4) { ctx.moveTo(s[i], s[i+1]); ctx.lineTo(s[i+2], s[i+3]); }
    ctx.strokeStyle = pass === 0 ? "rgba(10,20,30,0.28)" : "rgba(255,255,255,0.9)";
    ctx.lineWidth = pass === 0 ? 2.6 : 1;
    ctx.stroke();
  }
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const [lv, s] of all) {
    if (Math.round(lv) % labelEvery !== 0) continue;
    for (let i = 0; i < s.length; i += 4 * 7) {
      const x = (s[i] + s[i+2]) / 2, y = (s[i+1] + s[i+3]) / 2;
      if (x < 26 || x > W - 26 || y < 14 || y > H - 14) continue;
      let ok = true;
      for (const p of placed) if ((p[0]-x)*(p[0]-x) + (p[1]-y)*(p[1]-y) < 110*110) { ok = false; break; }
      if (!ok) continue;
      placed.push([x, y]);
      let ang = Math.atan2(s[i+3] - s[i+1], s[i+2] - s[i]);
      if (ang > Math.PI / 2) ang -= Math.PI; else if (ang < -Math.PI / 2) ang += Math.PI;
      ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
      haloText(ctx, String(Math.round(lv)), 0, 0, "600 10px " + FONT, "#111", "rgba(255,255,255,0.8)");
      ctx.restore();
    }
  }
}

function selectExtrema(f, mode) {
  const cand = [];
  for (let i = 0; i < NCELL; i++) if (MASK[i] && isFinite(f[i])) cand.push(i);
  cand.sort((a, b) => mode === "max" ? f[b] - f[a] : f[a] - f[b]);
  const sel = [];
  for (const idx of cand) {
    const la = B.min_lat + Math.floor(idx / NLON) * DLAT / (NLAT - 1);
    const lo = B.min_lon + (idx % NLON) * DLON / (NLON - 1);
    if (sel.every(s => Math.abs(la - s[0]) >= DATA.hl.minSep || Math.abs(lo - s[1]) >= DATA.hl.minSep)) sel.push([la, lo, f[idx]]);
    if (sel.length >= DATA.hl.count) break;
  }
  return sel;
}

/* ── WebGL shader ─────────────────────────────────────────────────── */
const VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uData;
uniform sampler2D uLut;
uniform ivec2 uDim;
uniform vec2 uRange;
uniform float uPow;
uniform float uOpacity;
in vec2 vUv;
out vec4 outColor;
float at(int i, int j) {
  return texelFetch(uData, ivec2(clamp(i, 0, uDim.x - 1), clamp(j, 0, uDim.y - 1)), 0).r;
}
vec4 cr(float t) {
  float t2 = t * t, t3 = t2 * t;
  return vec4(-0.5*t3 + t2 - 0.5*t, 1.5*t3 - 2.5*t2 + 1.0, -1.5*t3 + 2.0*t2 + 0.5*t, 0.5*t3 - 0.5*t2);
}
float row(int j, int i0, vec4 wx) {
  return wx.x*at(i0-1, j) + wx.y*at(i0, j) + wx.z*at(i0+1, j) + wx.w*at(i0+2, j);
}
void main() {
  vec2 g = vec2(vUv.x * float(uDim.x - 1), vUv.y * float(uDim.y - 1));
  ivec2 c = ivec2(floor(g));
  c = clamp(c, ivec2(0), uDim - 2);
  vec2 f = g - vec2(c);
  vec4 wx = cr(f.x), wy = cr(f.y);
  float v = wy.x*row(c.y-1, c.x, wx) + wy.y*row(c.y, c.x, wx) + wy.z*row(c.y+1, c.x, wx) + wy.w*row(c.y+2, c.x, wx);
  float a = at(c.x, c.y), b = at(c.x+1, c.y), d = at(c.x, c.y+1), e = at(c.x+1, c.y+1);
  v = clamp(v, min(min(a, b), min(d, e)), max(max(a, b), max(d, e)));
  float t = pow(clamp((v - uRange.x) / (uRange.y - uRange.x), 0.0, 1.0), uPow);
  vec4 col = texture(uLut, vec2(mix(0.5 / 2048.0, 1.0 - 0.5 / 2048.0, t), 0.5));
  float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  col.rgb += (n - 0.5) / 255.0 * col.a;
  outColor = col * uOpacity;
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

/* ── Panel ────────────────────────────────────────────────────────── */
class Panel {
  constructor(idx, el) {
    this.idx = idx; this.el = el; this.box = el.querySelector(".mapbox");
    this.cBase = el.querySelector(".c-base"); this.cGL = el.querySelector(".c-gl");
    this.cFlow = el.querySelector(".c-flow"); this.cOver = el.querySelector(".c-over");
    this.ctxB = this.cBase.getContext("2d"); this.ctxF = this.cFlow.getContext("2d"); this.ctxO = this.cOver.getContext("2d");
    this.W = 0; this.H = 0; this.dpr = 1; this.model = MODELS[idx];
    this.field = new Float32Array(NCELL); this.fu = new Float32Array(NCELL); this.fv = new Float32Array(NCELL);
    this.flowReady = false; this.missing = false; this.frame = 0; this.lutVar = null;
    this.initGL();
  }
  initGL() {
    const gl = this.cGL.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
    this.gl = null;
    if (!gl) return;
    try {
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
      gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      this.u = {};
      ["uData","uLut","uDim","uRange","uPow","uOpacity"].forEach(n => this.u[n] = gl.getUniformLocation(prog, n));
      this.vao = gl.createVertexArray();
      this.texData = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texData);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, NLON, NLAT, 0, gl.RED, gl.FLOAT, null);
      this.texLut = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texLut);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.prog = prog; this.gl = gl;
    } catch (e) { console.error("WebGL2 init failed:", e); this.gl = null; }
  }
  resize() {
    const cw = this.el.clientWidth, ch = this.el.clientHeight;
    if (cw < 20 || ch < 20) return false;
    let w, h;
    if (cw / ch > ASPECT) { h = ch; w = ch * ASPECT; } else { w = cw; h = cw / ASPECT; }
    w = Math.floor(w); h = Math.floor(h);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === this.W && h === this.H && dpr === this.dpr) return false;
    this.W = w; this.H = h; this.dpr = dpr;
    this.box.style.width = w + "px"; this.box.style.height = h + "px";
    for (const c of [this.cBase, this.cGL, this.cOver]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    this.cFlow.width = w; this.cFlow.height = h;
    this.drawBase(); this.initParticles(); this.lutVar = null;
    return true;
  }
  X(lon) { return (lon - B.min_lon) / DLON * this.W; }
  Y(lat) { return (B.max_lat - lat) / DLAT * this.H; }

  drawBase() {
    const ctx = this.ctxB, W = this.W, H = this.H, X = lon => this.X(lon), Y = lat => this.Y(lat);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const hasLand = BASE.land && BASE.land.length;
    ctx.fillStyle = hasLand ? "#a9bccb" : "#d9d6cd"; ctx.fillRect(0, 0, W, H);
    if (hasLand) {
      ctx.beginPath(); tracePath(ctx, BASE.land, X, Y);
      ctx.fillStyle = "#d9d6cd"; ctx.fill("evenodd");
      ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(70,90,110,0.55)"; ctx.stroke();
    }
    if (BASE.states && BASE.states.length) {
      ctx.beginPath(); tracePath(ctx, BASE.states, X, Y);
      ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(90,98,108,0.6)"; ctx.stroke();
    }
  }

  clearGL() {
    const gl = this.gl; if (!gl) return;
    gl.viewport(0, 0, this.cGL.width, this.cGL.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  }
  drawGL(v, opacity) {
    const gl = this.gl; if (!gl || !this.W) return;
    const lut = buildLut(v);
    gl.viewport(0, 0, this.cGL.width, this.cGL.height);
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texData);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NLON, NLAT, gl.RED, gl.FLOAT, this.field);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLut);
    if (this.lutVar !== v) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, LUT_N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut.u8);
      this.lutVar = v;
    }
    gl.uniform1i(this.u.uData, 0); gl.uniform1i(this.u.uLut, 1);
    gl.uniform2i(this.u.uDim, NLON, NLAT);
    gl.uniform2f(this.u.uRange, lut.vmin, lut.vmax);
    gl.uniform1f(this.u.uPow, lut.pow); gl.uniform1f(this.u.uOpacity, opacity);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /* ---- wind particles ---- */
  initParticles() {
    const n = Math.min(2600, Math.round(this.W * this.H / 650));
    this.pn = n; this.px = new Float32Array(n); this.py = new Float32Array(n); this.pa = new Int16Array(n);
    for (let i = 0; i < n; i++) this.respawn(i, true);
    this.ctxF.clearRect(0, 0, this.W, this.H);
  }
  respawn(i, init) {
    this.px[i] = Math.random() * this.W; this.py[i] = Math.random() * this.H;
    this.pa[i] = 40 + Math.floor(Math.random() * 90) + (init ? Math.floor(Math.random() * 60) : 0);
  }
  clearFlow() { this.ctxF.clearRect(0, 0, this.W, this.H); }
  stepFlow() {
    if (!this.flowReady || !this.W) return;
    const ctx = this.ctxF, W = this.W, H = this.H, u = this.fu, v = this.fv;
    this.frame++;
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0,0,0," + (this.frame % 20 === 0 ? 0.3 : 0.05) + ")";
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255,255,255,0.88)"; ctx.lineWidth = 1.1; ctx.lineCap = "round";
    ctx.beginPath();
    const k = 0.12 * W / 900, sx = (NLON - 1) / W, sy = (NLAT - 1) / H;
    for (let i = 0; i < this.pn; i++) {
      const x = this.px[i], y = this.py[i];
      if (--this.pa[i] <= 0) { this.respawn(i, false); continue; }
      const gx = x * sx, gy = (H - y) * sy;
      let i0 = Math.max(0, Math.min(NLON - 2, gx | 0)), j0 = Math.max(0, Math.min(NLAT - 2, gy | 0));
      const fx = gx - i0, fy = gy - j0, a = j0 * NLON + i0, b = a + 1, c = a + NLON, d = c + 1;
      const uu = (u[a] * (1 - fx) + u[b] * fx) * (1 - fy) + (u[c] * (1 - fx) + u[d] * fx) * fy;
      const vv = (v[a] * (1 - fx) + v[b] * fx) * (1 - fy) + (v[c] * (1 - fx) + v[d] * fx) * fy;
      const nx = x + uu * k, ny = y - vv * k;
      if (nx < 0 || nx > W || ny < 0 || ny > H) { this.respawn(i, false); continue; }
      ctx.moveTo(x, y); ctx.lineTo(nx, ny);
      this.px[i] = nx; this.py[i] = ny;
    }
    ctx.stroke();
  }

  /* ---- overlay ---- */
  drawOverlay(v, t) {
    const ctx = this.ctxO, W = this.W, H = this.H; if (!W) return;
    const X = lon => this.X(lon), Y = lat => this.Y(lat), meta = META[v], model = this.model;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    if (opts.outside !== "show") {
      ctx.beginPath(); ctx.rect(0, 0, W, H); tracePath(ctx, BASE.texas, X, Y);
      ctx.fillStyle = opts.outside === "dim" ? "rgba(12,18,26,0.42)" : "rgba(12,18,26,0.92)";
      ctx.fill("evenodd");
    }

    const placed = [];
    if (opts.iso && !isMissing(model, "pressure_msl")) drawIsolines(ctx, getField(model, "pressure_msl", t), 2, 4, W, H, placed);
    if (opts.isot && !isMissing(model, "temperature_2m")) drawIsolines(ctx, getField(model, "temperature_2m", t), 5, 10, W, H, placed);

    ctx.beginPath(); tracePath(ctx, BASE.texas, X, Y);
    ctx.lineJoin = "round"; ctx.lineWidth = 1.1; ctx.strokeStyle = "rgba(24,32,44,0.9)"; ctx.stroke();

    /* Wind-farm locations: intentionally shown only on the wind-speed map.
       The source is the same USWTDB fleet used for the MW conversion.
       Farms live under DATA.windForecast.windFarms (not top-level DATA.windFarms). */
    const farms = (DATA.windForecast && DATA.windForecast.windFarms) || [];
    if (opts.windFarms && v === "wind_speed_80m" && farms.length) {
      ctx.save();
      ctx.fillStyle = "rgba(100,105,110,0.95)";   /* light grey */
      ctx.strokeStyle = "rgba(50,55,62,0.55)";
      ctx.lineWidth = 0.65;
      for (const f of farms) {
        const x = X(f.lon), y = Y(f.lat);
        if (x < -3 || x > W + 3 || y < -3 || y > H + 3) continue;
        ctx.beginPath(); ctx.arc(x, y, 2.15, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.restore();
    }

    if (this.missing) {
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      haloText(ctx, MODEL_LABEL(model) + " does not provide", W / 2, H / 2 - 9, "600 13px " + FONT, "#111");
      haloText(ctx, meta.label, W / 2, H / 2 + 9, "600 13px " + FONT, "#111");
    } else {
      const dec = meta.decimals, f = this.field;
      if (opts.cities) {
        ctx.textAlign = "center";
        for (const c of DATA.cities) {
          const name = c[0], lat = c[1], lon = c[2], tx = X(lon + c[3]), ty = Y(lat + c[4]);
          const val = sample(f, lon, lat);
          ctx.beginPath(); ctx.arc(X(lon), Y(lat), 3, 0, 6.2832);
          ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 0.9; ctx.strokeStyle = "#111"; ctx.stroke();
          ctx.textBaseline = "bottom"; haloText(ctx, name, tx, ty - 6, "500 10px " + FONT, "#111");
          ctx.textBaseline = "top"; haloText(ctx, fmtV(val, dec), tx, ty + 5, "700 11px " + FONT, "#111");
        }
        for (const c of DATA.valueOnly) {
          const val = sample(f, c[2], c[1]);
          ctx.beginPath(); ctx.arc(X(c[2]), Y(c[1]), 2.6, 0, 6.2832);
          ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 0.8; ctx.strokeStyle = "#111"; ctx.stroke();
          ctx.textBaseline = "top"; haloText(ctx, fmtV(val, dec), X(c[2]), Y(c[1]) + 5, "700 10.5px " + FONT, "#111");
        }
      }
      if (opts.hl && v !== "precipitation_probability") {
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const isP = v === "precipitation" || v === "precip_3h";
        for (const mode of ["max", "min"]) {
          if (isP && mode === "min") continue;
          for (const e of selectExtrema(f, mode)) {
            if (mode === "max" && (isP ? e[2] < 0.1 : (v === "shortwave_radiation" && e[2] <= 0))) continue;
            if (mode === "min" && v === "shortwave_radiation" && e[2] <= 0) continue;
            haloText(ctx, (mode === "max" ? "H " : "L ") + fmtV(e[2], dec), X(e[1]), Y(e[0]),
                     "800 13px " + FONT, mode === "max" ? "#8a1010" : "#0d3a8a");
          }
        }
      }
    }
    /* model chip */
    const label = MODEL_LABEL(model);
    ctx.font = "700 13px " + FONT; const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(255,255,255,0.88)"; ctx.beginPath();
    ctx.moveTo(8, 8); ctx.arcTo(8 + tw + 16, 8, 8 + tw + 16, 30, 5); ctx.arcTo(8 + tw + 16, 30, 8, 30, 5);
    ctx.arcTo(8, 30, 8, 8, 5); ctx.arcTo(8, 8, 8 + tw + 16, 8, 5); ctx.closePath(); ctx.fill();
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = "#111"; ctx.fillText(label, 16, 19.5);
  }

  render(model, v, t, overlay) {
    this.model = model;
    this.missing = isMissing(model, v);
    if (!this.missing) { getField(model, v, t, this.field); this.drawGL(v, opts.opacity); } else this.clearGL();
    if (opts.flow && !isMissing(model, "wind_u") && !isMissing(model, "wind_v")) {
      getField(model, "wind_u", t, this.fu); getField(model, "wind_v", t, this.fv); this.flowReady = true;
    } else if (this.flowReady) { this.flowReady = false; this.clearFlow(); }
    if (overlay) this.drawOverlay(v, t);
  }
}
function MODEL_LABEL(m) { return (DATA.modelLabels && DATA.modelLabels[m]) || m; }

const panels = [0,1,2,3].map(i => new Panel(i, $("panel" + i)));
if (!panels[0].gl) { fatal("This viewer needs WebGL2, which your browser or GPU settings have disabled."); return; }

/* ── legend ───────────────────────────────────────────────────────── */
function drawLegend() {
  const meta = META[currentVar], stops = meta.stops, n = stops.length;
  legendLabel.textContent = meta.label + " (" + meta.unit + ")";
  const dpr = Math.min(window.devicePixelRatio || 1, 2), cw = legendCv.clientWidth, ch = legendCv.clientHeight;
  if (!cw) return;
  legendCv.width = Math.round(cw * dpr); legendCv.height = Math.round(ch * dpr);
  const ctx = legendCv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const x0 = 10, x1 = cw - 10, y0 = 2, bh = 13;
  ctx.fillStyle = "#d9d6cd"; ctx.fillRect(x0, y0, x1 - x0, bh);
  const g = ctx.createLinearGradient(x0, 0, x1, 0);
  stops.forEach((s, i) => g.addColorStop(i / (n - 1), cssColor(s[1])));
  ctx.fillStyle = g; ctx.fillRect(x0, y0, x1 - x0, bh);
  ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, bh - 1);
  const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((x1 - x0) / 46))));
  ctx.font = "11px " + FONT; ctx.fillStyle = "#93a0ae"; ctx.textBaseline = "top";
  stops.forEach((s, i) => {
    if (i % every !== 0 && i !== n - 1) return;
    const x = x0 + (x1 - x0) * i / (n - 1);
    ctx.textAlign = i === 0 ? "left" : i === n - 1 ? "right" : "center";
    ctx.fillText(fmtStop(s[0]), x, y0 + bh + 4);
  });
}


/* ── wind MW chart ───────────────────────────────────────────────── */
const WIND_COLORS = ["#5ba7ff", "#f2a93b", "#7bd389", "#c084fc", "#f26b6b"];
const windChartState = { visible: {}, hover: -1, dpr: 1 };

function windSeriesNames() {
  return DATA.windForecast && DATA.windForecast.series
    ? Object.keys(DATA.windForecast.series) : [];
}

function windChartX(t, n, left, width) {
  return n <= 1 ? left : left + t / (n - 1) * width;
}

function drawWindChart() {
  const wf = DATA.windForecast;
  if (!wf || !wf.times || !wf.times.length || !wf.series) {
    windDock.classList.remove("on");
    return;
  }
  const cw = windChartWrap.clientWidth, ch = windChartWrap.clientHeight;
  if (cw < 40 || ch < 40) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  windChartState.dpr = dpr;
  windChart.width = Math.round(cw * dpr); windChart.height = Math.round(ch * dpr);
  const ctx = windChart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const left = 42, right = 10, top = 8, bottom = 22;
  const w = Math.max(20, cw - left - right), h = Math.max(20, ch - top - bottom);
  const names = windSeriesNames().filter(n => windChartState.visible[n] !== false);
  const allVals = [];
  for (const n of names) for (const v of (wf.series[n] || [])) if (v != null && Number.isFinite(v)) allVals.push(v);
  const cap = Number(wf.fleetCapacityMw) || 0;
  /* Y-axis: 0 → (highest plotted value + 2000 MW). Capacity is drawn as a
     reference line only and no longer forces the scale up. */
  const dataMax = allVals.length ? Math.max(...allVals) : 0;
  const ymax = Math.max(dataMax + 2000, 100);
  const y = v => top + h - (v / ymax) * h;

  ctx.font = "9px " + FONT; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.strokeStyle = "rgba(147,160,174,.16)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = top + h * i / 4;
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + w, yy); ctx.stroke();
    const val = ymax * (1 - i / 4);
    ctx.fillStyle = "#93a0ae"; ctx.fillText(Math.round(val).toLocaleString(), left - 5, yy);
  }

  if (cap > 0) {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(220,225,230,.32)";
    ctx.beginPath(); ctx.moveTo(left, y(cap)); ctx.lineTo(left + w, y(cap)); ctx.stroke();
    ctx.setLineDash([]);
  }

  const times = wf.times, n = times.length;
  ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = "#93a0ae";
  const tickCount = Math.min(8, Math.max(2, Math.floor(w / 90)));
  for (let i = 0; i < tickCount; i++) {
    const k = Math.round(i * (n - 1) / (tickCount - 1));
    const xx = windChartX(k, n, left, w);
    /* HE convention: timestamp hour H (0–23) → HE H+1 (HE 1 … HE 24). e.g. 20:00 / 8pm → HE 21 */
    const hour = parseInt(times[k].slice(11, 13), 10);
    const he = hour + 1;  /* 0→1, 20→21, 23→24 */
    const d = new Date(times[k].slice(0, 10) + "T12:00:00");
    const dayLabel = d.toLocaleDateString(undefined, {month: "short", day: "numeric"});
    ctx.fillText(dayLabel + " HE " + he, xx, top + h + 5);
  }

  windSeriesNames().forEach((name, si) => {
    if (windChartState.visible[name] === false) return;
    const vals = wf.series[name] || [];
    ctx.strokeStyle = WIND_COLORS[si % WIND_COLORS.length];
    ctx.lineWidth = name === "ERCOT STWPF" ? 2.2 : 1.8;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      if (v == null || !Number.isFinite(v)) { started = false; continue; }
      const xx = windChartX(i, n, left, w), yy = y(v);
      if (!started) { ctx.moveTo(xx, yy); started = true; } else ctx.lineTo(xx, yy);
    }
    ctx.stroke();
  });

  /* Vertical marker: maps' currently selected hour */
  (function drawMapHourMarker() {
    const mapT = (typeof curT === "function") ? curT() : Math.round(tPos);
    const mapKey = (DATA.times && DATA.times[mapT]) || null;
    if (!mapKey) return;
    let mi = times.indexOf(mapKey);
    if (mi < 0) {
      const prefix = mapKey.slice(0, 13);
      mi = times.findIndex(s => s.slice(0, 13) === prefix);
    }
    if (mi < 0) return;
    const xx = windChartX(mi, n, left, w);
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = "rgba(160,170,180,0.85)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xx, top);
    ctx.lineTo(xx, top + h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  })();

  if (windChartState.hover >= 0 && windChartState.hover < n) {
    const i = windChartState.hover, xx = windChartX(i, n, left, w);
    ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, top + h); ctx.stroke();
    windSeriesNames().forEach((name, si) => {
      if (windChartState.visible[name] === false) return;
      const v = (wf.series[name] || [])[i];
      if (v == null || !Number.isFinite(v)) return;
      ctx.fillStyle = WIND_COLORS[si % WIND_COLORS.length];
      ctx.beginPath(); ctx.arc(xx, y(v), 3, 0, 6.2832); ctx.fill();
    });
  }
}

function updateWindLegend() {
  windLegend.innerHTML = "";
  if (!DATA.windForecast) return;
  windSeriesNames().forEach((name, i) => {
    if (!(name in windChartState.visible)) windChartState.visible[name] = true;
    const b = document.createElement("button");
    b.type = "button"; b.className = "sw" + (windChartState.visible[name] ? "" : " off");
    const dot = document.createElement("span");
    dot.className = "dot"; dot.style.background = WIND_COLORS[i % WIND_COLORS.length];
    b.appendChild(dot); b.appendChild(document.createTextNode(name));
    b.onclick = () => { windChartState.visible[name] = !windChartState.visible[name]; updateWindLegend(); drawWindChart(); };
    windLegend.appendChild(b);
  });
}

function updateWindDock() {
  const on = !!chkWindMW.checked;
  windDock.classList.toggle("on", on);
  if (!on) return;
  const wf = DATA.windForecast;
  if (!wf) {
    windMeta.textContent = "Unavailable";
    windNote.textContent = "Wind-fleet or ERCOT STWPF data could not be loaded.";
    return;
  }
  windMeta.textContent = `${wf.fleetFarmCount.toLocaleString()} farms • ${wf.fleetCapacityMw.toLocaleString()} MW`;
  windNote.textContent = wf.note + (wf.ercotAvailable ? "" : " ERCOT STWPF was unavailable for this run.");
  updateWindLegend();
  requestAnimationFrame(drawWindChart);
}

windChart.addEventListener("mousemove", e => {
  const wf = DATA.windForecast; if (!wf || !wf.times.length || !windDock.classList.contains("on")) return;
  const r = windChart.getBoundingClientRect(), x = e.clientX - r.left;
  const left = 42, right = 10, w = Math.max(20, r.width - left - right);
  const i = Math.max(0, Math.min(wf.times.length - 1, Math.round((x - left) / w * (wf.times.length - 1))));
  windChartState.hover = i; drawWindChart();
  /* HE convention to match map timeline and ERCOT HE labels */
  const hour = parseInt(wf.times[i].slice(11, 13), 10);
  const he = hour + 1;
  const d = new Date(wf.times[i].slice(0, 10) + "T12:00:00");
  let html = "<b>" + d.toLocaleDateString(undefined,{weekday:"short",month:"short",day:"numeric"}) +
             "  •  HE " + he + " CT</b>";
  for (const [si,name] of windSeriesNames().entries()) {
    if (windChartState.visible[name] === false) continue;
    const v = (wf.series[name] || [])[i];
    html += "<br><span style='color:" + WIND_COLORS[si % WIND_COLORS.length] + "'>●</span> " +
            name + ": " + (v == null ? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:0}) + " MW");
  }
  windChartTip.innerHTML = html; windChartTip.style.display = "block";
  windChartTip.style.left = Math.min(Math.max(6, x + 10), r.width - windChartTip.offsetWidth - 6) + "px";
  windChartTip.style.top = "6px";
});
windChart.addEventListener("mouseleave", () => { windChartState.hover = -1; windChartTip.style.display = "none"; drawWindChart(); });
new ResizeObserver(() => { if (windDock.classList.contains("on")) drawWindChart(); }).observe(windChartWrap);

function setWindExpanded(on) {
  windDock.classList.toggle("expanded", !!on);
  windExpandBtn.textContent = on ? "Collapse" : "Expand";
  windExpandBtn.title = on ? "Exit full screen (Esc)" : "Expand chart (Esc to exit)";
  requestAnimationFrame(() => { drawWindChart(); });
}
windExpandBtn.addEventListener("click", () => {
  if (!windDock.classList.contains("on")) return;
  setWindExpanded(!windDock.classList.contains("expanded"));
});
/* Collapse fullscreen when the dock is hidden */
const _updateWindDockOrig = updateWindDock;
updateWindDock = function () {
  _updateWindDockOrig();
  if (!windDock.classList.contains("on") && windDock.classList.contains("expanded")) {
    setWindExpanded(false);
  }
};



/* ── render loop / time ───────────────────────────────────────────── */
function curT() { return Math.max(0, Math.min(T - 1, Math.round(tPos))); }

function syncUI() {
  const t = curT();
  hourSlider.max = Math.max(0, T - 1);
  hourSlider.value = t;
  const d = DATE_OF[t];
  const dt = new Date(d + "T12:00:00").toLocaleDateString(undefined, {
    weekday: "short", month: "short", day: "numeric"
  });
  timeLabel.textContent = dt + "  •  HE " + (HOUR_OF[t] + 1) + " CT";
}

function renderAll(forceOverlay) {
  const doOverlay = forceOverlay || Math.abs(tPos - lastOverlayT) >= 0.25;
  if (doOverlay) lastOverlayT = tPos;
  for (let i = 0; i < layoutCount; i++) panels[i].render(panelModels[i], currentVar, tPos, doOverlay);
}

function frame(ts) {
  rafId = null;
  const dt = Math.min(0.1, (ts - lastFrame) / 1000); lastFrame = ts;
  if (playing) {
    tPos += dt * hoursPerSec; if (tPos > T - 1) tPos = 0;
    syncUI(); renderAll(false);
    if (windDock.classList.contains("on")) drawWindChart();
  }
  for (let i = 0; i < layoutCount; i++) panels[i].stepFlow();
  if (playing || opts.flow) rafId = requestAnimationFrame(frame);
}
function ensureLoop() {
  if (!rafId && (playing || opts.flow)) { lastFrame = performance.now(); rafId = requestAnimationFrame(frame); }
}
function refresh() {
  syncUI(); renderAll(true); ensureLoop(); hideProbe();
  if (windDock.classList.contains("on")) drawWindChart();
}

/* ── UI wiring ────────────────────────────────────────────────────── */
VARS.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = META[v].label; varSelect.appendChild(o); });
modelSelects.forEach((sel, idx) => {
  MODELS.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = MODEL_LABEL(m); sel.appendChild(o); });
  sel.value = MODELS[idx];
  sel.onchange = () => {
    const chosen = sel.value, prev = panelModels[idx];
    for (let j = 0; j < layoutCount; j++) if (j !== idx && panelModels[j] === chosen) { panelModels[j] = prev; modelSelects[j].value = prev; break; }
    panelModels[idx] = chosen; refresh();
  };
});

function resizeAll() { let changed = false; for (let i = 0; i < layoutCount; i++) if (panels[i].resize()) changed = true; return changed; }
const PRESET_3 = (function () {
  const out = [];
  [/hrrr/i, /gfs/i, /ecmwf|\bifs\b/i].forEach(re => {
    const m = MODELS.find(x => !out.includes(x) && (re.test(x) || re.test(MODEL_LABEL(x))));
    if (m) out.push(m);
  });
  MODELS.forEach(m => { if (out.length < 3 && !out.includes(m)) out.push(m); });
  return out;
})();

function applyLayout(n) {
  const prevCount = layoutCount;
  layoutCount = n; mapsEl.className = "layout-" + n;
  const start = (n === 3 && prevCount !== 3)
    ? PRESET_3.concat(MODELS.filter(m => !PRESET_3.includes(m)))
    : panelModels;
  const used = new Set(), next = [];
  for (let i = 0; i < n; i++) {
    let m = start[i];
    if (!m || used.has(m)) m = MODELS.find(x => !used.has(x));
    used.add(m); next.push(m);
  }
  panelModels = n === 4 ? MODELS.slice() : next.concat(MODELS.filter(m => !used.has(m)));
  panels.forEach((p, i) => p.el.classList.toggle("hidden", i >= n));
  modelWraps.forEach((w, i) => w.classList.toggle("hidden", i >= n));
  modelSelects.forEach((sel, i) => { sel.value = panelModels[i]; sel.disabled = (n === 4); });
  resizeAll(); refresh();
}

varSelect.onchange = e => {
  currentVar = e.target.value;
  chkWindFarms.disabled = currentVar !== "wind_speed_80m";
  drawLegend(); refresh();
};
layoutSelect.onchange = e => applyLayout(+e.target.value);
hourSlider.oninput = e => { tPos = +e.target.value; refresh(); };
speedSelect.onchange = e => { hoursPerSec = +e.target.value; };
outsideSelect.onchange = e => { opts.outside = e.target.value; renderAll(true); };
opacityRange.oninput = e => { opts.opacity = +e.target.value / 100; renderAll(false); };
[["chkFlow","flow"],["chkIso","iso"],["chkIsot","isot"],["chkCities","cities"],["chkHL","hl"],["chkWindFarms","windFarms"]].forEach(([id, k]) => {
  $(id).onchange = e => {
    opts[k] = e.target.checked;
    if (k === "flow" && !opts.flow) panels.forEach(p => { p.flowReady = false; p.clearFlow(); });
    renderAll(true); ensureLoop();
  };
});
chkWindMW.onchange = () => {
  updateWindDock();
  requestAnimationFrame(() => { resizeAll(); renderAll(true); drawLegend(); drawWindChart(); });
};
playBtn.onclick = () => {
  playing = !playing; playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
  if (playing) hideProbe(); else { tPos = curT(); refresh(); }
  ensureLoop();
};
document.addEventListener("keydown", e => {
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && e.key !== "Escape") return;
  if (e.key === "Escape") {
    if (windDock.classList.contains("expanded")) { setWindExpanded(false); return; }
    hideProbe();
  }
  else if (e.key === " ") { e.preventDefault(); playBtn.click(); }
  else if (e.key === "ArrowRight") { tPos = Math.min(T - 1, curT() + 1); refresh(); }
  else if (e.key === "ArrowLeft") { tPos = Math.max(0, curT() - 1); refresh(); }
});

/* ── probe ────────────────────────────────────────────────────────── */
function hideProbe() { probeEl.style.display = "none"; }
function showProbe(pi, e) {
  const p = panels[pi], r = p.box.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (x < 0 || y < 0 || x > r.width || y > r.height) { hideProbe(); return; }
  const lon = B.min_lon + x / r.width * DLON, lat = B.max_lat - y / r.height * DLAT;
  const model = panelModels[pi], t = curT();
  let rows = "";
  for (const v of VARS) {
    const meta = META[v], val = sampleQ(model, v, t, lon, lat);
    rows += '<tr><td class="k">' + meta.label + '</td><td class="v">' +
            (val == null ? "n/a" : fmtV(val, meta.decimals) + " " + meta.unit) + "</td></tr>";
  }
  const u = sampleQ(model, "wind_u", t, lon, lat), vv = sampleQ(model, "wind_v", t, lon, lat);
  if (u != null && vv != null) {
    const dir = (Math.atan2(-u, -vv) * 180 / Math.PI + 360) % 360;
    rows += '<tr><td class="k">Wind From</td><td class="v">' + compass(dir) + " (" + Math.round(dir) + "°)</td></tr>";
  }
  probeEl.innerHTML = '<div class="probe-title">' + MODEL_LABEL(model) + '</div>' +
    '<div class="probe-loc">' + lat.toFixed(2) + "°N, " + Math.abs(lon).toFixed(2) + "°W</div><table>" + rows + "</table>";
  probeEl.style.display = "block";
  const pad = 14, w = probeEl.offsetWidth || 230, h = probeEl.offsetHeight || 260;
  let left = e.clientX + pad, top = e.clientY + pad;
  if (left + w > innerWidth - 8) left = e.clientX - w - pad;
  if (top + h > innerHeight - 8) top = e.clientY - h - pad;
  probeEl.style.left = Math.max(8, left) + "px"; probeEl.style.top = Math.max(8, top) + "px";
}
panels.forEach((p, i) => p.box.addEventListener("click", e => showProbe(i, e)));
document.addEventListener("click", e => { if (!e.target.closest || !e.target.closest(".panel")) hideProbe(); }, true);

/* ── boot ─────────────────────────────────────────────────────────── */
$("subtitle").textContent = DATA.npoints + " grid points • " + MODELS.length + " models • rendered live on your GPU";
$("lastUpdated").textContent = "Last updated: " + DATA.generated_at + " CT";

(function pickStart() {   /* start at the current hour (Central Time) if it is in range */
  let t = -1;
  try {
    const s = new Date().toLocaleString("sv-SE", { timeZone: DATA.tz }).replace(" ", "T").slice(0, 13) + ":00";
    t = DATA.times.indexOf(s);
  } catch (e) {}
  if (t < 0) { const d0 = dayIdx[dates[0]]; t = d0[Math.min(12, d0.length - 1)]; }
  tPos = t;
})();
hourSlider.min = 0;
hourSlider.max = Math.max(0, T - 1);
hourSlider.value = curT();
/* Tick labels under slider: HE only, aligned to real hours across the full range */
(function buildHourTicks() {
  if (!hourTicks || T < 1) return;
  const count = Math.min(8, Math.max(2, T));
  const parts = [];
  for (let i = 0; i < count; i++) {
    const k = count === 1 ? 0 : Math.round(i * (T - 1) / (count - 1));
    const he = (HOUR_OF[k] != null ? HOUR_OF[k] : 0) + 1;
    parts.push("<span>HE " + he + "</span>");
  }
  hourTicks.innerHTML = parts.join("");
})();
varSelect.value = currentVar; layoutSelect.value = "3";
chkWindFarms.checked = true;
chkWindFarms.disabled = currentVar !== "wind_speed_80m";

/* make the controls and the state agree from the very first render */
outsideSelect.value = "dim";
opacityRange.value = 100;
opts.outside = outsideSelect.value;
opts.opacity = +opacityRange.value / 100;

new ResizeObserver(() => { if (resizeAll()) renderAll(true); drawLegend(); }).observe(mapsEl);
applyLayout(3); drawLegend();
updateWindDock();
requestAnimationFrame(() => { resizeAll(); renderAll(true); drawWindChart(); });  /* redraw once layout has settled */
window.__viewer = { panels: panels, opts: opts, setT: t => { tPos = t; refresh(); }, setVar: v => { currentVar = v; varSelect.value = v; drawLegend(); refresh(); } };
})();
</script>
</body>
</html>
"""


# ── main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Grid + base map…")
    points, lats, lons = generate_grid(TEXAS_BOUNDS, SPACING)
    print(f"  data grid {len(lats)}×{len(lons)} = {len(points)} points")
    base = load_basemap()
    texas_rings = [np.array(r, dtype=float).reshape(-1, 2) for r in base["texas"]]
    mask = compute_inside_mask(lats, lons, texas_rings, SPACING)
    print(f"  {int(mask.sum())} of {mask.size} grid cells fall inside Texas")

    if DEMO_MODE:
        print("DEMO MODE: generating synthetic data (no API calls)…")
        arrs, timestamps = make_demo_data(lats, lons, MODELS)
    else:
        per_point, timestamps = fetch_all(points, MODELS, FETCH_VARS, FORECAST_DAYS, BATCH_SIZE)
        arrs = build_arrays(lats, lons, MODELS, FETCH_VARS, per_point, timestamps)
        del per_point

    print("Deriving + cleaning fields…")
    add_derived(arrs, MODELS)
    missing = clean_arrays(arrs, MODELS, ENC_VARS)
    for m, vs in missing.items():
        if vs:
            print(f"  note: {MODEL_LABELS[m]} provides no data for: {', '.join(vs)}")

    print("Exporting grid data…")
    export_grid_data(out_dir, arrs, MODELS, ENC_VARS)

    wind_forecast = build_wind_forecast_payload(arrs, lats, lons, timestamps, MODELS)

    var_meta = {}
    for v in LAYERS:
        label, unit, decimals, stops, _q, pw = VAR_META[v]
        var_meta[v] = {"label": label, "unit": unit, "decimals": decimals,
                       "stops": [[s[0], s[1]] for s in stops], "pow": pw}

    payload = {
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
        "tz": TIMEZONE,
        "models": MODELS,
        "modelLabels": MODEL_LABELS,
        "variables": LAYERS,
        "dataVars": ENC_VARS,
        "varMeta": var_meta,
        "quant": {v: list(QUANT[v]) for v in ENC_VARS},
        "times": [ts.strftime("%Y-%m-%dT%H:%M") for ts in timestamps],
        "nlat": len(lats), "nlon": len(lons), "npoints": len(points),
        "bounds": TEXAS_BOUNDS,
        "mask": [int(x) for x in mask.ravel()],
        "missing": missing,
        "cities": [[n, la, lo, *CITY_LABEL_OFFSETS.get(n, (0.0, 0.0))] for n, la, lo in CITIES],
        "valueOnly": [[n, la, lo] for n, la, lo in VALUE_ONLY_CITIES],
        "base": base,
        "hl": {"count": HL_COUNT, "minSep": HL_MIN_SEP_DEG},
        "windForecast": wind_forecast,
    }
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.0f}s → {out_dir.resolve()}")
    print(f"Open {out_dir / 'index.html'}  (keep grid_data.js next to it)")


if __name__ == "__main__":
    main()