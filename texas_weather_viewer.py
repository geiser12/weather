#!/usr/bin/env python3
"""
Texas Multi-Model Weather Viewer – High Quality Image Edition (Optimized)
All settings are configured below – no command-line arguments needed.

Performance changes vs. the original:
  1. The upsampled lat/lon mesh and the Texas boundary mask are computed
     ONCE and reused for every image, instead of being recomputed from
     scratch (including a scipy.ndimage.zoom + Path.contains_points call)
     on every single one of the ~1,000+ images.
  2. Image rendering is parallelized across CPU cores with
     ProcessPoolExecutor, since each image is fully independent.
  3. Dropped `optimize=True` on PNG save (slow, not needed for a local
     viewer) and dropped `bbox_inches="tight"` (forces an extra expensive
     layout pass on every save) in favor of a fixed layout computed once.

Updates in this edition:
  - Highs/lows are computed only over grid cells that fall inside the Texas
    boundary mask (not the full rectangular bounding box).
  - Extra value-only stations: Fredericksburg, Killeen, Tyler, San Angelo
    (value only, no city name).
  - City labels near the state border are nudged inward so name + value stay
    readable and do not clip against the outline.
  - Precipitation: optional 3-hour block aggregation (sum) so values are
    larger and the Blues colormap shows meaningful contrast.
  - Sharper rendering: denser source grid (0.5°), higher DPI, much less
    Gaussian blur, and Tropical Tidbits–style discrete filled contours with
    thin black isopleths (USE_FILLED_CONTOURS=True).
"""

from __future__ import annotations
import io
import json
import math
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from PIL import Image
import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests

# ============================================================================
# SETTINGS – edit these
# ============================================================================

SPACING = 1          # grid spacing in degrees (0.5 = denser/sharper, 0.75/1.0 = faster)
FORECAST_DAYS = 2      # number of days to fetch
BATCH_SIZE = 100        # points per API request (lower = safer against rate limits)
OUTPUT_DIR = "tx_model_viewer"
DPI = 160              # image quality (150–180 is sharp without huge files)
SLEEP_BETWEEN_BATCHES = 1.8   # seconds between successful API calls
UPSAMPLE_FACTOR = 6           # spatial upsampling; lower + less blur = sharper edges
SMOOTH_SIGMA = 0.35           # light Gaussian only (0 = maximum sharpness; was 1.2)
MAX_WORKERS = 16            # None = os.cpu_count()
FETCH_DEADLINE_SECONDS = 600

# Per-batch retries for timeouts / rate limits / transient network errors.
FETCH_MAX_ATTEMPTS = 6
# Minimum fraction of grid points that must have data after fetch, else fail.
FETCH_MIN_COVERAGE = 0.90
# Precipitation: aggregate consecutive hours into blocks of this size (sum of
# mm amounts). 1 = keep hourly; 3 = 3-hour totals (recommended so light
# rain becomes visible on the Blues scale). Only affects precipitation.
PRECIP_AGG_HOURS = 3

# Tropical Tidbits–style discrete filled contours (True) vs continuous pcolormesh.
# Discrete levels + thin black contour lines look sharper and more like operational
# model guidance pages.
USE_FILLED_CONTOURS = True
CONTOUR_LINE_WIDTH = 0.35     # thin isopleths over the fill (0 to disable)

# ============================================================================

TEXAS_BOUNDS = {"min_lat": 25.7, "max_lat": 36.6, "min_lon": -106.7, "max_lon": -93.4}

MODELS = ["ecmwf_ifs", "gfs_hrrr", "ncep_nam_conus", "gfs_global"]
MODEL_LABELS = {
    "ecmwf_ifs": "ECMWF",
    "gfs_hrrr": "HRRR",
    "ncep_nam_conus": "NAM",
    "gfs_global": "GFS",
}

VARIABLES = [
    "temperature_2m", "apparent_temperature", "cloud_cover",
    "precipitation_probability", "precipitation", "wind_speed_10m",
    "relative_humidity_2m", "shortwave_radiation",
]

VAR_META = {
    "temperature_2m": {
        "label": "Temperature", "unit": "°F",
        "cmap": "RdYlBu_r", "vmin": None, "vmax": None,
    },
    "apparent_temperature": {
        "label": "Feels Like", "unit": "°F",
        "cmap": "RdYlBu_r", "vmin": None, "vmax": None,
    },
    "cloud_cover": {
        "label": "Cloud Cover", "unit": "%",
        "cmap": "gray_r", "vmin": 0, "vmax": 100,
    },
    "precipitation_probability": {
        "label": "Precip Probability", "unit": "%",
        "cmap": "Blues", "vmin": 0, "vmax": 100,
    },
    "precipitation": {
        "label": "Precipitation", "unit": "mm",
        "cmap": "Blues", "vmin": 0, "vmax": None,
    },
    "wind_speed_10m": {
        "label": "Wind Speed", "unit": "mph",
        "cmap": "wind_bg", "vmin": 0, "vmax": None,
    },
    "relative_humidity_2m": {
        "label": "Relative Humidity", "unit": "%",
        "cmap": "YlGnBu", "vmin": 0, "vmax": 100,
    },
    "shortwave_radiation": {
        "label": "Solar Radiation", "unit": "W/m²",
        "cmap": "YlOrBr", "vmin": 0, "vmax": None,
    },
}

# Full labels: name + value. Near-border cities get an inward nudge so text
# stays inside the state outline.
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

# Value-only markers (no city name drawn). Useful for denser coverage
# without cluttering the map with extra text.
VALUE_ONLY_CITIES = [
    ("Fredericksburg", 30.2752, -98.8720),
    ("Killeen", 31.1171, -97.7278),
    ("Tyler", 32.3513, -95.3011),
    ("San Angelo", 31.4638, -100.4370),
]

# Per-city label offset (degrees) to pull name/value away from the border.
# Applied only when drawing text; the sample point for the value stays at
# the true city coordinates.
CITY_LABEL_OFFSETS = {
    "El Paso": (0.15, 0.35),          # east + north (away from western border)
    "Brownsville": (0.0, 0.35),       # north (away from southern border)
    "Beaumont": (-0.25, 0.15),        # west + a bit north (away from eastern border)
    "Laredo": (0.25, 0.15),           # east (away from western border)
    "Corpus Christi": (0.0, 0.25),    # north (coast)
    "Amarillo": (0.0, -0.15),         # slight south if needed near panhandle
}

# Highs/lows: how many top / bottom points to mark on the map, and the
# minimum lat/lon separation (degrees) enforced between picks so all N
# markers don't cluster on the same local bump.
HL_COUNT = 3
HL_MIN_SEP_DEG = 0.9


# ── Helpers ───────────────────────────────────────────────────────────────

def generate_grid(bounds, spacing):
    # Use linspace (not arange) so the grid's last row/column always lands
    # exactly on max_lat / max_lon. arange(min, max, spacing) can fall up to
    # `spacing` degrees short of the true bound due to how the step count
    # rounds, which silently clips real coverage (e.g. the top of the
    # panhandle or the eastern border) even though the axis limits still
    # show the full bounding box.
    n_lat = int(np.ceil((bounds["max_lat"] - bounds["min_lat"]) / spacing)) + 1
    n_lon = int(np.ceil((bounds["max_lon"] - bounds["min_lon"]) / spacing)) + 1
    lats = np.linspace(bounds["min_lat"], bounds["max_lat"], n_lat)
    lons = np.linspace(bounds["min_lon"], bounds["max_lon"], n_lon)
    points = [(round(float(la), 4), round(float(lo), 4)) for la in lats for lo in lons]
    return points, lats, lons


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def fetch_texas_boundary():
    url = ("https://tigerweb.geo.census.gov/arcgis/rest/services/"
           "TIGERweb/State_County/MapServer/15/query")
    params = {"where": "STATE='48'", "outFields": "*", "returnGeometry": "true",
              "f": "geojson", "outSR": "4326"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()["features"][0]["geometry"]
    except Exception as e:
        print(f"[warn] Boundary download failed: {e}")
        return {
            "type": "Polygon",
            "coordinates": [[
                [TEXAS_BOUNDS["min_lon"], TEXAS_BOUNDS["min_lat"]],
                [TEXAS_BOUNDS["max_lon"], TEXAS_BOUNDS["min_lat"]],
                [TEXAS_BOUNDS["max_lon"], TEXAS_BOUNDS["max_lat"]],
                [TEXAS_BOUNDS["min_lon"], TEXAS_BOUNDS["max_lat"]],
                [TEXAS_BOUNDS["min_lon"], TEXAS_BOUNDS["min_lat"]],
            ]]
        }


def polygon_to_mpl_path(geometry):
    paths = []
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"]:
            paths.append(MplPath([(x, y) for x, y in ring]))
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                paths.append(MplPath([(x, y) for x, y in ring]))
    return paths


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = (
        "timeout", "timed out", "timeoutreached", "rate", "limit",
        "429", "503", "502", "504", "connection", "reset", "broken pipe",
        "temporarily", "unavailable", "stream",
    )
    return any(n in msg for n in needles)


def fetch_all(points, models, variables, forecast_days, batch_size):
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
            "timezone": "America/Chicago",
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

                # Backoff: timeouts/network shorter; rate-limit longer
                if "limit" in msg or "rate" in msg or "429" in msg:
                    wait = min(90 + attempt * 30, 180)
                else:
                    wait = min(5 * attempt, 45)  # timeout / connection

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

    # Coverage check: refuse to publish sparse grids after failed batches
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
            f"failed batches {failed_batches}. Not rendering incomplete maps."
        )
    return per_point, timestamps_ref


def build_values(points, lats, lons, models, variables, per_point, timestamps_ref):
    dates = sorted({ts.date().isoformat() for ts in timestamps_ref})
    nlat, nlon = len(lats), len(lons)
    values = {m: {v: {} for v in variables} for m in models}

    for model in models:
        for var in variables:
            series_list = per_point[model][var]
            filled = [s if s is not None else pd.Series(np.nan, index=timestamps_ref)
                      for s in series_list]
            df = pd.concat(filled, axis=1)
            for date in dates:
                day_index = [pd.Timestamp(date) + pd.Timedelta(hours=h) for h in range(24)]
                sub = df.reindex(day_index).to_numpy(dtype=float)
                fields = []
                for row in sub:
                    arr = row.reshape(nlat, nlon)
                    fields.append(arr)
                values[model][var][date] = fields
    return values, dates


def aggregate_precip_3h(values, models, dates, block=PRECIP_AGG_HOURS):
    """Replace hourly precipitation fields with summed N-hour blocks.

    For each day, hours 0..23 become block totals:
      hour h  →  sum of hours (h // block)*block  ..  (h // block)*block + block - 1
    (with the last partial block summing whatever remains).  This keeps the
    same 24 slots in the UI/slider while making light rain visible.
    """
    if block <= 1:
        return
    for model in models:
        for date in dates:
            hourly = values[model]["precipitation"][date]
            n = len(hourly)
            out = []
            for h in range(n):
                start = (h // block) * block
                end = min(start + block, n)
                stack = np.stack(hourly[start:end], axis=0)
                # nansum so missing hours don't wipe a whole block
                summed = np.nansum(stack, axis=0)
                # if every hour was NaN, keep NaN
                all_nan = np.all(~np.isfinite(stack), axis=0)
                summed = np.where(all_nan, np.nan, summed)
                out.append(summed)
            values[model]["precipitation"][date] = out


def compute_ranges(values, models, variables, dates):
    """Per-hour color scales, shared across all models.

    Fixed-scale variables (cloud, PoP, RH) keep their 0–100 meta limits.
    Auto-scaled variables (temp, feels-like, precip, wind, solar) use the
    min/max across *all models* for that specific date+hour only, so each
    hour's maps use the full colormap and show more spatial nuance instead
    of being crushed into a narrow band of a multi-day global range.
    """
    ranges = {v: {d: [None] * 24 for d in dates} for v in variables}
    for var in variables:
        meta = VAR_META[var]
        fixed_vmin = meta["vmin"]
        fixed_vmax = meta["vmax"]
        for date in dates:
            for h in range(24):
                if fixed_vmin is not None and fixed_vmax is not None:
                    ranges[var][date][h] = (float(fixed_vmin), float(fixed_vmax))
                    continue
                mn, mx = np.inf, -np.inf
                for model in models:
                    arr = values[model][var][date][h]
                    valid = arr[np.isfinite(arr)]
                    if len(valid):
                        mn = min(mn, float(valid.min()))
                        mx = max(mx, float(valid.max()))
                if not np.isfinite(mn):
                    mn, mx = 0.0, 1.0
                vmin = float(fixed_vmin) if fixed_vmin is not None else mn
                vmax = float(fixed_vmax) if fixed_vmax is not None else mx
                # Small pad so extrema aren't stuck on the colorbar edge
                if vmax > vmin:
                    pad = (vmax - vmin) * 0.04
                    if fixed_vmin is None:
                        vmin -= pad
                    if fixed_vmax is None:
                        vmax += pad
                if vmax <= vmin:
                    vmax = vmin + 1.0
                if var == "precipitation":
                    # Headroom + floor (mm) so light rain still uses color
                    vmax = max(vmax * 1.1, 4.0)
                    vmin = 0.0
                if var == "shortwave_radiation":
                    vmin = 0.0
                    vmax = max(vmax, 50.0)
                if var == "wind_speed_10m":
                    vmin = 0.0
                ranges[var][date][h] = (float(vmin), float(vmax))
    return ranges


# ── Precomputed geometry (built once, reused for every image) ─────────────

def precompute_geometry(lats, lons, texas_paths, factor=UPSAMPLE_FACTOR):
    """Compute the upsampled lat/lon mesh and the boolean Texas mask a
    single time. These never change between images since they only depend
    on the fixed grid + state boundary, not on the weather values.

    contains_points(radius=...) slightly expands the polygon so grid cells
    that only graze the official boundary still count as inside — reduces
    hairline gaps along the outline after upsampling.
    """
    nlat_up = (len(lats) - 1) * factor + 1
    nlon_up = (len(lons) - 1) * factor + 1
    lat_up = np.linspace(lats[0], lats[-1], nlat_up)
    lon_up = np.linspace(lons[0], lons[-1], nlon_up)
    Lon, Lat = np.meshgrid(lon_up, lat_up)

    # ~quarter of a source cell in degrees — enough to seal edges, not spill
    edge_radius = max(SPACING, 0.5) * 0.35

    mask = np.zeros(Lon.shape, dtype=bool)
    pts = np.column_stack([Lon.ravel(), Lat.ravel()])
    for p in texas_paths:
        mask |= p.contains_points(pts, radius=edge_radius).reshape(Lon.shape)

    # Also build a coarse-grid mask (same shape as the original field) so
    # H/L extrema can be restricted to cells that fall inside Texas.
    Lon_c, Lat_c = np.meshgrid(lons, lats)
    coarse_mask = np.zeros(Lon_c.shape, dtype=bool)
    pts_c = np.column_stack([Lon_c.ravel(), Lat_c.ravel()])
    for p in texas_paths:
        coarse_mask |= p.contains_points(pts_c, radius=edge_radius).reshape(Lon_c.shape)

    return {
        "Lon": Lon, "Lat": Lat, "mask": mask, "factor": factor,
        "coarse_mask": coarse_mask,
    }


def fill_inside_mask(field, mask):
    """Nearest-neighbor fill for any NaN cells that still fall inside Texas.

    After upsampling + masking, a few border pixels can remain empty when the
    source grid is coarse relative to the coastline/panhandle. Pull values
    from the nearest valid cell so the fill meets the black outline.
    """
    from scipy.ndimage import distance_transform_edt

    out = np.array(field, dtype=float, copy=True)
    inside_hole = mask & ~np.isfinite(out)
    if not np.any(inside_hole):
        return out
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    # indices of nearest valid cell for every pixel
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    filled = out[iy, ix]
    out[inside_hole] = filled[inside_hole]
    # Anything outside the state stays NaN for plotting
    out = np.where(mask, out, np.nan)
    return out


def upsample_field(field, target_shape, sigma=SMOOTH_SIGMA):
    """Upsample the coarse data grid onto the precomputed high-res mesh.

    Uses cubic-spline zoom for smooth curves through grid points. A *light*
    Gaussian (SMOOTH_SIGMA) removes spline ringing without the heavy blur
    that made earlier versions look soft/mushy. Set SMOOTH_SIGMA=0 for
    maximum edge sharpness (more Tropical Tidbits–like).
    """
    from scipy.ndimage import zoom, gaussian_filter
    zy = target_shape[0] / field.shape[0]
    zx = target_shape[1] / field.shape[1]
    finite = np.isfinite(field)
    fill_val = np.nanmean(field) if finite.any() else 0.0
    filled = np.where(finite, field, fill_val)
    up = zoom(filled, (zy, zx), order=3, mode="nearest")
    if sigma and sigma > 0:
        up = gaussian_filter(up, sigma=sigma)
    return up


def _contour_levels(var, vmin, vmax):
    """Discrete levels tuned for a Tropical Tidbits–style filled contour look."""
    if var in ("temperature_2m", "apparent_temperature"):
        # ~2–3 °F steps across the observed range
        step = 2.0 if (vmax - vmin) < 40 else 3.0
        lo = math.floor(vmin / step) * step
        hi = math.ceil(vmax / step) * step
        return np.arange(lo, hi + step * 0.5, step)
    if var == "cloud_cover":
        return np.arange(0, 101, 10)
    if var == "precipitation_probability":
        return np.arange(0, 101, 10)
    if var == "relative_humidity_2m":
        return np.arange(0, 101, 10)
    if var == "precipitation":
        # mm thresholds, denser at the low end so light/moderate rain (the
        # most common case) still shows real contrast instead of a single
        # flat "any rain" color.
        base = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
                10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0]
        levels = [x for x in base if x <= vmax * 1.05]
        if not levels or levels[-1] < vmax:
            levels.append(max(vmax, (levels[-1] if levels else 0) + 5.0))
        return np.array(levels, dtype=float)
    if var == "wind_speed_10m":
        # Finer steps than before so the map shows the same granularity the
        # click-probe already reveals underneath.
        rng = vmax - vmin
        step = 1.0 if rng <= 15 else 2.0 if rng <= 30 else 3.0
        lo = 0.0
        hi = math.ceil(vmax / step) * step
        return np.arange(lo, hi + step * 0.5, step)
    if var == "shortwave_radiation":
        # W/m² — typical daytime peaks ~800–1100
        step = 50.0 if vmax <= 600 else 100.0
        lo = 0.0
        hi = math.ceil(max(vmax, 50) / step) * step
        return np.arange(lo, hi + step * 0.5, step)
    # Fallback
    return np.linspace(vmin, vmax, 12)


def _select_extrema(field, mask, lats, lons, n=HL_COUNT, mode="max",
                     min_sep_deg=HL_MIN_SEP_DEG):
    """Pick up to n (lat, lon, value) points that are local extrema and
    spread out across the state, instead of just the single global max/min.

    Greedily walks the sorted valid values and keeps a candidate only if
    it's at least min_sep_deg (in both lat and lon) away from every point
    already picked, so 3 highs/lows don't all land on the same hot spot.
    """
    Lon_c, Lat_c = np.meshgrid(lons, lats)
    valid = mask & np.isfinite(field)
    if not np.any(valid):
        return []
    vals = field[valid]
    las = Lat_c[valid]
    los = Lon_c[valid]
    order = np.argsort(-vals) if mode == "max" else np.argsort(vals)

    selected = []
    for idx in order:
        la, lo, v = float(las[idx]), float(los[idx]), float(vals[idx])
        if all(abs(la - sla) >= min_sep_deg or abs(lo - slo) >= min_sep_deg
               for sla, slo, _ in selected):
            selected.append((la, lo, v))
        if len(selected) >= n:
            break
    return selected


# ── Rendering ───────────────────────────────────────────────────────────

def render_image(field, lats, lons, texas_paths, var, model_label, date, hour,
                  vmin, vmax, cities, value_only_cities, geometry, dpi=130,
                  figsize=(7.2, 6.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(TEXAS_BOUNDS["min_lon"], TEXAS_BOUNDS["max_lon"])
    ax.set_ylim(TEXAS_BOUNDS["min_lat"], TEXAS_BOUNDS["max_lat"])
    ax.set_aspect("equal")
    ax.axis("off")

    if var == "cloud_cover":
        ax.set_facecolor("#1a4a7a")
    else:
        ax.set_facecolor("#dce5eb")

    Lon, Lat = geometry["Lon"], geometry["Lat"]
    coarse_mask = geometry["coarse_mask"]
    # Full rectangular frame is filled (OK, NM, LA, Gulf, Mexico, etc. visible
    # outside the Texas outline). Outline is drawn on top for context.
    field_up = upsample_field(field, Lon.shape)
    field_masked = field_up

    if var == "cloud_cover":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "clouds", ["#1a4a7a", "#8eb4d9", "#ffffff"], N=256
        )
    elif var == "wind_speed_10m":
        # Blue (calm) -> teal -> green (windy), with enough stops for a
        # smooth ramp once it's cut into fine contour levels.
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "wind_bg",
            ["#08306b", "#2166ac", "#4393c3", "#66c2a4", "#238b45", "#00441b"],
            N=256,
        )
    else:
        cmap = plt.get_cmap(VAR_META[var]["cmap"])

    if USE_FILLED_CONTOURS:
        # Tropical Tidbits–style: discrete filled contours + thin isopleths.
        # Contourf is sharper than heavily-blurred pcolormesh and matches
        # operational model-guidance pages more closely.
        levels = _contour_levels(var, vmin, vmax)
        # Avoid all-NaN / flat fields crashing contourf
        finite = field_masked[np.isfinite(field_masked)]
        if len(finite) and np.nanmax(finite) > np.nanmin(finite):
            # Ensure levels span the data a bit
            if levels[0] > np.nanmin(finite):
                levels = np.concatenate([[np.nanmin(finite) - 1e-6], levels])
            if levels[-1] < np.nanmax(finite):
                levels = np.concatenate([levels, [np.nanmax(finite) + 1e-6]])
            levels = np.unique(levels)
            cf = ax.contourf(
                Lon, Lat, field_masked, levels=levels, cmap=cmap,
                vmin=vmin, vmax=vmax, extend="both", zorder=1,
            )
            if CONTOUR_LINE_WIDTH and CONTOUR_LINE_WIDTH > 0:
                ax.contour(
                    Lon, Lat, field_masked, levels=levels,
                    colors="#222222", linewidths=CONTOUR_LINE_WIDTH,
                    alpha=0.55, zorder=2,
                )
        else:
            ax.pcolormesh(Lon, Lat, field_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="nearest", zorder=1)
    else:
        # Continuous shaded look (legacy). Keep light power stretch on precip.
        if var == "precipitation":
            field_disp = np.where(
                field_masked > 0,
                np.power(np.clip(field_masked / max(vmax, 0.05), 0, 1), 0.45) * vmax,
                field_masked
            )
        else:
            field_disp = field_masked
        ax.pcolormesh(Lon, Lat, field_disp, cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="gouraud", zorder=1)

    for p in texas_paths:
        patch = PathPatch(p, facecolor="none", edgecolor="#111", lw=1.3, zorder=5)
        ax.add_patch(patch)

    def _sample(clat, clon):
        i = np.argmin(np.abs(lats - clat))
        j = np.argmin(np.abs(lons - clon))
        return field[i, j]

    def _fmt(val):
        if var == "precipitation":
            return f"{val:.1f}"
        return f"{val:.0f}"

    # Named cities: name + value, with optional inward offset for border cities
    for name, clat, clon in cities:
        val = _sample(clat, clon)
        if not np.isfinite(val):
            continue
        dlon, dlat = CITY_LABEL_OFFSETS.get(name, (0.0, 0.0))
        tlon, tlat = clon + dlon, clat + dlat
        ax.plot(clon, clat, "o", color="white", markersize=3.5,
                markeredgecolor="#111", markeredgewidth=0.6, zorder=6)
        ax.text(tlon, tlat + 0.22, name, fontsize=6.5, ha="center", va="bottom",
                color="#111", fontweight="normal", zorder=7)
        ax.text(tlon, tlat - 0.18, _fmt(val), fontsize=7, ha="center", va="top",
                color="#111", fontweight="bold", zorder=7)

    # Value-only cities (no name)
    for name, clat, clon in value_only_cities:
        val = _sample(clat, clon)
        if not np.isfinite(val):
            continue
        ax.plot(clon, clat, "o", color="white", markersize=3.0,
                markeredgecolor="#111", markeredgewidth=0.5, zorder=6)
        ax.text(clon, clat - 0.18, _fmt(val), fontsize=6.5, ha="center", va="top",
                color="#111", fontweight="bold", zorder=7)

    # Highs / lows: top-N / bottom-N distinct points restricted to cells
    # inside the Texas boundary (skip for precip probability — 0/100 noise).
    if var not in ("precipitation_probability",):
        highs = _select_extrema(field, coarse_mask, lats, lons, mode="max")
        lows = _select_extrema(field, coarse_mask, lats, lons, mode="min")
        for la, lo, v in highs:
            ax.text(lo, la, f"H {_fmt(v)}",
                    fontsize=8, fontweight="bold", color="#111", ha="center", zorder=8)
        for la, lo, v in lows:
            ax.text(lo, la, f"L {_fmt(v)}",
                    fontsize=8, fontweight="bold", color="#111", ha="center", zorder=8)

    ax.text(0.02, 0.97, model_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#111",
            va="top", ha="left", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    # Fixed layout instead of bbox_inches="tight" (which forces an extra,
    # expensive renderer pass on every single save call).
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Worker for parallel rendering ──────────────────────────────────────
# Module-level globals populated via ProcessPoolExecutor initializer so
# large, read-only data (lats/lons/paths/geometry) is set up once per
# worker process instead of being re-pickled with every single task.

_W = {}


def _init_worker(lats, lons, texas_paths, geometry, cities, value_only_cities, dpi):
    _W["lats"] = lats
    _W["lons"] = lons
    _W["texas_paths"] = texas_paths
    _W["geometry"] = geometry
    _W["cities"] = cities
    _W["value_only_cities"] = value_only_cities
    _W["dpi"] = dpi


def _render_task(task):
    field, var, model_label, date, hour, vmin, vmax, fname = task
    img = render_image(
        field, _W["lats"], _W["lons"], _W["texas_paths"], var, model_label,
        date, hour, vmin, vmax, _W["cities"], _W["value_only_cities"],
        _W["geometry"], dpi=_W["dpi"]
    )
    img.save(fname)
    return fname


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
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
             font-family: Segoe UI, Arial, sans-serif; overflow: hidden; }
header {
  height: 56px; display: flex; align-items: center; justify-content: space-between;
  padding: 0 18px; background: var(--panel); border-bottom: 1px solid var(--border);
}
.title { font-size: 17px; font-weight: 650; }
.subtitle { font-size: 11px; color: var(--muted); margin-top: 2px; }
.controls {
  height: 58px; display: flex; align-items: center; gap: 14px;
  padding: 0 18px; background: #151e29; border-bottom: 1px solid var(--border);
}
label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
select, button {
  background: #1a2430; color: var(--text); border: 1px solid #334253;
  border-radius: 5px; padding: 6px 11px; font-size: 12px; cursor: pointer;
}
button:hover { border-color: var(--accent); }
#maps {
  height: calc(100vh - 56px - 58px - 78px);
  display: grid;
  gap: 3px; background: #1c252f; padding: 3px;
}
#maps.layout-1 {
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
}

#maps.layout-2 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr;
}

#maps.layout-3 {
  grid-template-columns: 1fr 1fr 1fr;
  grid-template-rows: 1fr;
}

#maps.layout-4 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
}
.panel {
  position: relative; background: #0d131a; overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}
.panel.hidden { display: none; }
.panel img {
  max-width: 100%; max-height: 100%; object-fit: contain;
  display: block; cursor: crosshair;
}
.timeline {
  height: 78px; padding: 8px 18px; background: var(--panel);
  border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 4px;
}
#timeLabel { font-size: 13px; font-variant-numeric: tabular-nums; }
.slider-wrap { width: 100%; }
input[type=range] {
  width: 100%; accent-color: var(--accent);
  margin: 0; display: block;
}
.hour-ticks {
  display: flex; justify-content: space-between; color: var(--muted);
  font-size: 9px; margin-top: 2px;
  /* Match typical range thumb inset so labels sit under the track, not the ends */
  padding: 0 10px 0 14px;
  box-sizing: border-box;
}
.hour-ticks span { min-width: 2.4em; text-align: center; }
/* Click-to-inspect popup */
#probe {
  display: none; position: fixed; z-index: 50; min-width: 200px; max-width: 280px;
  background: rgba(17, 24, 33, 0.96); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; pointer-events: none;
  box-shadow: 0 8px 28px rgba(0,0,0,0.45); font-size: 12px;
}
#probe .probe-title {
  font-weight: 650; font-size: 12px; margin-bottom: 2px; color: var(--accent);
}
#probe .probe-loc {
  color: var(--muted); font-size: 10px; margin-bottom: 8px;
  font-variant-numeric: tabular-nums;
}
#probe table { width: 100%; border-collapse: collapse; }
#probe td { padding: 2px 0; }
#probe td.k { color: var(--muted); padding-right: 10px; }
#probe td.v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
#probeHint {
  font-size: 11px; color: var(--muted); margin-left: 8px;
}
#modelSelects {
  display: flex; align-items: flex-end; gap: 10px;
}
.model-select-wrap.hidden { display: none; }
.model-select-wrap select:disabled {
  opacity: 0.7; cursor: not-allowed;
}
</style>
</head>
<body>
<header>
  <div>
    <div class="title">Texas Multi-Model Weather Viewer</div>
    <div class="subtitle" id="subtitle"></div>
  </div>
  <div style="display:flex; align-items:center; gap:14px;">
    <div id="lastUpdated" style="font-size:11px; color:var(--muted);"></div>
  </div>
</header>

<div class="controls">
  <div>
    <label>Date</label><br>
    <select id="dateSelect"></select>
  </div>
  <div>
    <label>Variable</label><br>
    <select id="varSelect"></select>
  </div>
  <div>
    <label>Layout</label><br>
    <select id="layoutSelect">
      <option value="1">1 panel</option>
      <option value="2">2 panels</option>
      <option value="3">3 panels</option>
      <option value="4" selected>4 panels</option>
    </select>
  </div>
  <div id="modelSelects">
    <div class="model-select-wrap" id="modelWrap0">
      <label>Model 1</label><br>
      <select id="modelSelect0"></select>
    </div>
    <div class="model-select-wrap" id="modelWrap1">
      <label>Model 2</label><br>
      <select id="modelSelect1"></select>
    </div>
    <div class="model-select-wrap" id="modelWrap2">
      <label>Model 3</label><br>
      <select id="modelSelect2"></select>
    </div>
    <div class="model-select-wrap" id="modelWrap3">
      <label>Model 4</label><br>
      <select id="modelSelect3"></select>
    </div>
  </div>
  <button id="playBtn">▶ Play</button>
  <span id="probeHint">Click a map to probe values</span>
</div>

<div id="maps" class="layout-4">
  <div class="panel" id="panel0"><img id="img0" alt="model 0" data-model-idx="0"></div>
  <div class="panel" id="panel1"><img id="img1" alt="model 1" data-model-idx="1"></div>
  <div class="panel" id="panel2"><img id="img2" alt="model 2" data-model-idx="2"></div>
  <div class="panel" id="panel3"><img id="img3" alt="model 3" data-model-idx="3"></div>
</div>

<div class="timeline">
  <span id="timeLabel"></span>
  <div class="slider-wrap">
    <input type="range" id="hourSlider" min="0" max="23" value="12" step="1">
    <div class="hour-ticks">
      <span>HE 1</span><span>HE 7</span><span>HE 13</span><span>HE 19</span><span>HE 24</span>
    </div>
  </div>
</div>

<div id="probe"></div>

<script>
const DATA = __DATA_JSON__;
const BOUNDS = DATA.bounds;

const dateSelect   = document.getElementById("dateSelect");
const varSelect    = document.getElementById("varSelect");
const layoutSelect = document.getElementById("layoutSelect");
const hourSlider   = document.getElementById("hourSlider");
const timeLabel    = document.getElementById("timeLabel");
const playBtn      = document.getElementById("playBtn");
const probeEl      = document.getElementById("probe");
const mapsEl       = document.getElementById("maps");
const panels       = [0,1,2,3].map(i => document.getElementById("panel"+i));
const imgs         = [0,1,2,3].map(i => document.getElementById("img"+i));
const modelSelects = [0,1,2,3].map(i => document.getElementById("modelSelect"+i));
const modelWraps   = [0,1,2,3].map(i => document.getElementById("modelWrap"+i));

let currentDate = DATA.dates[0];
let currentVar  = DATA.variables[0];
let currentHour = 12;
let layoutCount = 4;
// Which model appears in each panel (indices into DATA.models)
let panelModels = DATA.models.slice();
let playing = false;
let timer = null;
let GRID = null;  // loaded async from grid_data.js

DATA.dates.forEach(d => {
  const o = document.createElement("option");
  o.value = d; o.textContent = d;
  dateSelect.appendChild(o);
});
DATA.variables.forEach(v => {
  const o = document.createElement("option");
  o.value = v; o.textContent = DATA.varMeta[v].label;
  varSelect.appendChild(o);
});

// Populate model dropdowns
function modelLabel(m) {
  return (DATA.modelLabels && DATA.modelLabels[m]) || m;
}
modelSelects.forEach((sel, idx) => {
  DATA.models.forEach(m => {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = modelLabel(m);
    sel.appendChild(o);
  });
  sel.value = DATA.models[idx];
  sel.onchange = () => {
    const chosen = sel.value;
    const prev = panelModels[idx];
    // If another visible panel already has this model, swap
    for (let j = 0; j < layoutCount; j++) {
      if (j !== idx && panelModels[j] === chosen) {
        panelModels[j] = prev;
        modelSelects[j].value = prev;
        break;
      }
    }
    panelModels[idx] = chosen;
    update();
  };
});

function imgName(model, variable, date, hour) {
  return `images/${model}_${variable}_${date}_${String(hour).padStart(2,"0")}.png`;
}

function update() {
  const d = new Date(currentDate + "T00:00:00");
  const dateText = d.toLocaleDateString(undefined, {weekday:"short", month:"short", day:"numeric"});
  // Hour-ending label: data hour h (beginning) spans h:00–(h+1):00 → HE (h+1).
  // Hour 0 (12AM–1AM) → HE 1; hour 7 (7AM–8AM) → HE 8; hour 23 (11PM–12AM) → HE 24.
  const he = currentHour + 1;
  timeLabel.textContent = `${dateText}  •  HE ${he} CT`;

  for (let i = 0; i < layoutCount; i++) {
    const model = panelModels[i];
    imgs[i].src = imgName(model, currentVar, currentDate, currentHour);
    imgs[i].dataset.modelIdx = String(DATA.models.indexOf(model));
  }
  hideProbe();
}

function applyLayout(n) {
  layoutCount = n;
  mapsEl.className = "layout-" + n;

  // Ensure panelModels has n unique models
  const used = new Set();
  const next = [];
  for (let i = 0; i < n; i++) {
    let m = panelModels[i];
    if (!m || used.has(m)) {
      m = DATA.models.find(x => !used.has(x));
    }
    used.add(m);
    next.push(m);
  }
  // If layout is 4, force all four models in default order
  if (n === 4) {
    panelModels = DATA.models.slice();
  } else {
    panelModels = next.concat(DATA.models.filter(m => !used.has(m)));
  }

  panels.forEach((p, i) => {
    if (i < n) p.classList.remove("hidden");
    else p.classList.add("hidden");
  });

  // Show N model dropdowns; disable all when n === 4 (must show all models)
  modelWraps.forEach((w, i) => {
    if (i < n) w.classList.remove("hidden");
    else w.classList.add("hidden");
  });
  modelSelects.forEach((sel, i) => {
    sel.value = panelModels[i];
    sel.disabled = (n === 4);
  });

  update();
}

dateSelect.onchange   = e => { currentDate = e.target.value; currentHour = 12; hourSlider.value = 12; update(); };
varSelect.onchange    = e => { currentVar  = e.target.value; update(); };
layoutSelect.onchange = e => { applyLayout(+e.target.value); };
hourSlider.oninput    = e => { currentHour = +e.target.value; update(); };

playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
  if (playing) {
    hideProbe();
    timer = setInterval(() => {
      currentHour = (currentHour + 1) % 24;
      hourSlider.value = currentHour;
      update();
    }, 700);
  } else clearInterval(timer);
};


document.getElementById("subtitle").textContent =
  `${DATA.npoints} source points • ${DATA.models.length} models • high-res images`;
document.getElementById("lastUpdated").textContent =
  `Last updated: ${DATA.generated_at} CT`;

dateSelect.value = currentDate;
varSelect.value  = currentVar;
layoutSelect.value = String(layoutCount);
hourSlider.value = currentHour;
applyLayout(layoutCount);
update();

/* ── Click-to-inspect ─────────────────────────────────────────────── */

function hideProbe() { probeEl.style.display = "none"; }

function clickToLatLon(img, event) {
  // Map click → lat/lon accounting for object-fit: contain letterboxing.
  const rect = img.getBoundingClientRect();
  const natW = img.naturalWidth || 1;
  const natH = img.naturalHeight || 1;
  const scale = Math.min(rect.width / natW, rect.height / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  const offsetX = (rect.width - dispW) / 2;
  const offsetY = (rect.height - dispH) / 2;
  const x = event.clientX - rect.left - offsetX;
  const y = event.clientY - rect.top - offsetY;
  if (x < 0 || y < 0 || x > dispW || y > dispH) return null;
  const lon = BOUNDS.min_lon + (x / dispW) * (BOUNDS.max_lon - BOUNDS.min_lon);
  // Image top = max_lat (matplotlib saves with north at top)
  const lat = BOUNDS.max_lat - (y / dispH) * (BOUNDS.max_lat - BOUNDS.min_lat);
  return { lat, lon };
}

function nearestIndex(arr, val) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function fmtVal(varKey, v) {
  if (v == null || Number.isNaN(v)) return "—";
  if (varKey === "precipitation") return v.toFixed(1);
  if (varKey === "cloud_cover" || varKey === "precipitation_probability"
      || varKey === "relative_humidity_2m")
    return Math.round(v).toString();
  if (varKey === "shortwave_radiation") return Math.round(v).toString();
  if (varKey === "wind_speed_10m") return v.toFixed(1);
  return Math.round(v).toString();
}

function showProbe(panelIdx, lat, lon, clientX, clientY) {
  if (!GRID) {
    probeEl.innerHTML = `<div class="probe-title">Loading grid…</div>`;
    probeEl.style.display = "block";
    placeProbe(clientX, clientY);
    return;
  }
  const model = panelModels[panelIdx];
  const label = (DATA.modelLabels && DATA.modelLabels[model]) || model;
  const i = nearestIndex(GRID.lats, lat);
  const j = nearestIndex(GRID.lons, lon);
  const flat = i * GRID.nlon + j;
  const sampleLat = GRID.lats[i];
  const sampleLon = GRID.lons[j];

  let rows = "";
  for (const v of DATA.variables) {
    const meta = DATA.varMeta[v] || {};
    const unit = meta.unit || "";
    let val = null;
    try {
      val = GRID.grids[model][v][currentDate][currentHour][flat];
    } catch (e) { val = null; }
    const name = meta.label || v;
    rows += `<tr><td class="k">${name}</td><td class="v">${fmtVal(v, val)}${unit && val != null && !Number.isNaN(val) ? " " + unit : ""}</td></tr>`;
  }

  probeEl.innerHTML =
    `<div class="probe-title">${label}</div>` +
    `<div class="probe-loc">${sampleLat.toFixed(2)}°N, ${Math.abs(sampleLon).toFixed(2)}°W</div>` +
    `<table>${rows}</table>`;
  probeEl.style.display = "block";
  placeProbe(clientX, clientY);
}

function placeProbe(clientX, clientY) {
  const pad = 14;
  const w = probeEl.offsetWidth || 220;
  const h = probeEl.offsetHeight || 180;
  let left = clientX + pad;
  let top = clientY + pad;
  if (left + w > window.innerWidth - 8) left = clientX - w - pad;
  if (top + h > window.innerHeight - 8) top = clientY - h - pad;
  probeEl.style.left = Math.max(8, left) + "px";
  probeEl.style.top = Math.max(8, top) + "px";
}

imgs.forEach((img, idx) => {
  img.addEventListener("click", (e) => {
    const ll = clickToLatLon(img, e);
    if (!ll) { hideProbe(); return; }
    showProbe(idx, ll.lat, ll.lon, e.clientX, e.clientY);
  });
});

// Dismiss probe on outside click / Escape
document.addEventListener("click", (e) => {
  if (!e.target.closest || !e.target.closest(".panel")) hideProbe();
}, true);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideProbe();
});

// Load compact grid data via <script> (works under file://; fetch does not)
(function loadGrid() {
  const s = document.createElement("script");
  s.src = "grid_data.js";
  s.async = true;
  s.onload = () => {
    if (window.GRID) {
      GRID = window.GRID;
      document.getElementById("probeHint").textContent =
        "Click a map to probe all variables";
    }
  };
  s.onerror = () => {
    document.getElementById("probeHint").textContent =
      "Probe unavailable (missing grid_data.js)";
  };
  document.head.appendChild(s);
})();
</script>
</body>
</html>
"""


def export_grid_data(out_dir, lats, lons, values, models, variables, dates):
    """Write grid_data.js used by the click-to-inspect probe.

    Uses a .js file (window.GRID = {...}) so the probe works when opening
    index.html directly from disk — fetch() of a .json is blocked under
    the file:// protocol in most browsers.

    Values are rounded per-variable to keep the file small. Flat row-major
    arrays (nlat * nlon) per hour.
    """
    def _round_arr(var, arr):
        a = np.asarray(arr, dtype=float)
        out = a.ravel().tolist()
        if var == "precipitation":
            return [None if (x is None or not np.isfinite(x)) else round(float(x), 2)
                    for x in out]
        if var in ("cloud_cover", "precipitation_probability", "relative_humidity_2m"):
            return [None if (x is None or not np.isfinite(x)) else int(round(float(x)))
                    for x in out]
        if var == "shortwave_radiation":
            return [None if (x is None or not np.isfinite(x)) else int(round(float(x)))
                    for x in out]
        if var == "wind_speed_10m":
            return [None if (x is None or not np.isfinite(x)) else round(float(x), 1)
                    for x in out]
        # temps
        return [None if (x is None or not np.isfinite(x)) else round(float(x), 1)
                for x in out]

    grids = {}
    for model in models:
        grids[model] = {}
        for var in variables:
            grids[model][var] = {}
            for date in dates:
                grids[model][var][date] = [
                    _round_arr(var, values[model][var][date][h])
                    for h in range(24)
                ]

    payload = {
        "lats": [round(float(x), 4) for x in lats],
        "lons": [round(float(x), 4) for x in lons],
        "nlat": len(lats),
        "nlon": len(lons),
        "grids": grids,
    }
    path = out_dir / "grid_data.js"
    body = "window.GRID=" + json.dumps(payload, separators=(",", ":")) + ";"
    path.write_text(body, encoding="utf-8")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {path.name} ({size_mb:.1f} MB)")
    return path


def main():
    out_dir = Path(OUTPUT_DIR)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Grid + boundary…")
    points, lats, lons = generate_grid(TEXAS_BOUNDS, SPACING)
    print(f"  data grid {len(lats)}×{len(lons)} = {len(points)} points")
    texas_geom = fetch_texas_boundary()
    texas_paths = polygon_to_mpl_path(texas_geom)

    print("Precomputing shared geometry (mesh + mask, once)…")
    geometry = precompute_geometry(lats, lons, texas_paths, factor=UPSAMPLE_FACTOR)

    print("Fetching Open-Meteo…")
    per_point, timestamps = fetch_all(points, MODELS, VARIABLES, FORECAST_DAYS, BATCH_SIZE)
    values, dates = build_values(points, lats, lons, MODELS, VARIABLES, per_point, timestamps)

    if PRECIP_AGG_HOURS > 1:
        print(f"Aggregating precipitation into {PRECIP_AGG_HOURS}-hour blocks…")
        aggregate_precip_3h(values, MODELS, dates, block=PRECIP_AGG_HOURS)

    ranges = compute_ranges(values, MODELS, VARIABLES, dates)

    # Build the full flat task list up front so we can parallelize cleanly.
    # Color scale is per date+hour (shared across models) for auto-scaled vars.
    tasks = []
    for model in MODELS:
        for var in VARIABLES:
            for date in dates:
                for hour in range(24):
                    vmin, vmax = ranges[var][date][hour]
                    field = values[model][var][date][hour]
                    fname = str(img_dir / f"{model}_{var}_{date}_{hour:02d}.png")
                    tasks.append((field, var, MODEL_LABELS[model], date, hour, vmin, vmax, fname))

    total = len(tasks)
    workers = MAX_WORKERS or os.cpu_count() or 4
    print(f"Rendering {total} high-quality images across {workers} workers…")

    count = 0
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(lats, lons, texas_paths, geometry, CITIES, VALUE_ONLY_CITIES, DPI),
    ) as ex:
        futures = [ex.submit(_render_task, t) for t in tasks]
        for fut in as_completed(futures):
            fut.result()  # surface any worker exceptions
            count += 1
            if count % 20 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else float("inf")
                print(f"  {count}/{total} images  ({rate:.1f}/s, ETA {eta:.0f}s)")

    print("Exporting click-probe grid data…")
    export_grid_data(out_dir, lats, lons, values, MODELS, VARIABLES, dates)

    payload = {
        "generated_at": datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M"),
        "models": MODELS,
        "modelLabels": MODEL_LABELS,
        "variables": VARIABLES,
        "varMeta": {
            v: {"label": VAR_META[v]["label"], "unit": VAR_META[v]["unit"]}
            for v in VARIABLES
        },
        "dates": dates,
        "npoints": len(points),
        "bounds": TEXAS_BOUNDS,
    }
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.0f}s → {out_dir.resolve()}")
    print(f"Open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()