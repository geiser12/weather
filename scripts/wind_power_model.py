#!/usr/bin/env python3
"""
wind_power_model.py
====================
Converts a model's gridded wind_speed_80m field (mph, as fetched from
Open-Meteo in texas_weather_viewer.py) into an estimated Texas wind-fleet
MW output, using one generic utility-scale turbine power curve applied to
every farm in the fleet.

This is a *rough* fleet-level estimate meant for comparing model skill
(HRRR vs ECMWF vs GFS vs NAM against each other, and eventually against
ERCOT's published forecast) - not a bankable production forecast:
  - one generic power curve stands in for many different real turbine
    models / hub heights
  - no wake losses, curtailment, transmission outages, or maintenance
  - wind speed is taken at 80m from the NWP model with no further
    hub-height extrapolation (most TX fleet hub heights are 80-100m, so
    this is a reasonable single-level proxy, not exact)
"""
from __future__ import annotations

import numpy as np

MPH_TO_MS = 0.44704

# Generic utility-scale wind turbine power curve (approximate composite of
# common 2-3 MW class turbines), as (wind speed m/s, fraction of rated
# power). Linearly interpolated between points, clamped outside the range.
GENERIC_POWER_CURVE = [
    (0.0, 0.00), (3.0, 0.00), (4.0, 0.03), (5.0, 0.09), (6.0, 0.18),
    (7.0, 0.31), (8.0, 0.47), (9.0, 0.64), (10.0, 0.81), (11.0, 0.93),
    (12.0, 0.99), (13.0, 1.00), (25.0, 1.00), (25.01, 0.00), (40.0, 0.00),
]
_CURVE_X = np.array([p[0] for p in GENERIC_POWER_CURVE])
_CURVE_Y = np.array([p[1] for p in GENERIC_POWER_CURVE])


def capacity_factor(wind_speed_ms: np.ndarray) -> np.ndarray:
    """Vectorised generic power curve: wind speed (m/s, any shape) -> capacity factor in [0, 1]."""
    return np.interp(wind_speed_ms, _CURVE_X, _CURVE_Y, left=0.0, right=0.0)


def _bilinear_weights(lats: np.ndarray, lons: np.ndarray, lat: float, lon: float):
    """Grid indices/weights for bilinear sampling of a (nlat, nlon) field at (lat, lon)."""
    nlat, nlon = len(lats), len(lons)
    gy = np.interp(lat, lats, np.arange(nlat))
    gx = np.interp(lon, lons, np.arange(nlon))
    j0 = int(np.clip(np.floor(gy), 0, nlat - 2))
    i0 = int(np.clip(np.floor(gx), 0, nlon - 2))
    fy, fx = gy - j0, gx - i0
    return j0, i0, fy, fx


def sample_series(field_txy: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                   lat: float, lon: float) -> np.ndarray:
    """Bilinearly sample a (T, nlat, nlon) field at one (lat, lon) -> (T,) series."""
    j0, i0, fy, fx = _bilinear_weights(lats, lons, lat, lon)
    a = field_txy[:, j0, i0]
    b = field_txy[:, j0, i0 + 1]
    c = field_txy[:, j0 + 1, i0]
    d = field_txy[:, j0 + 1, i0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fleet_mw_timeseries(wind_speed_mph_txy: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                         fleet: list[dict]) -> np.ndarray:
    """
    wind_speed_mph_txy : (T, nlat, nlon) array - one model's wind_speed_80m field, mph
    lats, lons         : 1-D grid coordinate arrays matching that field
    fleet              : list of {lat, lon, capacity_mw, ...} (see uswtdb_fleet.py)

    Returns (T,) array: estimated total TX wind fleet MW for that model, each hour.
    """
    T = wind_speed_mph_txy.shape[0]
    total_mw = np.zeros(T, dtype=np.float64)
    for farm in fleet:
        spd_ms = sample_series(wind_speed_mph_txy, lats, lons, farm["lat"], farm["lon"]) * MPH_TO_MS
        total_mw += capacity_factor(spd_ms) * farm["capacity_mw"]
    return total_mw


def fleet_total_capacity_mw(fleet: list[dict]) -> float:
    return float(sum(f["capacity_mw"] for f in fleet))