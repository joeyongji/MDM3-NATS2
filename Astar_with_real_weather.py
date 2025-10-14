import os
import sys
import math
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from mpl_toolkits.mplot3d import Axes3D

# --------------------- User configuration ---------------------
PROJECT_LOCATION = os.path.expanduser("~/PycharmProjects/NATS2/")
FILE_PATH = os.path.join(PROJECT_LOCATION, "weather_data_05_2025.grib")  # path to GRIB/NetCDF file
FETCH_FROM_CDS = False  # Set True to download via CDS (requires cdsapi and credentials)
SAVE_CSV = False
OUTPUT_CSV = os.path.join(PROJECT_LOCATION, "north_atlantic_temperature_05_2025.csv")
OUTPUT_PNG = os.path.join(PROJECT_LOCATION, "astar_3d_weather.png")

# Grid used by your A* (keep consistent with your previous script)
Nx, Ny, Nz = 15, 15, 5
lon_min, lon_max = -70.0, -10.0
lat_min, lat_max = 40.0, 60.0
alt_min, alt_max = 30000.0, 38000.0
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

# A* tuning
TAS = 0.5  # normalized ground speed (grids per time unit)
k = 0.5 + 0.5 / TAS  # combined weight

# Weather thresholds (tune these to be more/less conservative)
W_MPS_THRESHOLD = 0.8    # vertical velocity magnitude (m/s) considered hazardous
VORTICITY_THRESHOLD = 5e-5  # s^-1 considered hazardous

# If you want to force a particular ERA5 pressure level (hPa), set here:
TARGET_PRESSURE_HPA = 200
TARGET_TIME_INDEX = 0  # which time index to use from file (0-based)

# --------------------- Helper: Safe xarray open ---------------------

def open_era5_dataset_safe(path_or_bytes, engine_hint=None):
    """Try to open with cfgrib first, then fall back to netCDF (scipy engine).

    Returns xarray.Dataset
    """
    try:
        if engine_hint == 'cfgrib' or engine_hint is None:
            # try cfgrib first
            ds = xr.open_dataset(path_or_bytes, engine='cfgrib')
            print("Opened dataset with cfgrib")
            return ds
    except Exception as e:
        print("cfgrib open failed:", e)
    try:
        ds = xr.open_dataset(path_or_bytes)
        print("Opened dataset with default engine (likely netCDF)")
        return ds
    except Exception as e:
        print("Failed to open dataset with default engine:", e)
        raise

# --------------------- Data conversion utilities ---------------------

def find_var(ds, candidates):
    for c in candidates:
        if c in ds:
            return c
    return None


def extract_level_and_time(ds, varname, pressure_hpa=TARGET_PRESSURE_HPA, time_idx=TARGET_TIME_INDEX):
    """Return DataArray for given variable interpolated or selected at requested pressure hPa and time.

    The function tries to be robust to different coordinate names (level, isobaricInhPa, pressure, etc.).
    """
    da = ds[varname]
    # find pressure coordinate name
    p_coord = None
    for cn in ('isobaricInhPa', 'level', 'pressure', 'plev'):
        if cn in da.coords:
            p_coord = cn
            break
    # find time coord name
    t_coord = None
    for tn in ('time', 'valid_time'):
        if tn in da.coords:
            t_coord = tn
            break
    # select time
    if t_coord is not None:
        try:
            da = da.isel({t_coord: time_idx})
        except Exception:
            # if indexing fails, try to use first time
            da = da.isel({t_coord: 0})
    # select pressure
    if p_coord is not None:
        try:
            # Some datasets store pressure in hPa already (isobaricInhPa)
            if p_coord == 'isobaricInhPa' or p_coord == 'plev':
                # make sure dtype numeric
                da = da.sel({p_coord: pressure_hpa}, method='nearest')
            else:
                da = da.sel({p_coord: pressure_hpa}, method='nearest')
        except Exception:
            # if can't select, just take first level
            da = da.isel({p_coord: 0})
    return da

# --------------------- Weather -> blocked mask ---------------------

def build_blocked_mask_from_era5(ds, nx=Nx, ny=Ny, nz=Nz,
                                 lon_vals_local=lon_vals, lat_vals_local=lat_vals,
                                 w_thresh=W_MPS_THRESHOLD, vo_thresh=VORTICITY_THRESHOLD,
                                 pressure_hpa=TARGET_PRESSURE_HPA, time_idx=TARGET_TIME_INDEX):
    """Construct (Nz, Ny, Nx) boolean mask from ERA5 dataset ds.

    Steps:
      - find vertical velocity variable and convert to m/s
      - find vorticity variable
      - interpolate lat/lon to target grid
      - apply thresholds
      - replicate to Nz levels (simple approach)
    """
    # discover variable names
    w_var = find_var(ds, ['w', 'vertical_velocity', 'vertical_velocity_pa_s', 'vertical_velocity_pa_s'])
    vo_var = find_var(ds, ['vo', 'vorticity', 'vorticity_era5'])
    t_var = find_var(ds, ['t', 'temperature'])

    if w_var is None:
        raise ValueError('Could not find vertical-velocity variable in dataset (w).')
    if t_var is None:
        raise ValueError('Could not find temperature variable in dataset (t).')

    # extract DataArrays at target pressure and time
    w_da = extract_level_and_time(ds, w_var, pressure_hpa=pressure_hpa, time_idx=time_idx)
    t_da = extract_level_and_time(ds, t_var, pressure_hpa=pressure_hpa, time_idx=time_idx)

    # vorticity optional
    vo_da = None
    if vo_var is not None:
        vo_da = extract_level_and_time(ds, vo_var, pressure_hpa=pressure_hpa, time_idx=time_idx)

    # coordinate names for lat/lon (attempt common names)
    lat_name = 'latitude' if 'latitude' in w_da.coords else ('lat' if 'lat' in w_da.coords else list(w_da.coords)[0])
    lon_name = 'longitude' if 'longitude' in w_da.coords else ('lon' if 'lon' in w_da.coords else list(w_da.coords)[1])

    # interpolate to our small Nx x Ny grid
    print(f"Interpolating ERA5 fields to Nx={nx}, Ny={ny} grid (lon range {lon_vals_local[0]}..{lon_vals_local[-1]})")
    interp_kwargs = {lat_name: lat_vals_local, lon_name: lon_vals_local}
    try:
        w_interp = w_da.interp(interp_kwargs, method='linear')
        t_interp = t_da.interp(interp_kwargs, method='linear')
    except Exception as e:
        print('Interpolation failed, attempting fallback by transposing order or using nearest:', e)
        w_interp = w_da.interp({lon_name: lon_vals_local, lat_name: lat_vals_local}, method='nearest')
        t_interp = t_da.interp({lon_name: lon_vals_local, lat_name: lat_vals_local}, method='nearest')

    if vo_da is not None:
        try:
            vo_interp = vo_da.interp(interp_kwargs, method='linear')
        except Exception:
            vo_interp = vo_da.interp(interp_kwargs, method='nearest')
    else:
        vo_interp = None

    # compute density rho = p/(R_d * T)
    R_d = 287.05
    g = 9.80665
    p_pa = pressure_hpa * 100.0

    t_vals = t_interp.values  # shape (lat, lon) or (...)
    # ensure w is in Pa/s (ERA5 vertical_velocity is Pa/s) then convert to m/s
    w_vals_pa_s = w_interp.values
    rho = p_pa / (R_d * t_vals)
    # avoid zero
    rho = np.where(rho == 0, 1e-6, rho)
    w_mps = - w_vals_pa_s / (rho * g)

    # build boolean mask using thresholds
    mask2d = np.abs(w_mps) > w_thresh
    if vo_interp is not None:
        vo_vals = vo_interp.values
        mask2d = mask2d | (np.abs(vo_vals) > vo_thresh)

    # mask2d shape should be (lat, lon). We'll reorder to (ny, nx)
    # Many xarray results have dims (latitude, longitude)
    # Ensure mask2d is indexed [iy, ix]
    if mask2d.shape != (ny, nx):
        # try transpose if shape swapped
        if mask2d.shape == (nx, ny):
            mask2d = mask2d.T
        else:
            # attempt to reshape if dims reversed
            mask2d = np.array(mask2d)
            mask2d = mask2d.reshape((ny, nx))

    # replicate vertically
    blocked = np.repeat(mask2d[np.newaxis, :, :], nz, axis=0)

    return blocked, {'w_da': w_interp, 't_da': t_interp, 'vo_da': vo_interp}

# --------------------- A* implementation (adapted from your code) ---------------------

np.random.seed(53)

moves = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == dy == dz == 0:
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            moves.append((dx, dy, dz, dist))


def in_bounds(ix, iy, iz):
    return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz


def passable(ix, iy, iz, mask):
    return not bool(mask[iz, iy, ix])


def neighbors(ix, iy, iz, mask):
    for dx, dy, dz, dist in moves:
        nx_, ny_, nz_ = ix + dx, iy + dy, iz + dz
        if in_bounds(nx_, ny_, nz_) and passable(nx_, ny_, nz_, mask):
            yield nx_, ny_, nz_, dist


def heuristic(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist * k


def random_free_cell(mask, margin=1, min_dist=5):
    if Nx <= 2 * margin or Ny <= 2 * margin:
        margin = 0
    for _ in range(20000):
        ix = np.random.randint(margin, Nx - margin) if Nx > 2 * margin else np.random.randint(0, Nx)
        iy = np.random.randint(margin, Ny - margin) if Ny > 2 * margin else np.random.randint(0, Ny)
        iz = np.random.randint(0, Nz)
        if passable(ix, iy, iz, mask):
            for _ in range(2000):
                jx = np.random.randint(margin, Nx - margin) if Nx > 2 * margin else np.random.randint(0, Nx)
                jy = np.random.randint(margin, Ny - margin) if Ny > 2 * margin else np.random.randint(0, Ny)
                jz = np.random.randint(0, Nz)
                dist_val = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2 + (iz - jz) ** 2)
                if passable(jx, jy, jz, mask) and dist_val >= min_dist:
                    return (ix, iy, iz), (jx, jy, jz)
    raise RuntimeError("No valid start-goal pair found.")


def astar_3d(start, goal, mask):
    openh = []
    heappush(openh, (0.0, start))
    g = {start: 0.0}
    g_dist = {start: 0.0}
    parent = {start: None}
    closed = set()
    while openh:
        _, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, g[u], g_dist[u], closed
        ux, uy, uz = u
        for nx_, ny_, nz_, dist in neighbors(ux, uy, uz, mask):
            v = (nx_, ny_, nz_)
            time = dist / TAS
            edge_cost = 0.5 * dist + 0.5 * time
            new_g = g[u] + edge_cost
            if v not in g or new_g < g[v]:
                g[v] = new_g
                g_dist[v] = g_dist[u] + dist
                parent[v] = u
                f_cost = new_g + heuristic(v, goal)
                heappush(openh, (f_cost, v))
    return None, np.inf, np.inf, closed

# --------------------- Visualization helpers ---------------------

def idx_to_phys_x(ix):
    return np.interp(ix, np.arange(Nx), lon_vals)

def idx_to_phys_y(iy):
    return np.interp(iy, np.arange(Ny), lat_vals)

def idx_to_phys_z(iz):
    return np.interp(iz, np.arange(Nz), alt_vals)

# --------------------- Main execution ---------------------

def main():
    # 1) Open dataset
    if FETCH_FROM_CDS:
        try:
            import cdsapi
        except Exception:
            print('cdsapi not installed; set FETCH_FROM_CDS=False or install cdsapi')
            return
        # Example minimal fetch (user should adapt). Here we assume user has set up CDS keys.
        c = cdsapi.Client()
        # implement a small fetch if desired (omitted to avoid forcing credentials in this script)
        print('FETCH_FROM_CDS requested but fetch logic is left to user to customize in this script.')
        return

    if not os.path.exists(FILE_PATH):
        print('FILE_PATH not found:', FILE_PATH)
        print('Please set FILE_PATH to a valid ERA5 GRIB or NetCDF file (downloaded previously).')
        return

    ds = open_era5_dataset_safe(FILE_PATH)

    # Build mask
    blocked, aux = build_blocked_mask_from_era5(ds, nx=Nx, ny=Ny, nz=Nz,
                                               lon_vals_local=lon_vals, lat_vals_local=lat_vals,
                                               w_thresh=W_MPS_THRESHOLD, vo_thresh=VORTICITY_THRESHOLD,
                                               pressure_hpa=TARGET_PRESSURE_HPA, time_idx=TARGET_TIME_INDEX)

    # Optionally save CSV of fields used (temperature / w_mps)
    if SAVE_CSV:
        # Build dataframe from aux['w_da'] and aux['t_da']
        w_da = aux['w_da']
        t_da = aux['t_da']
        # convert to DataFrame (flatten)
        df_w = w_da.to_dataframe().reset_index()
        df_t = t_da.to_dataframe().reset_index()
        df = df_t.merge(df_w, on=[c for c in df_t.columns if c in df_w.columns])
        df.to_csv(OUTPUT_CSV, index=False)
        print('Saved CSV to', OUTPUT_CSV)

    # 2) Run A*
    try:
        start_point, goal_point = random_free_cell(blocked, min_dist=5)
    except RuntimeError as e:
        print('Failed to find start/goal on blocked mask:', e)
        return

    path_obs, cost_obs, dist_obs, explored = astar_3d(start_point, goal_point, blocked)
    empty_mask = np.zeros_like(blocked, dtype=bool)
    path_free, cost_free, dist_free, _ = astar_3d(start_point, goal_point, empty_mask)

    # 3) Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D A*: Dual Objective (Dist+Time) with ERA5-derived No-fly Zones")

    # Show blocked grid cubes as semi-transparent voxels (coarse)
    # We'll scatter blocked cell centers
    zs, ys, xs = np.where(blocked)
    if len(xs) > 0:
        ax.scatter([idx_to_phys_x(x) for x in xs], [idx_to_phys_y(y) for y in ys], [idx_to_phys_z(z) for z in zs],
                   s=40, c='red', alpha=0.25, label='Weather-blocked')

    # explored nodes (sample)
    if explored:
        ex = list(explored)
        if len(ex) > 1500:
            ex = random.sample(ex, 1500)
        ax.scatter([idx_to_phys_x(x) for x, y, z in ex], [idx_to_phys_y(y) for x, y, z in ex],
                   [idx_to_phys_z(z) for x, y, z in ex], s=3, c='gray', alpha=0.15, label='Explored')

    # paths
    if path_free:
        ax.plot([idx_to_phys_x(x) for x, y, z in path_free], [idx_to_phys_y(y) for x, y, z in path_free],
                [idx_to_phys_z(z) for x, y, z in path_free], linestyle='--', linewidth=2,
                label=f'Free-space (cost={cost_free:.2f}, dist={dist_free:.2f})')
    if path_obs:
        ax.plot([idx_to_phys_x(x) for x, y, z in path_obs], [idx_to_phys_y(y) for x, y, z in path_obs],
                [idx_to_phys_z(z) for x, y, z in path_obs], linewidth=3,
                label=f'Weather-avoiding (cost={cost_obs:.2f}, dist={dist_obs:.2f})')

    sx, sy, sz = start_point
    gx, gy, gz = goal_point
    ax.scatter([idx_to_phys_x(sx)], [idx_to_phys_y(sy)], [idx_to_phys_z(sz)], s=60, marker='o', label='Start', color='green')
    ax.scatter([idx_to_phys_x(gx)], [idx_to_phys_y(gy)], [idx_to_phys_z(gz)], s=80, marker='*', label='Goal', color='purple')

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_zlabel('Flight level (ft)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=170, bbox_inches='tight')
    print('Saved plot to', OUTPUT_PNG)
    plt.show()

    # 4) Print quantification
    if path_obs and path_free:
        increase_dist = (dist_obs - dist_free) / dist_free * 100 if dist_free > 0 else 0
        increase_cost = (cost_obs - cost_free) / cost_free * 100 if cost_free > 0 else 0
        print(f"Free-space Path: Cost={cost_free:.2f}, Distance={dist_free:.2f} grid units")
        print(f"Weather Path:    Cost={cost_obs:.2f} (+{increase_cost:.1f}%), Distance={dist_obs:.2f} grid units ({increase_dist:+.1f}% change)")


if __name__ == '__main__':
    main()
