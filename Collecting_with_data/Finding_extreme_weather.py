"""
def build_interpolated_mask(output_dir, nat, days, time=None, Nx=81, Ny=361, TAS=0.5, lon_range=(10, 40), lat_range=(-80, 60), alt_max=38000.0, n=0.25,):
    """
"""
    Build an interpolated 3D boolean mask of extreme weather regions directly from CSV files.

    Args:
        output_dir (str): Base directory containing weather CSV files.
        nat (str): Natural identifier string in filenames (used for splitting).
        days (list[str]): List of days to include (ignored if `specific_day` is set).
        Nx, Ny (int): Grid size in longitude and latitude.
        TAS (float): Threshold scaling factor.
        lon_range, lat_range (tuple): Geographic bounds.
        alt_max (float): Maximum altitude.
        n (float): Threshold for extreme vertical velocity.
        specific_day (str, optional): e.g. "01" → only process this day.
        specific_time (str, optional): e.g. "01-00-00" → only process this time (requires `specific_day` too).

    Returns:
        lon_vals, lat_vals, alt_vals, masks_interp_dict
    """"""
    import os
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d

    k = 0.5 + 0.5 / TAS  # weight, kept for context
    lon_vals = np.linspace(*lon_range, Nx)
    lat_vals = np.linspace(*lat_range, Ny)

    masks_interp_dict = {}

    contents = os.listdir(output_dir)
    if ".DS_Store" in contents:
        contents.remove(".DS_Store")

    for content in contents:
        content_split_day = content.split(nat)
        Date = content_split_day[1].split("_")
        day = Date[1]

        print(f"\n🗓️ Processing day {day}")

        output_month_day = os.path.join(output_dir, content)
        times = os.listdir(output_month_day)

        for time_file in times:
            # Extract time string
            Time = time_file.split(" ")[1].split("_")[0].split(".")[0]

            # ✅ Only process specific time if requested
            if time is not None and Time != time:
                continue

            print(f"⏰ Processing time {Time} {day}")

            df = pd.read_csv(os.path.join(output_month_day, time_file))
            df['isobaricInhPa'] = pd.to_numeric(df['isobaricInhPa'], errors='coerce')
            df['w_mps'] = pd.to_numeric(df['w_mps'], errors='coerce')
            df = df.dropna(subset=['isobaricInhPa', 'w_mps'])

            pressures_real = np.array(sorted(df['isobaricInhPa'].unique(), reverse=True))
            alt_real = (1 - pressures_real / pressures_real.max()) * alt_max

            layers_reshaped = []
            for p in pressures_real:
                layer = df[df['isobaricInhPa'] == p]['w_mps'].abs() > n
                layers_reshaped.append(layer.to_numpy().reshape(Ny, Nx))

            mask_real = np.stack(layers_reshaped, axis=0)

            alt_vals = np.linspace(0, alt_max, len(pressures_real))
            interp_func = interp1d(
                alt_real, mask_real.astype(float), axis=0,
                bounds_error=False, fill_value=0
            )
            mask_real_interp = interp_func(alt_vals) > 0.5

            masks_interp_dict[f"{day}_{Time}"] = mask_real_interp
            print(f"✅ Interpolated mask shape: {mask_real_interp.shape}, True count: {np.sum(mask_real_interp)}")

            # ✅ Stop early if only one specific time/day requested
            if time:
                return lon_vals, lat_vals, alt_vals, masks_interp_dict

    return lon_vals, lat_vals, alt_vals, masks_interp_dict
"""
import cdsapi
from numpy.f2py.crackfortran import endifs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
import os
import numpy as np
import math

# === Load the GRIB file ===
year = ["2025"]
month = [f"{d:02d}" for d in [7]]#, 8, 9, 10]]
day = [f"{d:02d}" for d in [1]]#,5,10,15,20,25,30]] # Auto-generate days 01–31 range(1, 3)
time = [f"{h:02d}:00" for h in range(24)] # Auto-generate hours 00:00–23:00
pressure_level = ["150", "175", "200", "250", "300"] # 200 hPa
area = [60, -80, 40, 10] # North, West, South, East (lat/lon)
d = 1

# Define output file name
if len(day) == 1:
    name_day = day[0]
else:
    name_day = f"{day[0]}_{day[-1]}"

if len(month) == 1:
    name_month = month[0]
else:
    name_month = f"{month[0]}_{month[-1]}"

if len(year) == 1:
    name_year = year[0]
else:
    name_year = f"{year[0]}_{year[-1]}"

target = f"weather_data_{name_day}_{name_month}_{name_year}.grib"

ds = xr.open_dataset(target, engine="cfgrib")

if ds.longitude.max() > 180:
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby("longitude")

# Focused subset
ds_na = ds.sel(latitude=slice(70, 0), longitude=slice(-80, 10))

ciwc = ds_na["ciwc"]
clwc = ds_na["clwc"]
d = ds_na["d"] #'divergence_of_wind'
r = ds_na["r"] #'relative_humidity'
t = ds_na["t"] #'air_temperature'
u = ds_na["u"] #'eastward_wind'
v = ds_na["v"] #'northward_wind'
vo = ds_na["vo"] #'atmosphere_relative_vorticity'
w_pa = ds_na["w"] #'lagrangian_tendency_of_air_pressure'
z = ds_na["z"]#'geopotential'

R_d = 287.05
g = 9.80665
p = ds_na["isobaricInhPa"] * 100
rho = p / (R_d * t)
w_mps = -w_pa / (rho * g)
w_mps.name = "w_mps"

d_values = d.values
r_values = r.values
t_values = t.values
u_values = u.values
v_values = v.values
vo_values = vo.values
w_values = w_mps.values
z_values = z.values


strong_upward_motion = 5 # < m/s
Divergence = 5*10**(-5) # < s^-1
Vorticity =  1*10**(-4) # < s^-1
High_humidity = 80 # < %
Temperature = [233.15, 273.15] # > k > k
High_cloud_liquid_ice_water_content = 0.01 # < g/kg
High_cloud_liquid_water_content = 0.05 # < g/kg
U_V_Wind_Components = 60 # < m/s

#Warm/moist lower layer + cold/dry upper layer (unstable lapse rate)
#Low-level convergence + upper-level divergence
#Strong wind shear or positive vorticity increase
diff_all = []
sum_diff= []
for t in range(1, len(d_values[1])):
    diff = d_values[:, t-1, :, :] - d_values[:, t, :, :]
    for r in diff:
        lo = sum(abs(r))
        sum_diffl = sum(lo)/(len(r)*len(lo))
        sum_diff.append(sum_diffl)
    diff_all.append(diff)
s = sum(sum_diff)/len(sum_diff)
values_w = w_values > strong_upward_motion
# Compare slice 1 vs 2
diff_1_2 = d_values[:, 1, :, :] -  d_values[:, 2, :, :]
# Compare slice 3 vs 2
diff_3_2 = d_values[:, 3, :, :] - d_values[:, 2, :, :]
values_d_1 = abs(diff_1_2) >= Divergence
values_d_2 = abs(diff_3_2) >= Divergence
values_d = values_d_1+values_d_2 #[d_values > ]
values_r = r_values > High_humidity
values_t = (t_values >= Temperature[0]) & (t_values <= Temperature[1])
values_vo = vo_values > Vorticity
u_v_values = np.sqrt(u_values**2 + v_values**2)
values_u_v = u_v_values > U_V_Wind_Components

a = values_w[:, 2, :, :] + values_d + values_r[:, 2, :, :] + values_t[:, 2, :, :] + values_vo[:, 2, :, :] + values_u_v[:, 2, :, :]
a_1 = a[1]

sum_values_w = sum(sum(values_w[:, 2, :, :][1]))
sum_values_d = sum(sum(values_d[1]))
sum_values_r = sum(sum(values_r[:, 2, :, :][1]))
sum_values_t = sum(sum(values_t[:, 2, :, :][1]))
sum_values_vo = sum(sum(values_vo[:, 2, :, :][1]))
sum_values_u_v = sum(sum(values_u_v[:, 2, :, :][1]))


# Replace this with your matrix:
# matrix = [[1,0,1], [0,1,0], ...]
# or as a numpy array: matrix = np.array(..., dtype=bool)
matrix = a_1

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(matrix, interpolation='nearest', aspect='auto')
ax.set_title("Boolean matrix plot (True=1(Yellow), False=0(Purple)")
ax.set_xlabel("Columns")
ax.set_ylabel("Rows")

nrows, ncols = matrix.shape
if nrows <= 30 and ncols <= 30:
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticklabels(np.arange(nrows))
else:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
fig.savefig("boolean_matrix.png", dpi=150, bbox_inches='tight')
