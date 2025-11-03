#import all_functions
#import ChatGPT_t
import os
import csv
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def build_interpolated_mask(output_dir, nat, days, Nx=81, Ny=361, TAS=0.5,
                            lon_range=(10, 40), lat_range=(-80, 60), alt_max=38000.0, n=0.25):
    """
    Build an interpolated 3D boolean mask of extreme weather regions directly from CSV files.

    Returns:
        lon_vals, lat_vals, alt_vals, masks_interp_dict
    """
    k = 0.5 + 0.5 / TAS  # weight, kept for context
    lon_vals = np.linspace(*lon_range, Nx)
    lat_vals = np.linspace(*lat_range, Ny)

    masks_interp_dict = {}  # store interpolated masks per day/time

    contents = os.listdir(output_dir)
    if ".DS_Store" in contents:
        contents.remove(".DS_Store")

    for content in contents:
        # Extract day
        content_split_day = content.split(nat)
        Date = content_split_day[1].split("_")
        day = Date[1]
        if day not in days:
            continue
        print(f"\n🗓️ Processing day {day}")

        output_month_day = os.path.join(output_dir, content)
        times = os.listdir(output_month_day)

        for time_file in times:
            # Extract time string
            Time = time_file.split(" ")[1].split("_")[0].split(".")[0]
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

            # Interpolation to model grid
            alt_vals = np.linspace(0, alt_max, len(pressures_real))
            interp_func = interp1d(
                alt_real, mask_real.astype(float), axis=0,
                bounds_error=False, fill_value=0
            )
            mask_real_interp = interp_func(alt_vals) > 0.5

            masks_interp_dict[f"{day}_{Time}"] = mask_real_interp
            print(f"✅ Interpolated mask shape: {mask_real_interp.shape}, True count: {np.sum(mask_real_interp)}")

    return lon_vals, lat_vals, alt_vals, masks_interp_dict
"""

# Example usage
year = "2024"
month = "01"
days = [f"{d:02d}" for d in range(1, 3)]
nat = "north_atlantic_temperature"
output_dir = f"{nat}_{year}_{month}"

lon_vals, lat_vals, alt_vals, masks_interp = build_interpolated_mask(output_dir, nat, days)



# Assuming masks_interp_dict from previous step
# masks_interp_dict keys: "day_Time", values: mask_real_interp (Nz, Ny, Nx)
aday = int(len(masks_interp.keys())/len(days) -1)
time_keys = sorted(masks_interp.keys())  # sort keys to loop in order
time_keys = time_keys[0:aday]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter([], [], [], c='red', s=2, alpha=0.4)

ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_zlabel("Altitude (km)")
ax.set_title("3D Boolean Mask Over Time")

ax.view_init(elev=30, azim=130)

max_points = 100000  # limit points for performance


def update(frame_idx):
    ax.cla()  # clear previous scatter

    mask = masks_interp[time_keys[frame_idx]]
    true_points = np.argwhere(mask)

    if len(true_points) > max_points:
        idx = np.random.choice(len(true_points), max_points, replace=False)
        true_points = true_points[idx]

    z_idx, y_idx, x_idx = true_points.T
    x = lon_vals[x_idx]
    y = lat_vals[y_idx]
    z = alt_vals[z_idx] / 1000.0  # km

    ax.scatter(x, y, z, c='red', s=2, alpha=0.4)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_zlabel("Altitude (km)")
    ax.set_title(f"3D Boolean Mask: {time_keys[frame_idx]}")
    ax.view_init(elev=30, azim=130)

    return scat,


anim = FuncAnimation(fig, update, frames=len(time_keys), interval=1000, blit=False)
plt.show()"""
