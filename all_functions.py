
import cdsapi
from numpy.f2py.crackfortran import endifs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

def fetch_data(target, year, month, day, time, pressure_level, area):
    # Initialize the CDS API client
    c = cdsapi.Client()

    # Define the dataset and request parameters
    dataset = "reanalysis-era5-pressure-levels"

    request = {
        "product_type": "reanalysis",
        "variable": [
            "potential_vorticity",
            "relative_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "vorticity"
        ],
        "year": year, #"2023"
        "month": month, #"07"
        "day": day,
        "time": time,
        "pressure_level": pressure_level,
        "area": area,
        "format": "grib"  # Must be 'grib' or 'netcdf'
    }

    # Retrieve and download data
    c.retrieve(dataset, request, target)

    print("✅ Data retrieval complete! File saved as:", target)

def grib_to_csv_with_w(file_path, output_csv, save_csv):
    """
    Converts ERA5 GRIB file to CSV including:
    - Temperature (t)
    - Wind components (u, v)
    - Vorticity (vo)
    - Vertical velocity (w) converted from Pa/s → m/s

    Parameters:
        file_path (str): Path to the GRIB file
        output_csv (str): Output CSV file path
    """

    # Open GRIB dataset
    ds = xr.open_dataset(file_path, engine="cfgrib", decode_times=True, decode_timedelta=True)

    # Fix longitude if needed (0–360 → -180–180)
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")

    # Subset North Atlantic region
    ds_na = ds.sel(latitude=slice(70, 0), longitude=slice(-80, 10))

    # Select one time step as example (change or loop over all times if needed)
    time_idx = 1
    t = ds_na['t'].isel(time=time_idx)
    u = ds_na['u'].isel(time=time_idx)
    v = ds_na['v'].isel(time=time_idx)
    vo = ds_na['vo'].isel(time=time_idx)
    # Select vertical velocity at the desired time step
    w_pa = ds_na['w'].isel(time=time_idx)

    # Pressure in Pa (from isobaricInhPa)
    p = ds_na['isobaricInhPa'] * 100  # hPa → Pa

    # Air density
    R_d = 287.05
    g = 9.80665
    rho = p / (R_d * t)

    # Convert vertical velocity to m/s
    w_mps = - w_pa / (rho * g)

    # Give the DataArray a name
    w_mps.name = "w_mps"

    # Convert to DataFrame
    df_w = w_mps.to_dataframe().reset_index()

    # Convert each variable to DataFrame
    df_t = t.to_dataframe().reset_index()
    df_u = u.to_dataframe().reset_index()
    df_v = v.to_dataframe().reset_index()
    df_vo = vo.to_dataframe().reset_index()

    # Merge all DataFrames
    df_combined = (
        df_t
        .merge(df_u[['latitude', 'longitude', 'time', 'u']], on=['latitude', 'longitude', 'time'])
        .merge(df_v[['latitude', 'longitude', 'time', 'v']], on=['latitude', 'longitude', 'time'])
        .merge(df_vo[['latitude', 'longitude', 'time', 'vo']], on=['latitude', 'longitude', 'time'])
        .merge(df_w[['latitude', 'longitude', 'time', 'w_mps']], on=['latitude', 'longitude', 'time'])
    )

    # Save to CSV
    if save_csv:
        df_combined.to_csv(output_csv, index=False)
        print(f"✅ CSV saved: {output_csv}")
    print(f"✅ returned: t, w_pa")
    return t, w_pa

def create_map_plot(data):
    """
    Plot temperature data (e.g. ERA5 at 200 hPa) over the North Atlantic region.

    Parameters:
        data (xarray.DataArray): The temperature field for a single time step.
    """
    # === Create the map ===
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([280, 360, 35, 60])  # [lon_min, lon_max, lat_min, lat_max]

    # === Add map features ===
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # === Plot the temperature field ===
    temp_plot = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        cbar_kwargs={"label": "Temperature (K)"}
    )

    # === Dynamic title based on timestamp ===
    if "time" in data.coords:
        timestamp = pd.to_datetime(data.time.values)
        ax.set_title(f"ERA5 Temperature (200 hPa) - North Atlantic\n{timestamp:%Y-%m-%d %H:%M UTC}")
    else:
        ax.set_title("ERA5 Temperature (200 hPa) - North Atlantic")

    plt.tight_layout()
    plt.show()
