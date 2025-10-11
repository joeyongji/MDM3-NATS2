import all_functions

Project_location = "Users/11722/PycharmProjects/MDM3_A_NATS/"
Flag_Fetched_data = 0
Flag_grib_to_csv = 1
Flag_plot = 1
save_csv = 0

# === Load the GRIB file ===
year = "2025"
month = "05" #print([f"{d:02d}" for d in range(1, 13)])
day = [f"{d:02d}" for d in range(1, 32)] # Auto-generate days 01–31
time = [f"{h:02d}:00" for h in range(24)] # Auto-generate hours 00:00–23:00
pressure_level = ["200"] # 200 hPa
area = [60, -80, 40, 10] # North, West, South, East (lat/lon)
d = 1

# Define output file name
if len(day) == 2:
    target = f"weather_data_{day}_{month}_{year}.grib"
    output_csv = f"north_atlantic_temperature_{day}_{month}_{year}.csv"
else:
    target = f"weather_data_{month}_{year}.grib"
    output_csv = f"north_atlantic_temperature_{month}_{year}.csv"

if Flag_Fetched_data:
    all_functions.fetch_data(target, year, month, day, time, pressure_level, area)

# === Turning the GRIB file to CSV ===
if Flag_grib_to_csv:
    file_path = f"{Project_location}{target}"
    t, w_pa = all_functions.grib_to_csv_with_w(file_path, output_csv, save_csv)

# === plotting ===
if Flag_plot:
    all_functions.create_map_plot(w_pa)
