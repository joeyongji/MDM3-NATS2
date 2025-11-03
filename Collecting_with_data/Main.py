import all_functions
#import ChatGPT_t
import os

Project_location = "/Users/oliverchadwick/PycharmProjects/MDM3_A_NATS/"
Flag_Fetched_data = 0
Flag_grib_to_csv = 1
Flag_plot = 0
save_csv = 1

# === Load the GRIB file ===
years = ["2025"]
months = [f"{d:02d}" for d in [7]]
days = [f"{d:02d}" for d in [1,5,10,15,20,25,30]] # Auto-generate days 01–31 range(1, 3)
time = [f"{h:02d}:00" for h in range(24)] # Auto-generate hours 00:00–23:00
pressure_level = [
        "1", "2", "3",
        "5", "7", "10",
        "20", "30", "50",
        "70", "100", "125",
        "150", "175", "200",
        "225", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000"
    ] # 200 hPa
area = [60, -80, 40, 10] # North, West, South, East (lat/lon)
d = 1
# Define output file name
if len(days) == 1:
    name_day = days[0]
else:
    name_day = f"{days[0]}_{days[-1]}"

if len(months) == 1:
    name_month = months[0]
else:
    name_month = f"{months[0]}_{months[-1]}"

if len(years) == 1:
    name_year = years[0]
else:
    name_year = f"{years[0]}_{years[-1]}"
# === Loop through years ===
for year in years:
    # === Loop through month ===
    for month in months:
        # === Loop through days ===
        for day in days:
            print(f"\n🗓️ Processing day {day}-{month}-{year}")

            # Define filenames
            target = f"weather_data_{name_day}_{name_month}_{name_year}.grib"
            output_dir = f"north_atlantic_weather_data_for_{year}/{month}"  # main monthly folder
            day_folder = os.path.join(output_dir)
            os.makedirs(day_folder, exist_ok=True)

            output_csv = os.path.join(day_folder, f"north_atlantic_temperature_{day}_{month}_{year}.csv")

            data_exists = os.path.exists(target)

            # === Fetch data (optional) ===
            if Flag_Fetched_data and not(data_exists):
                all_functions.fetch_data(target, year, month, day, time, pressure_level, area)

            # === Convert GRIB → CSV ===
            if Flag_grib_to_csv:
                file_path = os.path.join(Project_location, target)
                all_functions.grib_to_csv_with_w(file_path, output_csv, f"{year}-{month}-{day}", save_csv, chunk_by="time")

            # === Optional plotting ===
            if Flag_plot:
                all_functions.create_map_plot(w_pa)

print("\n✅ All days processed successfully.")
"""
# Define output file name
if len(days) == 2:
    target = f"weather_data_{days}_{month}_{year}.grib"
    output_csv = f"north_atlantic_temperature_{days}_{month}_{year}.csv"
else:
    target = f"weather_data_{month}_{year}.grib"
    output_csv = f"north_atlantic_temperature_{month}_{year}.csv"

if Flag_Fetched_data:
    all_functions.fetch_data(target, year, month, days, time, pressure_level, area)

# === Turning the GRIB file to CSV ===
if Flag_grib_to_csv:
    file_path = f"{Project_location}{target}"
    ChatGPT_t.grib_to_csv_with_w(file_path, output_csv, save_csv) #t, w_pa =

# === plotting ===
if Flag_plot:
    all_functions.create_map_plot(w_pa)
"""