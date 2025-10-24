import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === Step 1: Load and parse the KML ===
kml_file = "FlightAware_AAL104_KJFK_EGLL_20251007.kml"
tree = ET.parse(kml_file)
root = tree.getroot()

# Namespaces
ns = {
    'kml': 'http://www.opengis.net/kml/2.2',
    'gx': 'http://www.google.com/kml/ext/2.2'
}

# === Step 2: Find gx:Track blocks ===
coords = []
times = []

for track in root.findall('.//gx:Track', ns):
    these_times = [t.text for t in track.findall('kml:when', ns)]
    these_coords = []
    for c in track.findall('gx:coord', ns):
        lon, lat, alt = map(float, c.text.split())
        these_coords.append((lat, lon, alt))

    # Only keep pairs that match lengths
    n = min(len(these_times), len(these_coords))
    times.extend(these_times[:n])
    coords.extend(these_coords[:n])

# === Step 3: Build DataFrame ===
data = pd.DataFrame(coords, columns=['Latitude', 'Longitude', 'Altitude'])
data['Time'] = pd.to_datetime(times[:len(data)], errors='coerce')

# === Step 4: Plot flight path ===
plt.figure(figsize=(10, 6))
plt.plot(data['Longitude'], data['Latitude'], color='blue', linewidth=1)
plt.title("Flight Track: JFK → Heathrow AAL104 20251007")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# === Create the Cartopy map ===
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([280, 360, 0, 70])  # lon_min, lon_max, lat_min, lat_max

# Add map features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)

# === Step 5: Save to Excel ===
data['Time'] = data['Time'].dt.tz_localize(None)
data.to_excel("flight_track_JFK_to_EGLL 1007.xlsx")
print(f"Extracted {len(data)} track points and saved to Excel.")

