"""
estimate_fuel_b77w_openap_v2_4.py
---------------------------------
Fuel burn estimation for Boeing 777-300ER (OpenAP v2.4)
"""

import pandas as pd
import numpy as np
from openap import prop, fuel

INPUT_FILE = "flight_track_JFK_to_EGLL 0706.xlsx"
OUTPUT_FILE = "fuel_estimate_openap_b77w_v24.csv"
AIRCRAFT = "B77W"


# ========== ISA Model ==========
def isa_atmosphere(alt_m):
    T0, p0, L, R, g, gamma = 288.15, 101325.0, 0.0065, 287.05, 9.80665, 1.4
    if alt_m < 11000:
        T = T0 - L * alt_m
        p = p0 * (1 - L * alt_m / T0) ** (g / (R * L))
    else:
        T = 216.65
        p = 22632.06 * np.exp(-g * (alt_m - 11000) / (R * T))
    rho = p / (R * T)
    a = np.sqrt(gamma * R * T)
    return p, T, rho, a


# ========== Load Input ==========
df = pd.read_excel(INPUT_FILE)
cols = [c.lower() for c in df.columns]
df.columns = cols

rename_map = {}
for c in cols:
    if "lat" in c: rename_map[c] = "lat"
    if "lon" in c or "long" in c: rename_map[c] = "lon"
    if "alt" in c: rename_map[c] = "alt"
    if "time" in c or "stamp" in c: rename_map[c] = "time"
df = df.rename(columns=rename_map)

if df["alt"].median() > 2000:
    df["alt"] *= 0.3048
if not np.issubdtype(df["time"].dtype, np.number):
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
df = df.sort_values("time").reset_index(drop=True)

# ========== Initialize OpenAP ==========
fuel_model = fuel.FuelFlow(AIRCRAFT)

# ========== Estimate ==========
fuel_total = 0
results = []

for i in range(1, len(df)):
    row0, row1 = df.iloc[i - 1], df.iloc[i]
    dt = row1["time"] - row0["time"]
    if dt <= 0:
        continue

    alt = (row0["alt"] + row1["alt"]) / 2
    vs = (row1["alt"] - row0["alt"]) / dt
    p, T, rho, a = isa_atmosphere(alt)

    # simple mach estimate
    tas = 230 if alt < 10000 else 250 + (alt - 10000) / 1000 * 2
    mach = tas / a

    # --- determine phase manually ---
    if vs > 1.0 and alt < 11000:
        current_phase = "climb"
    elif vs < -1.0 and alt > 1000:
        current_phase = "descent"
    else:
        current_phase = "cruise"

    # --- fuel flow from OpenAP v2.4 ---
    ff = fuel_model.enroute(mass=mach, alt=alt, tas=230)
    seg_fuel = ff * dt
    fuel_total += seg_fuel

    results.append({
        "seg": i,
        "alt_m": alt,
        "mach": mach,
        "phase": current_phase,
        "fuel_flow_kg_s": ff,
        "fuel_seg_kg": seg_fuel,
        "fuel_total_kg": fuel_total
    })

out = pd.DataFrame(results)
out.to_csv(OUTPUT_FILE)
print(f"✅ Estimated total fuel burn (B77W, OpenAP v2.4): {fuel_total:.1f} kg")
print(f"Results saved to {OUTPUT_FILE}")
