import xarray as xr
import pandas as pd

# ===== 用户参数 =====
GRIB_FILE = "b8cfd82fa64c733fc1f115f884a33fbc.grib"
OUT_CSV = "windfield_northatlantic_20251007.csv"

LAT_MIN, LAT_MAX = 40, 60
LON_MIN, LON_MAX = -80, 0   # 如果GRIB是0~360，会自动处理
# ====================

# 读取GRIB
ds = xr.open_dataset(
    GRIB_FILE,
    engine="cfgrib",
    backend_kwargs={"indexpath": ""}
)

# 如果经度是 0~360，需要转为 -180~180
if ds.longitude.max() > 200:
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

# 裁剪经纬度范围
ds_subset = ds.where(
    (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) &
    (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX),
    drop=True
)

# 转为DataFrame并导出为CSV
df = ds_subset.to_dataframe().reset_index()
df.to_csv(OUT_CSV, index=False)

print("✅ DONE!")
print(f"裁剪范围: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
print(f"输出文件: {OUT_CSV}")
print("Data preview:\n", df.head())
