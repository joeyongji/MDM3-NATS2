import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from heapq import heappush, heappop
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import math
import random

# =========================
# 读取真实风场数据（Excel）
# =========================
excel_path = "/Users/ivy/Desktop/MDM3-NATS2/windfield_20250725.csv"  # 如果路径不同请改这里
df = pd.read_csv(excel_path)

# df 需要包含列：
# 'isobaricInhPa', 'latitude', 'longitude', 'u', 'v'
# 其中 u,v 当前是 m/s

# 先整理经纬网格
lats_sorted = np.sort(df["latitude"].unique())     # shape Ny
lons_sorted = np.sort(df["longitude"].unique())    # shape Nx
pressures   = [225, 250, 300, 350]                 # hPa levels we care about

Ny = len(lats_sorted)
Nx = len(lons_sorted)

lat_to_i = {lat:i for i,lat in enumerate(lats_sorted)}
lon_to_i = {lon:i for i,lon in enumerate(lons_sorted)}

# wind_grids[p] = (U_kmh, V_kmh) for that pressure level
# 注意：我们在这里就把 m/s -> km/h，后面全程用 km/h
wind_grids = {}
for p in pressures:
    sub = df[df["isobaricInhPa"] == p]

    # 初始化
    U_kmh = np.full((Ny, Nx), np.nan, dtype=float)  # km/h
    V_kmh = np.full((Ny, Nx), np.nan, dtype=float)  # km/h

    for row in sub.itertuples(index=False):
        iy = lat_to_i[row.latitude]
        ix = lon_to_i[row.longitude]

        # row.u, row.v 是 m/s
        # 1 m/s = 3.6 km/h
        U_kmh[iy, ix] = row.u * 3.6
        V_kmh[iy, ix] = row.v * 3.6

    # 缺测点用0补，避免 NaN 让算法崩
    U_kmh = np.nan_to_num(U_kmh, nan=0.0)
    V_kmh = np.nan_to_num(V_kmh, nan=0.0)

    wind_grids[p] = (U_kmh, V_kmh)  # 之后一律按 km/h 用


# =========================
# 高度层（飞行高度，离散每1000 ft）
# =========================
alt_vals = np.arange(26000.0, 37000.0 + 1, 1000.0)  # [26000,27000,...,37000]
Nz = len(alt_vals)

def idx_to_lon(ix): return lons_sorted[ix]
def idx_to_lat(iy): return lats_sorted[iy]
def idx_to_alt(iz): return alt_vals[iz]

def lon_to_idx(lon_target):
    return int(np.argmin(np.abs(lons_sorted - lon_target)))

def lat_to_idx_func(lat_target):
    return int(np.argmin(np.abs(lats_sorted - lat_target)))

def alt_to_idx(alt_ft_target):
    return int(np.argmin(np.abs(alt_vals - alt_ft_target)))


# =========================
# 高度(ft) -> 使用哪一层等压面风 (hPa)
# =========================
def pressure_for_alt_ft(alt_ft):
    # 37000/36000/35000 ft → 225 hPa
    # 34000/33000/32000 ft → 250 hPa
    # 31000/30000/29000 ft → 300 hPa
    # 28000/27000/26000 ft → 350 hPa
    if alt_ft >= 35000.0:
        return 225
    elif alt_ft >= 32000.0:
        return 250
    elif alt_ft >= 29000.0:
        return 300
    else:
        return 350


# =========================
# 从风场取 (u,v) @ (ix,iy,高度层)，单位 km/h
# =========================
def get_wind_uv_kmh(ix, iy, alt_ft):
    """
    输出:
        u_kmh: 向东为正的水平风速 (km/h)
        v_kmh: 向北为正的水平风速 (km/h)
    逻辑:
        1. 根据高度(ft)确定使用的等压面
        2. 从对应等压面风场提取该经纬格点的 u,v
        3. 这些 u,v 已经是 km/h
    """
    p = pressure_for_alt_ft(alt_ft)
    U_kmh, V_kmh = wind_grids[p]  # [Ny, Nx], km/h
    return U_kmh[iy, ix], V_kmh[iy, ix]


# =========================
# 禁飞区（球形泡泡，索引空间）
# =========================
def generate_spherical_mask_index(Nx, Ny, Nz):
    """
    仿照原始定义方式（欧几里得球），但改为手动控制中心与半径
    """
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []

    # 手动设定中心（靠航线中段）
    cx = int(Nx * 0.45)
    cy = int(Ny * 0.7)
    cz = int(Nz * 0.6)
    r  = 4   # 半径

    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                if (ix - cx)**2 + (iy - cy)**2 + (iz - cz) **2 <= r**2:
                    mask[iz, iy, ix] = True

    spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx

blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)


# =========================
# 网格邻居（26邻接）
# =========================
moves = [(dx,dy,dz) for dx in (-1,0,1)
                  for dy in (-1,0,1)
                  for dz in (-1,0,1)
                  if not (dx==0 and dy==0 and dz==0)]

def in_bounds(ix,iy,iz):
    return (0 <= ix < Nx) and (0 <= iy < Ny) and (0 <= iz < Nz)

def passable(ix,iy,iz,mask):
    return not mask[iz,iy,ix]


# =========================
# 距离计算（全部用 km）
# =========================
DEG2KM_LAT = 111.0     # 每度纬度 ≈111 km
FT_TO_KM   = 0.0003048 # ft -> km

def east_north_km_between(ix1,iy1, ix2,iy2):
    """
    两个经纬格点间的水平位移分量 (east_km, north_km)
    east_km >0 表示向东, north_km >0 表示向北
    """
    lon1, lat1 = idx_to_lon(ix1), idx_to_lat(iy1)
    lon2, lat2 = idx_to_lon(ix2), idx_to_lat(iy2)

    lat_mid = math.radians(0.5*(lat1+lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    east_km  = dlon * DEG2KM_LAT * math.cos(lat_mid)
    north_km = dlat * DEG2KM_LAT
    return east_km, north_km

def vert_km_between(iz1, iz2):
    """
    垂直位移（km），由飞行高度(英尺)差换算
    """
    z1_km = idx_to_alt(iz1) * FT_TO_KM
    z2_km = idx_to_alt(iz2) * FT_TO_KM
    return abs(z2_km - z1_km)

def phys3d_km_between(a,b):
    """
    三维直线距离 (km)
    """
    ix1,iy1,iz1 = a
    ix2,iy2,iz2 = b
    east_km, north_km = east_north_km_between(ix1,iy1, ix2,iy2)
    vert_km          = vert_km_between(iz1,iz2)
    horiz_km = math.hypot(east_km, north_km)
    dist_km  = math.hypot(horiz_km, vert_km)
    return dist_km


# =========================
# 飞机地速建模 & A* 核心
# =========================
TAS = 900.0  # km/h，飞机巡航真空速近似
VMIN = 50.0  # km/h，地速下限，避免出现0或反向

def neighbors(ix,iy,iz,mask):
    """
    给出当前节点的所有可达相邻节点，以及：
      dist_km   : 3D距离 (km)
      east_km   : 水平方向东西分量 (km)
      north_km  : 水平方向南北分量 (km)
    """
    for dx,dy,dz in moves:
        nx,ny,nz = ix+dx, iy+dy, iz+dz
        if in_bounds(nx,ny,nz) and passable(nx,ny,nz,mask):
            dist_km = phys3d_km_between((ix,iy,iz),(nx,ny,nz))
            east_km, north_km = east_north_km_between(ix,iy, nx,ny)
            yield nx,ny,nz, dist_km, east_km, north_km

def heuristic(a,b):
    """
    A* 启发式：
    用极小比例的直线距离来保证乐观（不会高估真实代价）
    单位上无所谓，只要保证不高估
    """
    return phys3d_km_between(a,b) * 0.01

def astar_3d(start, goal, mask):
    """
    cost 累积规则：
        edge_cost = 0.5 * dist_km  + 0.5 * t_edge_hr
    其中
        dist_km 是边长 (km)
        t_edge_hr = dist_km / Vg
        Vg = 地速 (km/h)
    地速建模：
        Vg = TAS + 风在当前段的水平方向投影 (km/h)
        最低限速 VMIN 保护
    """

    openh = []
    heappush(openh, (0.0, start))

    g_cost   = {start: 0.0}  # 累积代价(无单位，混合指标)
    g_distkm = {start: 0.0}  # 累积飞行距离(km)
    parent   = {start: None}
    closed   = set()

    EPS = 1e-6

    while openh:
        _, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)

        # 到终点就回溯
        if u == goal:
            path=[]
            cur=u
            while cur is not None:
                path.append(cur)
                cur=parent[cur]
            path.reverse()
            return path, g_cost[u], g_distkm[u], closed

        ux,uy,uz = u
        alt_ft_here = idx_to_alt(uz)

        # 当前格点的风速 (km/h)
        u_wind_kmh, v_wind_kmh = get_wind_uv_kmh(ux, uy, alt_ft_here)

        for nx,ny,nz, dist_km, east_km, north_km in neighbors(ux,uy,uz,mask):
            v = (nx,ny,nz)

            # 算水平方向的单位向量，用来投影风
            horiz = math.hypot(east_km, north_km)

            if horiz < EPS:
                # 几乎纯爬升/下降，给它地速≈TAS
                Vg = TAS
            else:
                ex = east_km  / horiz  # 朝向的东西分量单位向量
                ey = north_km / horiz  # 朝向的南北分量单位向量
                # 风对该飞行方向的贡献 (km/h)
                Vg = TAS + (u_wind_kmh * ex + v_wind_kmh * ey)

            # 不允许地速太低或负
            if Vg < VMIN:
                Vg = VMIN

            # 飞这一小段需要的时间 (小时)
            t_edge_hr = dist_km / Vg

            # 边的代价（我们定义的混合指标）
            edge_cost = 0.5 * dist_km + 0.5 * t_edge_hr

            new_g = g_cost[u] + edge_cost
            if (v not in g_cost) or (new_g < g_cost[v]):
                g_cost[v]   = new_g
                g_distkm[v] = g_distkm[u] + dist_km
                parent[v]   = u
                heappush(openh, ( new_g + heuristic(v,goal), v ))

    return None, np.inf, np.inf, closed


# =========================
# 固定起点 / 终点
# =========================
# 起点: -73W, 41.5N
# 终点:  -3W, 53.5N
# 先假设两边都在巡航高度 ~36000ft
start_lon_target = -73.0
start_lat_target =  41.5
goal_lon_target  =  -3.0
goal_lat_target  =  53.5
cruise_alt_ft    = 36000.0

start_ix = lon_to_idx(start_lon_target)
start_iy = lat_to_idx_func(start_lat_target)
goal_ix  = lon_to_idx(goal_lon_target)
goal_iy  = lat_to_idx_func(goal_lat_target)

base_iz  = alt_to_idx(cruise_alt_ft)

def find_valid_iz(ix, iy, iz_guess, mask):
    """
    起点/终点如果在原定高度层是禁飞，就往附近高度层找最近的可用层
    """
    layers = list(range(Nz))
    layers.sort(key=lambda z: abs(z - iz_guess))
    for z in layers:
        if passable(ix, iy, z, mask):
            return z
    raise RuntimeError("All altitude layers blocked at this lon/lat")

start_iz = find_valid_iz(start_ix, start_iy, base_iz, blocked)
goal_iz  = find_valid_iz(goal_ix,  goal_iy,  base_iz, blocked)

start_point = (start_ix, start_iy, start_iz)
goal_point  = (goal_ix,  goal_iy,  goal_iz)


# =========================
# A* 路径 (有禁飞区 vs 无禁飞区)
# =========================
path_obs,  cost_obs,  dist_obs,  explored = astar_3d(start_point, goal_point, blocked)

empty_mask = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, dist_free, _ = astar_3d(start_point, goal_point, empty_mask)


# =========================
# 可视化
# =========================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')


ax.set_title(
    "Baseline Model: 3D A* Path under Static Wind Field (02:00 UTC)\n"
    "good weather condition"
)

# 画禁飞泡泡（球体）
# 画禁飞泡泡（球体）
u_mesh = np.linspace(0, np.pi, 30)
v_mesh = np.linspace(0, 2*np.pi, 60)
uu, vv = np.meshgrid(u_mesh, v_mesh)

for cx, cy, cz, r in spheres_idx:
    Xs = cx + r * np.sin(uu) * np.cos(vv)
    Ys = cy + r * np.sin(uu) * np.sin(vv)
    Zs = cz + r * np.cos(uu)

    # 将浮点索引转成最近格点索引后映射到物理经纬/高度
    Xs_i = np.clip(np.rint(Xs).astype(int), 0, Nx-1)
    Ys_i = np.clip(np.rint(Ys).astype(int), 0, Ny-1)
    Zs_i = np.clip(np.rint(Zs).astype(int), 0, Nz-1)

    # 将网格点映射到物理坐标（经纬度+高度）
    Xlon = idx_to_lon(Xs_i)
    Ylat = idx_to_lat(Ys_i)
    Zalt = idx_to_alt(Zs_i)

    # 绘制球体表面（增强立体感）
    ax.plot_surface(
        Xlon, Ylat, Zalt,
        alpha = 0.28, linewidth = 0, cmap = 'Reds'
    )


def path_lon_lat_alt(path):
    xs = [idx_to_lon(i) for (i,j,k) in path]
    ys = [idx_to_lat(j) for (i,j,k) in path]
    zs = [idx_to_alt(k) for (i,j,k) in path]
    return xs, ys, zs

# 无禁飞路线（baseline）
if path_free:
    xF, yF, zF = path_lon_lat_alt(path_free)
    ax.plot(
        xF, yF, zF,
        linestyle='--', color='black', linewidth=2,
        marker='o', markersize=3,
        markevery=max(1, len(xF)//12),
        label=f"Free-space (cost={cost_free:.2f}, dist={dist_free:.1f} km)"
    )

# 有禁飞路线（实际结果）
if path_obs:
    xO, yO, zO = path_lon_lat_alt(path_obs)
    ax.plot(
        xO, yO, zO,
        linestyle='-', color='black', linewidth=3,
        marker='^', markersize=4,
        markevery=max(1, len(xO)//12),
        label=f"With bubbles (cost={cost_obs:.2f}, dist={dist_obs:.1f} km)"
    )

# 起点和终点
sx,sy,sz = start_point
gx,gy,gz = goal_point
ax.scatter(
    [idx_to_lon(sx)], [idx_to_lat(sy)], [idx_to_alt(sz)],
    c='green', s=60, depthshade=False,
    label='Start (-73W, 41.5N)'
)
ax.scatter(
    [idx_to_lon(gx)], [idx_to_lat(gy)], [idx_to_alt(gz)],
    c='purple', s=80, depthshade=False, marker='*',
    label='Goal (-3W, 53.5N)'
)

# 风矢量可视化（在一个巡航高度层）
viz_iz     = base_iz
viz_alt_ft = idx_to_alt(viz_iz)

# 把该高度层的风拿出来 (km/h)
U_kmh = np.zeros((Ny, Nx), dtype=float)
V_kmh = np.zeros((Ny, Nx), dtype=float)
for iy in range(Ny):
    for ix in range(Nx):
        u_kmh, v_kmh = get_wind_uv_kmh(ix, iy, viz_alt_ft)
        U_kmh[iy, ix] = u_kmh
        V_kmh[iy, ix] = v_kmh

LON, LAT = np.meshgrid(lons_sorted, lats_sorted)  # (Ny,Nx)
Zplane = np.full((Ny, Nx), viz_alt_ft)

# 为了在经纬度坐标里显示箭头，把 km/h 转成 "度/小时"
DEG2KM_LAT = 111.0
LAT_RAD = np.radians(LAT)
U_degph = U_kmh / (DEG2KM_LAT * np.clip(np.cos(LAT_RAD), 1e-6, None))
V_degph = V_kmh /  DEG2KM_LAT

# 按东西向风速u_kmh上色：红=向东，蓝=向西
cmap = plt.get_cmap('bwr')
u_abs_max = max(1.0, float(np.max(np.abs(U_kmh))))
norm = colors.Normalize(vmin=-u_abs_max, vmax=+u_abs_max)

step = 8  # 稀疏画，避免太密
arrow_colors = cmap(norm(U_kmh[::step, ::step].ravel()))

# 半透明风平面
ax.plot_surface(
    LON, LAT, Zplane,
    rstride=Ny-1, cstride=Nx-1,
    color=(0.6,0.6,0.6,0.2),
    linewidth=0, shade=False
)

ax.quiver(
    LON[::step, ::step],
    LAT[::step, ::step],
    Zplane[::step, ::step],
    U_degph[::step, ::step],
    V_degph[::step, ::step],
    0,
    length=2.0,
    normalize=False,
    colors=arrow_colors,
    alpha=0.9,
    linewidth=0.8
)

mappable = ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cb = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.08, shrink=0.8)
cb.set_label(f"Zonal wind u (km/h) @ ~{int(viz_alt_ft):,} ft")

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Altitude (ft)")
ax.legend(loc='upper left')


plt.tight_layout()
plt.show()

# =========================
# 总结输出
# =========================
if path_obs and path_free and dist_free > 0 and cost_free > 0:
    inc_dist = (dist_obs  - dist_free) / dist_free * 100.0
    inc_cost = (cost_obs  - cost_free) / cost_free * 100.0
    print(f"[Summary] Free-space : cost={cost_free:.2f}, dist={dist_free:.1f} km")
    print(f"          With bubbles: cost={cost_obs:.2f} ({inc_cost:+.1f}%), "
          f"dist={dist_obs:.1f} km ({inc_dist:+.1f}%)")
else:
    print("Warning: one of the paths not found; maybe bubbles block everything.")
