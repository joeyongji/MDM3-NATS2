import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from heapq import heappush, heappop
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import math, random

# =========================
# 固定随机种子：保证禁飞区与 baseline 完全一致、可重复
# =========================
np.random.seed(53)
random.seed(53)

# =========================
# 读取逐小时真实风场（CSV）
# 要有列: time, isobaricInhPa, latitude, longitude, u, v
# time 例如: 2025-07-25 04:00:00
# =========================
csv_path = "windfield_20250725.csv"
df = pd.read_csv(csv_path)
df["time"] = pd.to_datetime(df["time"])

# 仅保留 04:00–09:00 这6个整点
HOURS = [4, 5, 6, 7, 8, 9]
df = df[df["time"].dt.hour.isin(HOURS)].copy()

# u,v: m/s -> km/h（和baseline一致）
df["u_kmh"] = df["u"] * 3.6
df["v_kmh"] = df["v"] * 3.6

# 经纬网格（由真实风场给出）
lats_sorted = np.sort(df["latitude"].unique())
lons_sorted = np.sort(df["longitude"].unique())
Ny, Nx = len(lats_sorted), len(lons_sorted)

lat_to_i = {lat: i for i, lat in enumerate(lats_sorted)}
lon_to_i = {lon: i for i, lon in enumerate(lons_sorted)}

# 高度层：26000ft 到 37000ft，每1000ft（和baseline一致）
alt_vals = np.arange(26000.0, 37000.0 + 1, 1000.0)
Nz = len(alt_vals)

def idx_to_lon(ix): return lons_sorted[ix]
def idx_to_lat(iy): return lats_sorted[iy]
def idx_to_alt(iz): return alt_vals[iz]

def lon_to_idx(lon_target):
    return int(np.clip(np.argmin(np.abs(lons_sorted - lon_target)), 0, Nx-1))

def lat_to_idx_func(lat_target):
    return int(np.clip(np.argmin(np.abs(lats_sorted - lat_target)), 0, Ny-1))

def alt_to_idx(alt_ft_target):
    return int(np.clip(np.argmin(np.abs(alt_vals - alt_ft_target)), 0, Nz-1))

# 压力层映射（和baseline一致）
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

# 把所有小时、所有等压面的风格点化到4维结构
pressures = [225, 250, 300, 350]
times_sorted = sorted(df["time"].unique())  # e.g. [04:00,05:00,...09:00]

U_all = {t: {} for t in times_sorted}  # U_all[time][pressure] = 2D array (Ny,Nx)
V_all = {t: {} for t in times_sorted}

for tstamp in times_sorted:
    dft = df[df["time"] == tstamp]
    for p in pressures:
        sub = dft[dft["isobaricInhPa"] == p]
        U = np.full((Ny, Nx), 0.0, dtype=float)
        V = np.full((Ny, Nx), 0.0, dtype=float)
        for row in sub.itertuples(index=False):
            iy = lat_to_i[row.latitude]
            ix = lon_to_i[row.longitude]
            U[iy, ix] = row.u_kmh
            V[iy, ix] = row.v_kmh
        U_all[tstamp][p] = U
        V_all[tstamp][p] = V

# 找全场最大风速，用来给启发式一个乐观的“最快地速上限”
max_wind_mag = 0.0
for tstamp in times_sorted:
    for p in pressures:
        wind_mag_here = np.hypot(U_all[tstamp][p], V_all[tstamp][p])
        max_wind_mag = max(max_wind_mag, float(np.nanmax(wind_mag_here)))

# =========================
# 常量
# =========================
TAS = 900.0           # km/h 真空速
VMIN = 50.0           # km/h 地速下限，避免负速
DEG2KM_LAT = 111.0
FT_TO_KM   = 0.0003048
BEST_POSSIBLE_GROUND_SPEED = TAS + max_wind_mag  # 用于A*启发式里的乐观时间估计

# =========================
# 禁飞区：随机球泡泡（保持 baseline 逻辑）
# =========================
def generate_spherical_mask_index(Nx, Ny, Nz, k_min=2, k_max=4):
    k = np.random.randint(k_min, k_max + 1)
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx = np.random.randint(Nx // 3, 2 * Nx // 3)
        cy = np.random.randint(Ny // 3, 2 * Ny // 3)
        cz = np.random.randint(1, Nz - 1)
        r  = np.random.randint(1, max(2, min(Nx, Ny) // 5))
        for iz in range(Nz):
            for iy in range(Ny):
                for ix in range(Nx):
                    if (ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2 <= r**2:
                        mask[iz, iy, ix] = True
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx

blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)

# =========================
# 网格 & 几何
# =========================
moves = [(dx,dy,dz)
         for dx in (-1,0,1)
         for dy in (-1,0,1)
         for dz in (-1,0,1)
         if not (dx==0 and dy==0 and dz==0)]

def in_bounds(ix,iy,iz):
    return (0 <= ix < Nx) and (0 <= iy < Ny) and (0 <= iz < Nz)

def passable(ix,iy,iz,mask):
    return not mask[iz,iy,ix]

def east_north_km_between(ix1,iy1, ix2,iy2):
    lon1, lat1 = idx_to_lon(ix1), idx_to_lat(iy1)
    lon2, lat2 = idx_to_lon(ix2), idx_to_lat(iy2)
    lat_mid = math.radians(0.5*(lat1+lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    east_km  = dlon * DEG2KM_LAT * math.cos(lat_mid)
    north_km = dlat * DEG2KM_LAT
    return east_km, north_km

def vert_km_between(iz1, iz2):
    z1_km = idx_to_alt(iz1) * FT_TO_KM
    z2_km = idx_to_alt(iz2) * FT_TO_KM
    return abs(z2_km - z1_km)

def phys3d_km_between(a,b):
    ix1,iy1,iz1 = a
    ix2,iy2,iz2 = b
    e_km, n_km = east_north_km_between(ix1,iy1, ix2,iy2)
    v_km       = vert_km_between(iz1,iz2)
    return math.hypot(math.hypot(e_km,n_km), v_km)

def neighbors(ix,iy,iz,mask):
    for dx,dy,dz in moves:
        nx,ny,nz = ix+dx, iy+dy, iz+dz
        if in_bounds(nx,ny,nz) and passable(nx,ny,nz,mask):
            dist_km = phys3d_km_between((ix,iy,iz),(nx,ny,nz))
            e_km, n_km = east_north_km_between(ix,iy, nx,ny)
            yield nx,ny,nz, dist_km, e_km, n_km

# =========================
# 风场采样：在给定时刻、给定高度层，取 (u,v) km/h
# =========================
def get_wind_uv_kmh(ix, iy, alt_ft, tstamp):
    p = pressure_for_alt_ft(alt_ft)
    U = U_all[tstamp][p]
    V = V_all[tstamp][p]
    return float(U[iy, ix]), float(V[iy, ix])

# =========================
# A* 搜索（改良版，快很多）
# - 使用强启发式 (distance + optimistic time)
# - 使用 closed 集合避免重复扩展
# =========================
def astar_3d(start, goal, mask, wD, wT, tstamp):
    def heuristic_weighted(a, b):
        dist = phys3d_km_between(a, b)
        time_est = dist / BEST_POSSIBLE_GROUND_SPEED
        return wD*dist + wT*time_est

    openh = []
    heappush(openh, (heuristic_weighted(start, goal), start))

    g_cost = {start: 0.0}   # 累积混合代价
    g_dist = {start: 0.0}   # 累积距离 km
    g_time = {start: 0.0}   # 累积时间 h
    parent = {start: None}

    closed = set()
    EPS = 1e-6

    while openh:
        _, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)

        if u == goal:
            # 回溯路径
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1], g_dist[u], g_time[u]

        ux,uy,uz = u
        alt_ft_here = idx_to_alt(uz)
        u_wind_kmh, v_wind_kmh = get_wind_uv_kmh(ux, uy, alt_ft_here, tstamp)

        for nx,ny,nz, dist_km, east_km, north_km in neighbors(ux,uy,uz,mask):
            v = (nx,ny,nz)

            horiz = math.hypot(east_km, north_km)
            if horiz < EPS:
                Vg = TAS
            else:
                ex = east_km / horiz
                ey = north_km / horiz
                Vg = TAS + (u_wind_kmh*ex + v_wind_kmh*ey)

            Vg = max(VMIN, Vg)
            t_edge = dist_km / Vg

            edge_cost = wD*dist_km + wT*t_edge
            new_g = g_cost[u] + edge_cost

            if (v not in g_cost) or (new_g < g_cost[v]):
                g_cost[v]  = new_g
                g_dist[v]  = g_dist[u] + dist_km
                g_time[v]  = g_time[u] + t_edge
                parent[v]  = u
                fscore = new_g + heuristic_weighted(v, goal)
                heappush(openh, (fscore, v))

    # 没找到路
    return None, np.inf, np.inf

# =========================
# Pareto–Knee：扫描 11 个权重组合并选折点
# =========================
def find_knee_point(points):
    """
    points: [(dist, time), ...] 按距离升序
    用“到起终连线的最大垂直距离”作为 knee
    """
    pts = np.array(points, dtype=float)
    a = pts[0]
    b = pts[-1]
    line = b - a
    norm_line = np.linalg.norm(line)
    if norm_line < 1e-12:
        return 0
    line_unit = line / norm_line
    vecs = pts - a
    proj = vecs @ line_unit
    foot = a + np.outer(proj, line_unit)
    dist = np.linalg.norm(pts - foot, axis=1)
    return int(np.argmax(dist))

def optimal_weights_for_hour(start_node, end_node, mask, tstamp):
    """
    返回 knee 选出来的 (wD, wT)
    """
    candidates = [(w/10.0, 1.0 - w/10.0) for w in range(11)]
    results = []
    for wD, wT in candidates:
        path, dist, tval = astar_3d(start_node, end_node, mask, wD, wT, tstamp)
        if path:
            results.append((wD, wT, dist, tval))

    if not results:
        return 0.5, 0.5

    results.sort(key=lambda r: r[2])  # 按距离升序
    pts = [(r[2], r[3]) for r in results]
    knee_i = find_knee_point(pts)
    best_wD, best_wT = results[knee_i][0], results[knee_i][1]
    return best_wD, best_wT

# =========================
# 固定起点 / 终点（和baseline一致）
# 起点: -73W, 41.5N
# 终点:  -3W, 53.5N
# =========================
start_lon, start_lat = -73.0, 41.5
goal_lon,  goal_lat  =  -3.0, 53.5
cruise_alt_ft = 36000.0

start_ix = lon_to_idx(start_lon)
start_iy = lat_to_idx_func(start_lat)
goal_ix  = lon_to_idx(goal_lon)
goal_iy  = lat_to_idx_func(goal_lat)
base_iz  = alt_to_idx(cruise_alt_ft)

def find_valid_iz(ix, iy, iz_guess, mask):
    # 如果该高度被禁飞，就找最近可用层
    layer_order = sorted(range(Nz), key=lambda z: abs(z - iz_guess))
    for z in layer_order:
        if passable(ix, iy, z, mask):
            return z
    raise RuntimeError("All altitude layers blocked at this lon/lat")

start_iz = find_valid_iz(start_ix, start_iy, base_iz, blocked)
goal_iz  = find_valid_iz(goal_ix,  goal_iy,  base_iz, blocked)

start_point = (start_ix, start_iy, start_iz)
goal_point  = (goal_ix,  goal_iy,  goal_iz)

print("Start grid:", start_point,
      "→", idx_to_lon(start_ix), idx_to_lat(start_iy), idx_to_alt(start_iz), "ft")
print("Goal  grid:", goal_point,
      "→", idx_to_lon(goal_ix),  idx_to_lat(goal_iy),  idx_to_alt(goal_iz),  "ft")

# =========================
# 动态重规划主循环（逐小时风场 + Pareto–Knee）
# 每个小时：
# 1. 根据该小时风场优化权重
# 2. 用权重跑A*
# 3. 按实际地速推进一小时飞行
# =========================
HOURLY_FLIGHT_DURATION = 1.0  # h
alpha = 0.6                   # 策略惯性平滑
current_pos = start_point
prev_wD, prev_wT = 0.5, 0.5
dynamic_segments = []    # 每小时真的飞的那一小段
weights_history  = []    # [(hour_idx, wD, wT)]

print("\n--- Dynamic Pareto–Knee Flight Simulation (Hourly real wind) ---")
for h_idx, tstamp in enumerate(times_sorted, start=1):
    if current_pos == goal_point:
        break

    print(f"\n[Hour {h_idx}] Wind @ {tstamp}  from {current_pos}")

    # 1) 针对这一小时风场，做 Pareto 扫描，找 knee 权重
    new_wD, new_wT = optimal_weights_for_hour(current_pos, goal_point, blocked, tstamp)

    # 2) 平滑策略
    wD = alpha*prev_wD + (1-alpha)*new_wD
    wT = 1.0 - wD
    prev_wD, prev_wT = wD, wT
    weights_history.append((h_idx, wD, wT))
    print(f"   → dynamic weights: wD={wD:.2f}, wT={wT:.2f}")

    # 3) 用平滑后权重在当前小时风下算整条A*路径
    full_path, _, _ = astar_3d(current_pos, goal_point, blocked, wD, wT, tstamp)
    if not full_path:
        print("   [WARN] No path found under current wind/weights.")
        break

    # 4) 沿路径推进1小时
    acc_time = 0.0
    cut_idx = 0
    for i in range(1, len(full_path)):
        p1 = full_path[i-1]
        p2 = full_path[i]
        dist_km = phys3d_km_between(p1,p2)

        # 以段起点的风，估出地速
        ux,uy,uz = p1
        alt_ft_here = idx_to_alt(uz)
        u_kmh, v_kmh = get_wind_uv_kmh(ux,uy,alt_ft_here,tstamp)
        east_km, north_km = east_north_km_between(p1[0],p1[1], p2[0],p2[1])
        horiz = math.hypot(east_km, north_km)

        if horiz < 1e-6:
            Vg = TAS
        else:
            ex = east_km / horiz
            ey = north_km / horiz
            Vg = TAS + (u_kmh*ex + v_kmh*ey)
        Vg = max(VMIN, Vg)

        acc_time += dist_km / Vg
        if acc_time >= HOURLY_FLIGHT_DURATION:
            cut_idx = i
            break

    hour_segment = full_path if cut_idx == 0 else full_path[:cut_idx+1]
    current_pos = hour_segment[-1]
    dynamic_segments.append(hour_segment)
    print(f"   ✔ moved to {current_pos}")

print("\n--- Simulation Complete ---\n")

# =========================
# 可视化：3D 航迹 + 风矢量平面 + baseline式图例
# =========================
def path_lon_lat_alt(path):
    xs = [idx_to_lon(i) for i,j,k in path]
    ys = [idx_to_lat(j) for i,j,k in path]
    zs = [idx_to_alt(k) for i,j,k in path]
    return xs, ys, zs

fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Dynamic 3D Route (Hourly Real Wind)\nPareto–Knee Per Hour")

# 画禁飞球
u_mesh = np.linspace(0, np.pi, 20)
v_mesh = np.linspace(0, 2*np.pi, 40)
uu, vv = np.meshgrid(u_mesh, v_mesh)
for cx,cy,cz,r in spheres_idx:
    Xs = cx + r*np.sin(uu)*np.cos(vv)
    Ys = cy + r*np.sin(uu)*np.sin(vv)
    Zs = cz + r*np.cos(uu)
    Xsi = np.clip(np.rint(Xs).astype(int), 0, Nx-1)
    Ysi = np.clip(np.rint(Ys).astype(int), 0, Ny-1)
    Zsi = np.clip(np.rint(Zs).astype(int), 0, Nz-1)
    ax.plot_surface(
        idx_to_lon(Xsi),
        idx_to_lat(Ysi),
        idx_to_alt(Zsi),
        alpha=0.25,
        cmap="Reds",
        linewidth=0
    )

# 每个小时飞行的段（不同颜色），图例给出真实时钟段
segment_colors = plt.cm.plasma(np.linspace(0, 1, max(1, len(dynamic_segments))))
for i, seg in enumerate(dynamic_segments):
    xs,ys,zs = path_lon_lat_alt(seg)
    label = f"{HOURS[0]+i}:00–{HOURS[0]+i+1}:00"
    ax.plot(xs, ys, zs, color=segment_colors[i], linewidth=3, label=label)

# 起点 / 终点
sx,sy,sz = start_point
gx,gy,gz = goal_point
ax.scatter([idx_to_lon(sx)],[idx_to_lat(sy)],[idx_to_alt(sz)],
           c='green', s=60, label='Start (-73W, 41.5N)')
ax.scatter([idx_to_lon(gx)],[idx_to_lat(gy)],[idx_to_alt(gz)],
           c='purple', s=80, marker='*', label='Goal (-3W, 53.5N)')

# 风矢量平面：用最后一次使用的小时（或第一小时）在 ~36000 ft
viz_alt_ft = 36000.0
viz_iz = alt_to_idx(viz_alt_ft)
if dynamic_segments:
    last_hour_used = min(len(dynamic_segments), len(times_sorted))
else:
    last_hour_used = 1
tstamp_viz = times_sorted[last_hour_used-1]

U_kmh_plane = np.zeros((Ny, Nx))
V_kmh_plane = np.zeros((Ny, Nx))
for iy in range(Ny):
    for ix in range(Nx):
        u_kmh_tmp, v_kmh_tmp = get_wind_uv_kmh(ix, iy, viz_alt_ft, tstamp_viz)
        U_kmh_plane[iy, ix] = u_kmh_tmp
        V_kmh_plane[iy, ix] = v_kmh_tmp

LON, LAT = np.meshgrid(lons_sorted, lats_sorted)
Zplane = np.full((Ny, Nx), viz_alt_ft)

LAT_RAD = np.radians(LAT)
U_degph = U_kmh_plane / (DEG2KM_LAT * np.clip(np.cos(LAT_RAD), 1e-6, None))
V_degph = V_kmh_plane /  DEG2KM_LAT

cmap = plt.get_cmap('bwr')
u_abs_max = max(1.0, float(np.max(np.abs(U_kmh_plane))))
norm = colors.Normalize(vmin=-u_abs_max, vmax=+u_abs_max)
step = max(1, min(Nx, Ny)//10)

ax.plot_surface(
    LON, LAT, Zplane,
    rstride=Ny-1, cstride=Nx-1,
    color=(0.85,0.85,0.85,0.25),
    linewidth=0,
    shade=False
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
    colors=cmap(norm(U_kmh_plane[::step, ::step].ravel())),
    alpha=0.9,
    linewidth=0.5
)

mappable = ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cb_quiv = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
cb_quiv.set_label(
    f"Zonal wind u (km/h) @ ~{int(viz_alt_ft):,} ft (hour={tstamp_viz.strftime('%H:%M')})"
)

# =========================
# baseline-style 图例: Free-space vs With bubbles
# =========================

# # 1) Free-space baseline：无禁飞区，用第1小时风场，用 (wD,wT)=(0.5,0.5)
# empty_mask = np.zeros_like(blocked, dtype=bool)
# b_path, b_dist, b_time = astar_3d(
#     start_point,
#     goal_point,
#     empty_mask,
#     0.5, 0.5,
#     times_sorted[0]  # 用第1小时的风
# )
# b_cost = 0.5*b_dist + 0.5*b_time
#
# if b_path:
#     bx, by, bz = path_lon_lat_alt(b_path)
#     ax.plot(
#         bx, by, bz,
#         linestyle='--', color='black', linewidth=2,
#         marker='o', markersize=3,
#         markevery=max(1, len(bx)//12),
#         label=f"Free-space (cost={b_cost:.2f}, dist={b_dist:.1f} km)"
#     )

# 2) Dynamic totals: 把整段动态飞的距离/代价累计起来
dyn_dist = 0.0
dyn_cost = 0.0
for seg_i, seg in enumerate(dynamic_segments):
    # 这一段对应的小时
    used_tstamp = times_sorted[min(seg_i, len(times_sorted)-1)]
    # 这一小时使用的权重（平滑后的）
    wD_i, wT_i = weights_history[min(seg_i, len(weights_history)-1)][1:3]

    for n in range(1, len(seg)):
        p1, p2 = seg[n-1], seg[n]
        d_km = phys3d_km_between(p1,p2)

        # 用段起点的风估时间
        ux,uy,uz = p1
        alt_ft_here = idx_to_alt(uz)
        u_kmh_here, v_kmh_here = get_wind_uv_kmh(ux,uy,alt_ft_here,used_tstamp)

        east_km, north_km = east_north_km_between(p1[0],p1[1], p2[0],p2[1])
        horiz = math.hypot(east_km, north_km)
        if horiz < 1e-6:
            Vg = TAS
        else:
            ex = east_km/horiz
            ey = north_km/horiz
            Vg = TAS + (u_kmh_here*ex + v_kmh_here*ey)
        Vg = max(VMIN, Vg)
        t_edge = d_km / Vg

        dyn_dist += d_km
        dyn_cost += (wD_i*d_km + wT_i*t_edge)

# 用一条“空曲线”只放进图例
ax.plot(
    [], [], color='black', linestyle='-', linewidth=3,
    label=f"With bubbles (cost={dyn_cost:.2f}, dist={dyn_dist:.1f} km)"
)

# =========================
# 轴标签 / 图例 / 展示
# =========================
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Altitude (ft)")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# =========================
# （可选）权重随时间演化图
# =========================
if weights_history:
    hrs  = [h for h,_,_ in weights_history]
    wDs  = [w for _,w,_ in weights_history]
    wTs  = [w for _,_,w in weights_history]

    plt.figure(figsize=(8,5))
    plt.plot(hrs, wDs, 'o-', label='wD (distance weight)')
    plt.plot(hrs, wTs, 's--', label='wT (time weight)')
    plt.xlabel("Hour index (1 = 04:00)")
    plt.ylabel("Weight")
    plt.title("Pareto–Knee Weights Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()
