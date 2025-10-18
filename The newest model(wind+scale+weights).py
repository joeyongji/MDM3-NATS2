"""
Dynamic Pareto–Knee 3D Flight Planner
------------------------------------------
This script simulates a multi-hour flight through a dynamic 3D environment,
incorporating an evolving wind field and static no-fly zones. At each hour,
it performs a multi-objective optimization to determine the best flight
strategy for the remaining journey by balancing flight distance and time.

Key Features:
 - 3D A* pathfinding on a grid mapped to realistic geographic coordinates.
 - A high-fidelity physical model using kilometers and km/h for all calculations.
 - A dynamic wind field model with altitude-dependent layers and a latitudinal jet stream.
 - Hourly re-optimization: At each simulated hour, a Pareto frontier is generated
   by scanning a range of weights for the distance vs. time objectives.
 - Automatic decision-making: A "knee point" detection algorithm identifies the
   most balanced trade-off from the Pareto frontier to select the optimal weights.
 - Strategy smoothing: An inertia parameter (alpha) ensures that the flight strategy
   evolves gracefully over time, preventing erratic hourly changes.
 - Comprehensive Visualization: Produces a 3D plot of the final, dynamically
   evolved flight path with color-coded segments, and a 2D plot showing the
   evolution of the chosen weights over the flight duration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from heapq import heappush, heappop
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import math, random

# =========================
# Basic Simulation Parameters
# =========================
np.random.seed(53)
random.seed(53)
Nx, Ny, Nz = 30, 25, 5
TAS = 900.0  # km/h
HOURLY_FLIGHT_DURATION = 1.0
TOTAL_SIM_HOURS = 6

lon_min, lon_max = -70.0, -10.0
lat_min, lat_max = 40.0, 60.0
alt_min, alt_max = 30000.0, 38000.0

lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

def idx_to_lon(ix): return np.interp(ix, np.arange(Nx), lon_vals)
def idx_to_lat(iy): return np.interp(iy, np.arange(Ny), lat_vals)
def idx_to_alt(iz): return np.interp(iz, np.arange(Nz), alt_vals)

# =========================
# Environment Modeling
# =========================
def generate_spherical_mask_index(Nx, Ny, Nz, k_min=2, k_max=4):
    k = np.random.randint(k_min, k_max + 1)
    Z, Y, X = np.ogrid[:Nz, :Ny, :Nx]
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx,cy,cz = np.random.randint(Nx//3,2*Nx//3), np.random.randint(Ny//3,2*Ny//3), np.random.randint(1,Nz-1)
        r = np.random.randint(1, max(2, min(Nx, Ny)//5))
        mask |= (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 <= r**2
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx


# =========================
# 26 directions
# =========================
moves = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if not (dx==0 and dy==0 and dz==0)]
def in_bounds(ix,iy,iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix,iy,iz,mask): return not mask[iz,iy,ix]

DEG2KM_LAT = 111.0
FT_TO_KM = 0.0003048

def east_north_km_between(ix1,iy1, ix2,iy2):
    lon1, lat1, lon2, lat2 = idx_to_lon(ix1), idx_to_lat(iy1), idx_to_lon(ix2), idx_to_lat(iy2)
    lat_mid = math.radians(0.5*(lat1+lat2))
    return (lon2-lon1)*DEG2KM_LAT*math.cos(lat_mid), (lat2-lat1)*DEG2KM_LAT

def vert_km_between(iz1, iz2): return abs(idx_to_alt(iz1) - idx_to_alt(iz2)) * FT_TO_KM

def phys3d_km_between(a,b):
    ex,ny = east_north_km_between(a[0],a[1], b[0],b[1])
    vk = vert_km_between(a[2],b[2])
    return math.hypot(math.hypot(ex,ny), vk)

# =========================
# 风场（静态，水平风）：层风 + 纬向急流条带
# =========================
# 每层基础风（km/h）——把某些层改为负数就能看到向西的箭头
u_layer = np.array([-20.,-40.,+80.,+40.,-10.]) # 东西向（+东，-西）
v_layer = np.array([0.,-10.,0.,+10.,0.]) # 南北向（+北，-南）
A_JET, Y0_JET, SIGMA_JET = -150.0, (Ny - 1)/2.0, Ny/6.0
def u_jet(iy): return A_JET * math.exp(-((iy - Y0_JET)**2)/(2*SIGMA_JET**2))

def wind_uv(ix, iy, iz, t_hour=0):
    """动态风场：随时间演变"""
    base_u = float(u_layer[iz]) + u_jet(iy)
    base_v = float(v_layer[iz])
    delta_u = 30.0 * math.sin(0.5 * t_hour + iy / Ny * np.pi)
    delta_v = 15.0 * math.cos(0.3 * t_hour + ix / Nx * np.pi)
    return base_u + delta_u, base_v + delta_v

max_wind_mag = max(math.hypot(u_layer[z]+A_JET, v_layer[z]) for z in range(Nz))
BEST_POSSIBLE_GROUND_SPEED = TAS + max_wind_mag

def neighbors(ix,iy,iz,mask):
    for dx,dy,dz in moves:
        nx,ny,nz = ix+dx, iy+dy, iz+dz
        if in_bounds(nx,ny,nz) and passable(nx,ny,nz,mask):
            dist_km = phys3d_km_between((ix,iy,iz),(nx,ny,nz))
            east_km, north_km = east_north_km_between(ix,iy, nx,ny)
            yield nx,ny,nz, dist_km, east_km, north_km


# =========================
# 起终点
# =========================
def random_free_cell(mask, margin=1, min_dist_km=1000.0):
    for _ in range(20000):
        ix,iy,iz = np.random.randint(margin,Nx-margin), np.random.randint(margin,Ny-margin), np.random.randint(0,Nz)
        if not passable(ix,iy,iz,mask): continue
        for _ in range(2000):
            jx,jy,jz = np.random.randint(margin,Nx-margin), np.random.randint(margin,Ny-margin), np.random.randint(0,Nz)
            if passable(jx,jy,jz,mask) and phys3d_km_between((ix,iy,iz),(jx,jy,jz)) >= min_dist_km:
                return (ix,iy,iz), (jx,jy,jz)
    raise RuntimeError("No valid start-goal pair found.")

# =========================
# A* algorithm
# =========================
def astar_3d(start, goal, mask, w_d, w_t, t_hour=0):
    def heuristic_weighted(a,b):
        dist = phys3d_km_between(a,b)
        time_est = dist / BEST_POSSIBLE_GROUND_SPEED
        return w_d * dist + w_t * time_est
    openh, closed = [], set()
    heappush(openh, (heuristic_weighted(start, goal), start))
    g, g_dist, g_time, parent = {start:0.0}, {start:0.0}, {start:0.0}, {start:None}

    EPS, VMIN = 1e-6, 50.0

    while openh:
        _, u = heappop(openh)
        if u in closed: continue
        closed.add(u)
        if u == goal:
            path=[]; cur=u
            while cur is not None: path.append(cur); cur=parent[cur]
            return path[::-1], g_dist[u], g_time[u]
        ux,uy,uz = u
        u_wind, v_wind = wind_uv(ux,uy,uz,t_hour)
        for nx,ny,nz, dist_km, east_km, north_km in neighbors(ux,uy,uz,mask):
            v = (nx,ny,nz)
            horiz = math.hypot(east_km, north_km)
            Vg = TAS if horiz < EPS else TAS + (u_wind*(east_km/horiz) + v_wind*(north_km/horiz))
            Vg = max(VMIN, Vg)
            t_edge = dist_km / Vg
            edge_cost = w_d*dist_km + w_t*t_edge
            new_g = g[u] + edge_cost
            if v not in g or new_g < g[v]:
                g[v]=new_g; g_dist[v]=g_dist[u]+dist_km; g_time[v]=g_time[u]+t_edge; parent[v]=u
                heappush(openh, (new_g + heuristic_weighted(v,goal), v))
    return None, np.inf, np.inf

# =========================
# Pareto + Knee point
# =========================
def find_knee_point(points):
    """Finds the knee point on a Pareto curve using the distance-to-line method."""
    pts=np.array(points)
    minv,maxv=pts.min(axis=0),pts.max(axis=0)
    rng=maxv-minv
    if np.any(rng<1e-6): return 0
    norm=(pts-minv)/rng
    line_vec=norm[-1]-norm[0]
    line_vec_norm=line_vec/np.linalg.norm(line_vec)
    vecs=norm-norm[0]
    proj=vecs.dot(line_vec_norm)
    parallel=proj[:,np.newaxis]*line_vec_norm
    to_line=vecs-parallel
    dist=np.linalg.norm(to_line,axis=1)
    return np.argmax(dist)

def find_optimal_weights_for_leg(start_node, end_node, mask, t_hour=0):
    """
        Performs a Pareto scan for a given flight leg and returns the optimal weights.
    """
    weights_to_test = [(w / 10.0, 1.0 - w / 10.0) for w in range(11)]
    results = []
    for wD, wT in weights_to_test:
        path, dist, time = astar_3d(start_node, end_node, mask, wD, wT, t_hour)
        if path: results.append({'wD': wD, 'wT': wT, 'dist': dist, 'time': time})
    if not results: return (0.5, 0.5, [])
    results.sort(key=lambda r: r['dist'])
    pareto_points = np.array([[r['dist'], r['time']] for r in results])
    knee_idx = find_knee_point(pareto_points)
    return (results[knee_idx]['wD'], results[knee_idx]['wT'], results)

# =========================
# Main Simulation Loop
# =========================
blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)
start_point, goal_point = random_free_cell(blocked, min_dist_km=1000.0)

dynamic_segments, current_pos, hour = [], start_point, 0
weights_history = []
all_pareto_data = {}
prev_wD, prev_wT = 0.5, 0.5
alpha = 0.6  # Inertia factor for weight smoothing
"""
An alpha of 0.6 means that:
the aircraft's strategy over the next hour will be a mixture of 60% of the old strategy and 40% of the new "ideal" strategy.
Significance: This simulates real-world operational stability. 
A system won't make a sudden 180-degree policy change in response to a slight change in the environment. 
This "decision inertia" ensures smooth and gradual evolution of flight strategy, 
avoiding drastic and unstable changes in flight path caused by short-term forecast fluctuations.
"""


print("\n--- Dynamic Pareto–Knee Flight Simulation ---")
while hour < TOTAL_SIM_HOURS and current_pos != goal_point:
    hour += 1
    print(f"\n[Hour {hour}] Re-planning from {current_pos}...")
    new_wD, new_wT, results = find_optimal_weights_for_leg(current_pos, goal_point, blocked, t_hour=hour)
    wD = alpha*prev_wD + (1-alpha)*new_wD
    wT = 1 - wD
    prev_wD, prev_wT = wD, wT
    weights_history.append((hour, wD, wT))
    all_pareto_data[hour] = {'results': results, 'knee': (new_wD, new_wT)}
    print(f"  → Dynamic weights: (wD={wD:.2f}, wT={wT:.2f})")
    full_path, _, _ = astar_3d(current_pos, goal_point, blocked, wD, wT, t_hour=hour)
    if not full_path:
        print("  [WARN] No path found at this step.")
        break
    accumulated_time, next_idx = 0.0, 0
    for i in range(1, len(full_path)):
        p1,p2 = full_path[i-1], full_path[i]
        dist = phys3d_km_between(p1,p2)
        u_w,v_w = wind_uv(p1[0],p1[1],p1[2],t_hour=hour)
        east_km,north_km = east_north_km_between(p1[0],p1[1],p2[0],p2[1])
        horiz = math.hypot(east_km,north_km)
        vg = max(50.0, TAS + (u_w*(east_km/horiz if horiz>1e-6 else 0)+(v_w*(north_km/horiz if horiz>1e-6 else 0))))
        accumulated_time += dist/vg
        if accumulated_time >= HOURLY_FLIGHT_DURATION:
            next_idx = i
            break
    segment = full_path if next_idx==0 else full_path[:next_idx+1]
    current_pos = segment[-1]
    dynamic_segments.append(segment)
    print(f"  ✔ Segment with {len(segment)} waypoints completed → new position {current_pos}")

print("\n--- Simulation Complete ---")

# =========================
# 可视化：航迹 + 风场 + 权重演化
# =========================
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
# =========================
# 辅助函数：将索引路径转换为真实经纬高坐标
# =========================
def path_lon_lat_alt(path):
    xs = [idx_to_lon(i) for i, j, k in path]
    ys = [idx_to_lat(j) for i, j, k in path]
    zs = [idx_to_alt(k) for i, j, k in path]
    return xs, ys, zs


# 禁飞区
u_mesh,v_mesh = np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,40)
uu,vv=np.meshgrid(u_mesh,v_mesh)
for cx,cy,cz,r in spheres_idx:
    Xs,Ys,Zs = cx+r*np.sin(uu)*np.cos(vv), cy+r*np.sin(uu)*np.sin(vv), cz+r*np.cos(uu)
    ax.plot_surface(idx_to_lon(Xs),idx_to_lat(Ys),idx_to_alt(Zs),alpha=0.25,color='r',linewidth=0)

def path_to_phys(path):
    return [idx_to_lon(i) for i,j,k in path],[idx_to_lat(j) for i,j,k in path],[idx_to_alt(k) for i,j,k in path]


# --- 动态航迹（自动图例标签） ---
segment_colors = plt.cm.plasma(np.linspace(0, 1, len(dynamic_segments)))

for i, seg in enumerate(dynamic_segments):
    xs, ys, zs = path_lon_lat_alt(seg)
    label = f"Hour {i + 1}"  # ✅ 每个小时都有自己的图例标签
    ax.plot(xs, ys, zs, color=segment_colors[i], linewidth=3.2, label=label)


sx,sy,sz=start_point; gx,gy,gz=goal_point
ax.scatter([idx_to_lon(sx)],[idx_to_lat(sy)],[idx_to_alt(sz)],c='lime',s=100,marker='o',label='Start')
ax.scatter([idx_to_lon(gx)],[idx_to_lat(gy)],[idx_to_alt(gz)],c='purple',s=150,marker='*',label='Goal')

# 风矢量层
z_view=Nz//2
z_view_alt_ft=idx_to_alt(z_view)

LON,LAT=np.meshgrid(lon_vals,lat_vals)
U,V=np.zeros((Ny,Nx)),np.zeros((Ny,Nx))
for iy in range(Ny):
    for ix in range(Nx):
        U[iy,ix],V[iy,ix]=wind_uv(ix,iy,z_view,t_hour=hour)

Zplane = np.full_like(LON, z_view_alt_ft)
ax.plot_surface(LON, LAT, Zplane, rstride=Ny-1, cstride=Nx-1,
                color=(0.85, 0.85, 0.85, 0.25), linewidth=0, shade=False)

# km/h -> 度/小时（仅用于可视化尺度）
LAT_RAD = np.radians(LAT)
U_degph = U / (DEG2KM_LAT * np.clip(np.cos(LAT_RAD), 1e-6, None))
V_degph = V /  DEG2KM_LAT

cmap=plt.get_cmap('bwr')
u_abs_max=max(1.0,float(np.max(np.abs(U))))
norm=colors.Normalize(vmin=-u_abs_max,vmax=+u_abs_max)
step = 2  # 稀疏化

colors_flat = cmap(norm(U[::step, ::step].ravel()))
quiv = ax.quiver(
    LON[::step, ::step], LAT[::step, ::step], Zplane[::step, ::step],
    U_degph[::step, ::step], V_degph[::step, ::step], 0,
    length=2.0, normalize=False,
    colors=colors_flat,
    alpha=0.9, linewidth=0.5
)

# 颜色条：u (km/h) @ 该高度层
mappable = ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cb_quiv = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
cb_quiv.set_label(f"Zonal wind u (km/h) @ Alt ≈ {int(z_view_alt_ft):,} ft")

# 标题补充层信息
ax.set_title(
    "3D A*: Free-space vs With-spheres (km-based cost, lon-lat-ft display)\n"
    f"Wind vectors shown at ~{int(z_view_alt_ft):,} ft (layer z={z_view})",
    pad=14
)

ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)"); ax.set_zlabel("Altitude (ft)")
ax.legend(loc='upper left'); plt.tight_layout()
plt.show()

# 权重演化曲线
if weights_history:
    hours=[h for h,_,_ in weights_history]
    wDs=[wD for _,wD,_ in weights_history]
    wTs=[wT for _,_,wT in weights_history]
    plt.figure(figsize=(8,5))
    plt.plot(hours,wDs,'o-',label='wD (Distance Weight)')
    plt.plot(hours,wTs,'s--',label='wT (Time Weight)')
    plt.xlabel("Hour"); plt.ylabel("Weight Value")
    plt.title("Dynamic Weight Evolution Over Time")
    plt.legend(); plt.grid(True); plt.show()


# =========================
# Pareto 前沿曲线绘图
# =========================
num_hours = len(all_pareto_data)
fig, axes = plt.subplots(1, num_hours, figsize=(5*num_hours,5))
if num_hours == 1:
    axes = [axes]
for i, hour in enumerate(sorted(all_pareto_data.keys())):
    ax = axes[i]
    data = all_pareto_data[hour]['results']
    if not data:
        ax.set_title(f"Hour {hour}: No data"); continue
    dists = [r['dist'] for r in data]
    times = [r['time'] for r in data]
    knee_wD, knee_wT = all_pareto_data[hour]['knee']
    ax.plot(dists, times, 'o--', color='gray', alpha=0.7)
    knee_idx = find_knee_point(np.column_stack((dists, times)))
    ax.plot(dists[knee_idx], times[knee_idx], 'r*', markersize=15, label=f"Knee (wD={knee_wD:.2f})")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time (h)")
    ax.set_title(f"Hour {hour} Pareto Frontier")
    ax.legend()
plt.suptitle("Distance–Time Pareto Frontiers with Knee Points (Each Hour)")
plt.tight_layout()
plt.show()


# =========================
# 数值对比（报告可直接引用）
# =========================
print("\n================= NUMERICAL SUMMARY =================")

# ---- 计算：有障碍（实际飞行路径） ----
actual_total_dist = 0.0
actual_total_time = 0.0
for seg in dynamic_segments:
    for i in range(1, len(seg)):
        p1, p2 = seg[i - 1], seg[i]
        d = phys3d_km_between(p1, p2)
        u_w, v_w = wind_uv(p1[0], p1[1], p1[2])
        east_km, north_km = east_north_km_between(p1[0], p1[1], p2[0], p2[1])
        horiz = math.hypot(east_km, north_km)
        vg = max(50.0, TAS + (u_w * (east_km / horiz if horiz > 1e-6 else 0) + v_w * (north_km / horiz if horiz > 1e-6 else 0)))
        actual_total_dist += d
        actual_total_time += d / vg

# ---- 计算：无障碍理想自由空间路径 ----
empty_mask = np.zeros_like(blocked, dtype=bool)
ideal_path, ideal_dist, ideal_time = astar_3d(start_point, goal_point, empty_mask, 0.5, 0.5)
if ideal_path:
    print(f"[Summary] Dynamic (Obstacle) : dist={actual_total_dist:.1f} km, time={actual_total_time:.2f} h")
    print(f"          Ideal   (Free)     : dist={ideal_dist:.1f} km, time={ideal_time:.2f} h")
    inc_dist = (actual_total_dist - ideal_dist) / ideal_dist * 100
    inc_time = (actual_total_time - ideal_time) / ideal_time * 100
    print(f"          Δdist={inc_dist:+.1f}%, Δtime={inc_time:+.1f}%")
else:
    print("Warning: ideal (free-space) path not found — check grid size or mask settings.")

print("=====================================================")
