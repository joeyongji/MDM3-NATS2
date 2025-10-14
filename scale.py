import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from heapq import heappush, heappop
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import math, random

# =========================
# 基本网格与可视化坐标（用于显示）
# =========================
np.random.seed(53)
Nx, Ny, Nz = 15, 15, 5

lon_min, lon_max = -70.0, -10.0
lat_min, lat_max =  40.0,  60.0
alt_min, alt_max = 30000.0, 38000.0  # ft

lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

def idx_to_lon(ix): return np.interp(ix, np.arange(Nx), lon_vals)
def idx_to_lat(iy): return np.interp(iy, np.arange(Ny), lat_vals)
def idx_to_alt(iz): return np.interp(iz, np.arange(Nz), alt_vals)  # ft

# =========================
# 禁飞区：索引空间球体（计算用 & 3D 展示）
# =========================
def generate_spherical_mask_index(Nx, Ny, Nz, k_min=2, k_max=4):
    if Nx < 4 or Ny < 4 or Nz < 2:
        raise ValueError("Grid too small")
    k = np.random.randint(k_min, k_max + 1)
    Z, Y, X = np.ogrid[:Nz, :Ny, :Nx]
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx = np.random.randint(Nx // 3, 2 * Nx // 3)
        cy = np.random.randint(Ny // 3, 2 * Ny // 3)
        cz = np.random.randint(1, Nz - 1)
        r  = np.random.randint(1, max(2, min(Nx, Ny) // 5))
        mask |= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx

blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)

# =========================
# 26-邻接（方向；距离在物理空间算）
# =========================
moves = [(dx,dy,dz) for dx in (-1,0,1)
                  for dy in (-1,0,1)
                  for dz in (-1,0,1)
                  if not (dx==0 and dy==0 and dz==0)]
def in_bounds(ix,iy,iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix,iy,iz,mask): return not mask[iz,iy,ix]

# =========================
# 物理空间（用于计算距离/速度）
# =========================
DEG2KM_LAT = 111.0     # 每度纬度约 111 km
FT_TO_KM   = 0.0003048 # ft -> km

def east_north_km_between(ix1,iy1, ix2,iy2):
    lon1, lat1 = idx_to_lon(ix1), idx_to_lat(iy1)
    lon2, lat2 = idx_to_lon(ix2), idx_to_lat(iy2)
    lat_mid = math.radians(0.5*(lat1+lat2))
    dlon = (lon2 - lon1)
    dlat = (lat2 - lat1)
    east_km  = dlon * DEG2KM_LAT * math.cos(lat_mid)
    north_km = dlat * DEG2KM_LAT
    return east_km, north_km

def vert_km_between(iz1, iz2):
    z1_km = idx_to_alt(iz1) * FT_TO_KM
    z2_km = idx_to_alt(iz2) * FT_TO_KM
    return abs(z2_km - z1_km)

def phys3d_km_between(a,b):
    ix1,iy1,iz1 = a; ix2,iy2,iz2 = b
    ex,ny = east_north_km_between(ix1,iy1, ix2,iy2)
    vk    = vert_km_between(iz1,iz2)
    return math.hypot(math.hypot(ex,ny), vk)

# =========================
# 风场（静态，水平风）：层风 + 纬向急流条带
# =========================
TAS = 900.0  # km/h

# 每层基础风（km/h）——把某些层改为负数就能看到向西的箭头
u_layer = np.array([-20., -40., +80., +40., -10.])  # 东西向（+东，-西）
v_layer = np.array([ 0., -10.,  0., +10.,  0.])     # 南北向（+北，-南）

# 纬向急流（叠加到 u；若想向西急流，将 A_JET 设为负）
A_JET     = -150.0
Y0_JET    = (Ny - 1)/2.0
SIGMA_JET = Ny/6.0
def u_jet(iy):
    return A_JET * math.exp(-((iy - Y0_JET)**2)/(2*SIGMA_JET**2))

def wind_uv(ix,iy,iz):
    """返回 (u_total, v_total) in km/h"""
    return float(u_layer[iz]) + u_jet(iy), float(v_layer[iz])

# 启发函数最乐观地速
max_wind_mag = max(math.hypot(u_layer[z]+A_JET, v_layer[z]) for z in range(Nz))
K_HEUR = 0.5 + 0.5/(TAS + max_wind_mag)
def heuristic(a,b): return phys3d_km_between(a,b) * K_HEUR

# 邻居生成（同时给出 EN 分量便于投影）
def neighbors(ix,iy,iz,mask):
    for dx,dy,dz in moves:
        nx,ny,nz = ix+dx, iy+dy, iz+dz
        if in_bounds(nx,ny,nz) and passable(nx,ny,nz,mask):
            dist_km   = phys3d_km_between((ix,iy,iz),(nx,ny,nz))
            east_km, north_km = east_north_km_between(ix,iy, nx,ny)
            yield nx,ny,nz, dist_km, east_km, north_km

# =========================
# 起终点
# =========================
def random_free_cell(mask, margin=1, min_dist_km=1000.0):
    if Nx <= 2*margin or Ny <= 2*margin:
        margin = 0
    for _ in range(20000):
        ix = np.random.randint(margin, Nx-margin) if Nx>2*margin else np.random.randint(0,Nx)
        iy = np.random.randint(margin, Ny-margin) if Ny>2*margin else np.random.randint(0,Ny)
        iz = np.random.randint(0, Nz)
        if passable(ix,iy,iz,mask):
            for _ in range(2000):
                jx = np.random.randint(margin, Nx-margin) if Nx>2*margin else np.random.randint(0,Nx)
                jy = np.random.randint(margin, Ny-margin) if Ny>2*margin else np.random.randint(0,Ny)
                jz = np.random.randint(0, Nz)
                if passable(jx,jy,jz,mask):
                    if phys3d_km_between((ix,iy,iz),(jx,jy,jz)) >= min_dist_km:
                        return (ix,iy,iz), (jx,jy,jz)
    raise RuntimeError("No valid start-goal pair found.")

# =========================
# A*（cost = 0.5*距离 + 0.5*时间）
# =========================
def astar_3d(start, goal, mask):
    openh=[]; heappush(openh,(0.0,start))
    g={start:0.0}; g_dist={start:0.0}; parent={start:None}; closed=set()
    EPS=1e-6; VMIN=50.0  # km/h

    while openh:
        _, u = heappop(openh)
        if u in closed: continue
        closed.add(u)
        if u==goal:
            path=[]; cur=u
            while cur is not None: path.append(cur); cur=parent[cur]
            return path[::-1], g[u], g_dist[u], closed

        ux,uy,uz = u
        u_wind, v_wind = wind_uv(ux,uy,uz)
        for nx,ny,nz, dist_km, east_km, north_km in neighbors(ux,uy,uz,mask):
            v = (nx,ny,nz)
            horiz = math.hypot(east_km, north_km)
            if horiz < EPS:
                Vg = TAS
            else:
                ex,ey = east_km/horiz, north_km/horiz
                Vg = TAS + (u_wind*ex + v_wind*ey)
            Vg = max(VMIN, Vg)
            t_edge = dist_km / Vg
            edge_cost = 0.5*dist_km + 0.5*t_edge

            new_g = g[u] + edge_cost
            if v not in g or new_g < g[v]:
                g[v]=new_g; g_dist[v]=g_dist[u]+dist_km; parent[v]=u
                heappush(openh, (new_g + heuristic(v,goal), v))
    return None, np.inf, np.inf, closed

# =========================
# 规划：Free-space vs With-spheres
# =========================
start_point, goal_point = random_free_cell(blocked, min_dist_km=1000.0)
path_obs,  cost_obs,  dist_obs,  explored = astar_3d(start_point, goal_point, blocked)
empty = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, dist_free, _ = astar_3d(start_point, goal_point, empty)

# =========================
# 3D 可视化（两条路径不同线型 + 风矢量层/色条）
# =========================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D A*: Free-space vs With-spheres (km-based cost, lon-lat-ft display)")

# --- 3D 球体禁飞区
u_mesh = np.linspace(0, np.pi, 20)
v_mesh = np.linspace(0, 2 * np.pi, 40)
uu, vv = np.meshgrid(u_mesh, v_mesh)
for cx, cy, cz, r in spheres_idx:
    Xs = cx + r * np.sin(uu) * np.cos(vv)
    Ys = cy + r * np.sin(uu) * np.sin(vv)
    Zs = cz + r * np.cos(uu)
    ax.plot_surface(idx_to_lon(Xs), idx_to_lat(Ys), idx_to_alt(Zs),
                    alpha=0.28, linewidth=0, cmap='Reds')

# --- 路径转经纬高
def path_lon_lat_alt(path):
    xs=[idx_to_lon(i) for i,j,k in path]
    ys=[idx_to_lat(j) for i,j,k in path]
    zs=[idx_to_alt(k) for i,j,k in path]
    return xs, ys, zs

# --- 绘制两条路径（线型 + 标记）
if path_free:
    xF, yF, zF = path_lon_lat_alt(path_free)
    ax.plot(xF, yF, zF,
            linestyle='--', color='black', linewidth=3,
            marker='o', markersize=4, markevery=max(1, len(xF)//12),
            label=f"Free-space (cost={cost_free:.2f}, dist={dist_free:.1f} km)")

if path_obs:
    xO, yO, zO = path_lon_lat_alt(path_obs)
    ax.plot(xO, yO, zO,
            linestyle='-', color='black', linewidth=3,
            marker='^', markersize=5, markevery=max(1, len(xO)//12),
            label=f"With spheres (cost={cost_obs:.2f}, dist={dist_obs:.1f} km)")

# --- 起终点
sx,sy,sz = start_point
gx,gy,gz = goal_point
ax.scatter([idx_to_lon(sx)], [idx_to_lat(sy)], [idx_to_alt(sz)],
           c='green', s=60, depthshade=False, label='Start')
ax.scatter([idx_to_lon(gx)], [idx_to_lat(gy)], [idx_to_alt(gz)],
           c='purple', s=80, depthshade=False, marker='*', label='Goal')

# --- 风矢量：在一个高度层 z_view 上，并加参照平面与 bwr 色条（红=东，蓝=西）
z_view = Nz // 2
z_view_alt_ft = idx_to_alt(z_view)

LON, LAT = np.meshgrid(lon_vals, lat_vals)
U = np.zeros((Ny, Nx)); V = np.zeros((Ny, Nx))
for iy in range(Ny):
    for ix in range(Nx):
        U[iy, ix], V[iy, ix] = wind_uv(ix, iy, z_view)

# 参照平面（半透明），帮助读者知道箭头所在高度层
Zplane = np.full_like(LON, z_view_alt_ft)
ax.plot_surface(LON, LAT, Zplane, rstride=Ny-1, cstride=Nx-1,
                color=(0.85, 0.85, 0.85, 0.25), linewidth=0, shade=False)

# km/h -> 度/小时（仅用于可视化尺度）
LAT_RAD = np.radians(LAT)
U_degph = U / (DEG2KM_LAT * np.clip(np.cos(LAT_RAD), 1e-6, None))
V_degph = V /  DEG2KM_LAT

# 用签名的 u 给箭头上色（bwr：蓝=向西，红=向东）
cmap = plt.get_cmap('bwr')
u_abs_max = max(1.0, float(np.max(np.abs(U))))
norm = colors.Normalize(vmin=-u_abs_max, vmax=+u_abs_max)

step = 2  # 稀疏化
# 关键：把颜色展平为 (N_arrows, 4)
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

# 轴标签与图例
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Altitude (ft)")
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# =========================
# 数值对比（报告可直接引用）
# =========================
if path_obs and path_free and dist_free>0 and cost_free>0:
    inc_dist = (dist_obs - dist_free)/dist_free*100
    inc_cost = (cost_obs - cost_free)/cost_free*100
    print(f"[Summary] Free : cost={cost_free:.2f}, dist={dist_free:.1f} km")
    print(f"          Obst : cost={cost_obs:.2f} ({inc_cost:+.1f}%), dist={dist_obs:.1f} km ({inc_dist:+.1f}%)")
else:
    print("Warning: one of the paths not found; adjust obstacles or grid size.")
