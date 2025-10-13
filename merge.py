import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math

# -----------------------------
# 基本参数
# -----------------------------
np.random.seed(53)

Nx, Ny, Nz = 15, 15, 5

# 空速（格/时间单位）
TAS = 0.5

# 地图坐标轴（仅用于可视化标签）
lon_min, lon_max = -70.0, -10.0
lat_min, lat_max =  40.0,  60.0
alt_min, alt_max = 30000.0, 38000.0
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

def idx_to_phys_x(ix): return np.interp(ix, np.arange(Nx), lon_vals)
def idx_to_phys_y(iy): return np.interp(iy, np.arange(Ny), lat_vals)
def idx_to_phys_z(iz): return np.interp(iz, np.arange(Nz), alt_vals)

# -----------------------------
# 索引空间内的实心球禁飞区
# -----------------------------
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

# -----------------------------
# 26 邻接动作集合
# -----------------------------
moves = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == dy == dz == 0:
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)  # 欧氏步长（格）
            moves.append((dx, dy, dz, dist))

def in_bounds(ix, iy, iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix, iy, iz, mask): return not mask[iz, iy, ix]

def neighbors(ix, iy, iz, mask):
    for dx, dy, dz, dist in moves:
        nx, ny, nz = ix + dx, iy + dy, iz + dz
        if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask):
            yield nx, ny, nz, dist, dx, dy, dz

# -----------------------------
# 风场（静态，水平风）：每个高度常风 + 纬向急流条带（叠加到 u 上）
# -----------------------------
# 每个高度层的基础风 (u_z, v_z) —— 你可按需要修改
u_layer = np.array([+0.05, +0.10, +0.20, +0.10, +0.00])  # 东西向：中层最强尾风
v_layer = np.array([ 0.00, -0.05,  0.00, +0.05,  0.00])  # 南北向：上下层给一点侧风

# 急流条带参数（按 y 索引定义）
A_JET     = 0.30                 # 峰值风速（叠加到 u）
Y0_JET    = (Ny - 1) / 2.0       # 条带中心（网格索引）
SIGMA_JET = Ny / 6.0             # 条带宽度（索引单位）
def u_jet(y):
    return A_JET * math.exp(-((y - Y0_JET) ** 2) / (2.0 * SIGMA_JET ** 2))

def get_wind_uv(ix, iy, iz):
    """返回 (u_total, v_total) at (ix,iy,iz)，只含水平风。"""
    u = float(u_layer[iz]) + u_jet(iy)
    v = float(v_layer[iz])
    return u, v

# 启发函数的“最乐观地速上界”——遍历各高度的最大风模长（急流取峰值）
max_wind_mag = 0.0
for z in range(Nz):
    u_peak = u_layer[z] + A_JET
    v_val  = v_layer[z]
    max_wind_mag = max(max_wind_mag, math.hypot(u_peak, v_val))
K_HEUR = 0.5 + 0.5 / (TAS + max_wind_mag)   # h = dist * K_HEUR

def heuristic(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist * K_HEUR

# -----------------------------
# 起终点生成
# -----------------------------
def random_free_cell(mask, margin=1, min_dist=5):
    if Nx <= 2 * margin or Ny <= 2 * margin:
        margin = 0
    for _ in range(20000):
        ix = np.random.randint(margin, Nx - margin) if Nx > 2 * margin else np.random.randint(0, Nx)
        iy = np.random.randint(margin, Ny - margin) if Ny > 2 * margin else np.random.randint(0, Ny)
        iz = np.random.randint(0, Nz)
        if passable(ix, iy, iz, mask):
            for _ in range(2000):
                jx = np.random.randint(margin, Nx - margin) if Nx > 2 * margin else np.random.randint(0, Nx)
                jy = np.random.randint(margin, Ny - margin) if Ny > 2 * margin else np.random.randint(0, Ny)
                jz = np.random.randint(0, Nz)
                dist_val = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2 + (iz - jz) ** 2)
                if passable(jx, jy, jz, mask) and dist_val >= min_dist:
                    return (ix, iy, iz), (jx, jy, jz)
    raise RuntimeError("No valid start-goal pair found.")

# -----------------------------
# A*（加入风影响）
# -----------------------------
def astar_3d(start, goal, mask):
    openh = []
    heappush(openh, (0.0, start))
    g = {start: 0.0}         # 累计代价（0.5*dist + 0.5*time）
    g_dist = {start: 0.0}    # 累计距离（可选，便于对比）
    parent = {start: None}
    closed = set()

    EPS = 1e-3   # 防止 Vg 过小
    VMIN_CLAMP = 0.05

    while openh:
        _, u_node = heappop(openh)
        if u_node in closed:
            continue
        closed.add(u_node)

        if u_node == goal:
            path = []
            cur = u_node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, g[u_node], g_dist[u_node], closed

        ux, uy, uz = u_node
        # 当前格点风（水平）
        u_wind, v_wind = get_wind_uv(ux, uy, uz)

        for nx, ny, nz, dist, dx_step, dy_step, dz_step in neighbors(ux, uy, uz, mask):
            v_node = (nx, ny, nz)

            # 单位方向向量（索引空间）
            ex = dx_step / dist
            ey = dy_step / dist
            # ez = dz_step / dist  # 垂直风忽略

            # 地速近似（风在水平面的投影）
            Vg = TAS + (u_wind * ex + v_wind * ey)
            Vg = max(EPS, Vg)           # 防止 0/负
            Vg = max(VMIN_CLAMP, Vg)    # 给一个更保守的下限

            # 边时间 & 代价
            time_edge = dist / Vg
            edge_cost = 0.5 * dist + 0.5 * time_edge

            new_g = g[u_node] + edge_cost

            if v_node not in g or new_g < g[v_node]:
                g[v_node] = new_g
                g_dist[v_node] = g_dist[u_node] + dist
                parent[v_node] = u_node
                f_cost = new_g + heuristic(v_node, goal)
                heappush(openh, (f_cost, v_node))

    return None, np.inf, np.inf, closed

# -----------------------------
# 规划（有禁飞 & 无禁飞）——风场都开启
# -----------------------------
start_point, goal_point = random_free_cell(blocked, min_dist=5)
path_obs,  cost_obs,  dist_obs,  explored = astar_3d(start_point, goal_point, blocked)
empty_mask = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, dist_free, _       = astar_3d(start_point, goal_point, empty_mask)

# -----------------------------
# 可视化
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D A*: Distance+Time with Layered Wind + Latitudinal Jet & Spherical No-fly")

# 画球体禁飞区（索引球 -> 物理表面）
u_mesh = np.linspace(0, np.pi, 15)
v_mesh = np.linspace(0, 2 * np.pi, 30)
uu, vv = np.meshgrid(u_mesh, v_mesh)
for cx, cy, cz, r in spheres_idx:
    Xs = cx + r * np.sin(uu) * np.cos(vv)
    Ys = cy + r * np.sin(uu) * np.sin(vv)
    Zs = cz + r * np.cos(uu)
    ax.plot_surface(idx_to_phys_x(Xs), idx_to_phys_y(Ys), idx_to_phys_z(Zs),
                    alpha=0.28, linewidth=0, cmap='Reds')

# （可选）显示抽样的探索节点
if path_obs:
    ex = list(explored)
    if len(ex) > 1000:
        ex = random.sample(ex, 1000)
    ax.scatter([idx_to_phys_x(x) for x, y, z in ex],
               [idx_to_phys_y(y) for x, y, z in ex],
               [idx_to_phys_z(z) for x, y, z in ex],
               s=3, c='gray', alpha=0.12, label="Explored Nodes")

# 无禁飞参考路径（虚线）——同样包含风的影响
if path_free:
    ax.plot([idx_to_phys_x(x) for x, y, z in path_free],
            [idx_to_phys_y(y) for x, y, z in path_free],
            [idx_to_phys_z(z) for x, y, z in path_free],
            linestyle='--', linewidth=2,
            label=f"Free-space (cost={cost_free:.2f}, dist={dist_free:.2f})")

# 有禁飞路径（实线）
if path_obs:
    ax.plot([idx_to_phys_x(x) for x, y, z in path_obs],
            [idx_to_phys_y(y) for x, y, z in path_obs],
            [idx_to_phys_z(z) for x, y, z in path_obs],
            linewidth=3,
            label=f"With spheres (cost={cost_obs:.2f}, dist={dist_obs:.2f})")

# 起终点
sx, sy, sz = start_point
gx, gy, gz = goal_point
ax.scatter([idx_to_phys_x(sx)], [idx_to_phys_y(sy)], [idx_to_phys_z(sz)],
           s=60, marker='o', label='Start', color='green', depthshade=False)
ax.scatter([idx_to_phys_x(gx)], [idx_to_phys_y(gy)], [idx_to_phys_z(gz)],
           s=80, marker='*', label='Goal', color='purple', depthshade=False)

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Flight level (ft)")
# 在图例中附带风场关键信息，帮助读图
ax.legend(loc='upper left',
          title=f"Layer wind (u,v) at z & Jet: A={A_JET:.2f}, y0={Y0_JET:.1f}, σ={SIGMA_JET:.1f}")
plt.tight_layout()
plt.show()

# -----------------------------
# 数值对比（节省/增加百分比）
# -----------------------------
if path_obs and path_free and dist_free > 0 and cost_free > 0:
    increase_dist = (dist_obs - dist_free) / dist_free * 100
    increase_cost = (cost_obs - cost_free) / cost_free * 100
    print(f"Free-space Path: Cost={cost_free:.2f}, Distance={dist_free:.2f} grid units")
    print(f"Obstacle Path:   Cost={cost_obs:.2f} (+{increase_cost:.1f}%), Distance={dist_obs:.2f} grid units ({increase_dist:+.1f}% longer)")

# ---- 工具：大圆距离（km）
EARTH_RADIUS_KM = 6371.0
def haversine_km(lon1, lat1, lon2, lat2):
    import math
    rlat1 = math.radians(lat1); rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def phys_km_between(a, b):
    """只用经纬度计算水平大圆距离（忽略高度差）。"""
    lon1, lat1 = idx_to_phys_x(a[0]), idx_to_phys_y(a[1])
    lon2, lat2 = idx_to_phys_x(b[0]), idx_to_phys_y(b[1])
    return haversine_km(lon1, lat1, lon2, lat2)

# ---- 工具：就近找可通行网格
def nearest_free_cell_around(xf, yf, zf, mask, max_radius=3):
    """从连续坐标 (xf,yf,zf) 四舍五入到网格，并在邻域内找可通行单元。"""
    xi, yi, zi = int(round(xf)), int(round(yf)), int(round(zf))
    for r in range(max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx, ny, nz = xi + dx, yi + dy, zi + dz
                    if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask):
                        return (nx, ny, nz)
    return None

# ---- 选择“本段”的目标点（尽量距当前 ~leg_km，并朝向总目标）
def choose_leg_goal(current, final_goal, leg_km, mask):
    total_km = phys_km_between(current, final_goal)
    if total_km <= leg_km:
        return final_goal  # 剩余不足一段，直接终点

    # 沿 current->final_goal 的索引直线，取比例 t 对应的连续坐标
    t = leg_km / total_km
    xf = current[0] + t * (final_goal[0] - current[0])
    yf = current[1] + t * (final_goal[1] - current[1])
    zf = current[2] + t * (final_goal[2] - current[2])  # 保持高度随比例平滑变化

    # 就近贴回可通行网格
    subgoal = nearest_free_cell_around(xf, yf, zf, mask, max_radius=3)
    if subgoal is not None:
        return subgoal

    # 若附近找不到空格，退化：再扩一圈或直接用最终目标
    subgoal = nearest_free_cell_around(xf, yf, zf, mask, max_radius=5)
    return subgoal if subgoal is not None else final_goal

# ---- 汇总路径的真实公里数（便于打印）
def path_length_km(path):
    if not path or len(path) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(path)):
        s += phys_km_between(path[i-1], path[i])
    return s

# ---- 分段规划主函数
def plan_by_legs(start, final_goal, mask, leg_km=1000.0, max_legs=100):
    curr = start
    segments = []
    total_path = [curr]
    total_cost = 0.0
    total_dist = 0.0   # 仍是你 A* 内部的“格距离”统计
    explored_union = set()

    for leg_idx in range(1, max_legs + 1):
        if curr == final_goal:
            break
        subgoal = choose_leg_goal(curr, final_goal, leg_km, mask)
        path, cost, dist, explored = astar_3d(curr, subgoal, mask)
        if not path:
            print(f"[WARN] 第 {leg_idx} 段规划失败（从 {curr} 到 {subgoal}）。终止。")
            return segments, total_path, total_cost, total_dist, explored_union

        seg_km = path_length_km(path)
        segments.append({
            "idx": leg_idx,
            "start": curr,
            "goal": subgoal,
            "path": path,
            "cost": cost,
            "dist_grid": dist,
            "length_km": seg_km
        })

        # 累加
        total_cost += cost
        total_dist += dist
        explored_union.update(explored)

        # 拼接路径（避免重复当前点）
        total_path.extend(path[1:])
        curr = subgoal

    return segments, total_path, total_cost, total_dist, explored_union

# ============================================
# 使用示例：对“有禁飞”的地图，按 1000 km 分段规划
# ============================================
LEG_KM = 1000.0
segments, total_path_seg, tot_cost_seg, tot_dist_seg, explored_seg = plan_by_legs(
    start_point, goal_point, blocked, leg_km=LEG_KM, max_legs=100
)

# 打印结果
print(f"\n== 分段规划（每段 ~{LEG_KM:.0f} km）==")
total_km_seg = path_length_km(total_path_seg)
print(f"段数: {len(segments)}")
for s in segments:
    print(f"  段 {s['idx']:>2}: km={s['length_km']:.1f}, cost={s['cost']:.2f}, grid_dist={s['dist_grid']:.2f}, "
          f"start={s['start']} -> goal={s['goal']}")
print(f"总长度(公里): {total_km_seg:.1f} km, 总代价: {tot_cost_seg:.2f}, 总格距: {tot_dist_seg:.2f}\n")