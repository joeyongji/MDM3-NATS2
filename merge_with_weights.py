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


def generate_spherical_mask_index(Nx, Ny, Nz, k_min=3, k_max=5):
    if Nx < 4 or Ny < 4 or Nz < 2: raise ValueError("Grid too small")
    k = np.random.randint(k_min, k_max + 1)
    Z, Y, X = np.ogrid[:Nz, :Ny, :Nx]
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx, cy, cz = np.random.randint(Nx // 3, 2 * Nx // 3), np.random.randint(Ny // 3,
                                                                                2 * Ny // 3), np.random.randint(1,
                                                                                                                Nz - 1)
        r = np.random.randint(2, max(3, min(Nx, Ny) // 4))
        mask |= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx


moves = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == dy == dz == 0: continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            moves.append((dx, dy, dz, dist))


def in_bounds(ix, iy, iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix, iy, iz, mask): return not mask[iz, iy, ix]
def neighbors(ix, iy, iz, mask):
    for dx, dy, dz, dist in moves:
        nx, ny, nz = ix + dx, iy + dy, iz + dz
        if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask):
            yield nx, ny, nz, dist, dx, dy


u_layer = np.array([-0.20, 0.00, +0.60, 0.00, -0.20])
v_layer = np.array([0.00, -0.05, 0.00, +0.05, 0.00])
A_JET, Y0_JET, SIGMA_JET = 0.30, (Ny - 1) / 2.0, Ny / 6.0


def u_jet(y): return A_JET * math.exp(-((y - Y0_JET) ** 2) / (2.0 * SIGMA_JET ** 2))


def get_wind_uv(ix, iy, iz):
    u = float(u_layer[iz]) + u_jet(iy)
    v = float(v_layer[iz])
    return u, v


max_wind_mag = max(math.hypot(u_layer[z] + A_JET, v_layer[z]) for z in range(Nz))
BEST_POSSIBLE_GROUND_SPEED = TAS + max_wind_mag


def random_free_cell(mask, margin=1, min_dist=10):
    if Nx <= 2 * margin or Ny <= 2 * margin: margin = 0
    for _ in range(20000):
        ix, iy, iz = np.random.randint(margin, Nx - margin), np.random.randint(margin, Ny - margin), np.random.randint(
            0, Nz)
        if passable(ix, iy, iz, mask):
            for _ in range(2000):
                jx, jy, jz = np.random.randint(margin, Nx - margin), np.random.randint(margin,
                                                                                       Ny - margin), np.random.randint(
                    0, Nz)
                dist_val = math.sqrt((ix - jx) ** 2 + (iy - jy) ** 2 + (iz - jz) ** 2)
                if passable(jx, jy, jz, mask) and dist_val >= min_dist:
                    return (ix, iy, iz), (jx, jy, jz)
    raise RuntimeError("No valid start-goal pair found.")


# --- 3. 参数化的 A* 算法 ---
def astar_3d(start, goal, mask, w_d, w_t):
    def heuristic_weighted(a, b):
        dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        time_est = dist / BEST_POSSIBLE_GROUND_SPEED
        return w_d * dist + w_t * time_est

    openh, closed = [], set()
    heappush(openh, (heuristic_weighted(start, goal), start))
    g, g_dist, g_time, parent = {start: 0.0}, {start: 0.0}, {start: 0.0}, {start: None}
    EPS = 1e-5
    while openh:
        _, u_node = heappop(openh)
        if u_node in closed: continue
        closed.add(u_node)
        if u_node == goal:
            path = []
            cur = u_node
            while cur is not None: path.append(cur); cur = parent[cur]
            path.reverse()
            return path, g[u_node], g_dist[u_node], g_time[u_node], closed
        ux, uy, uz = u_node
        u_wind, v_wind = get_wind_uv(ux, uy, uz)
        for nx, ny, nz, dist, dx_step, dy_step in neighbors(ux, uy, uz, mask):
            v_node = (nx, ny, nz)
            ex, ey = dx_step / dist, dy_step / dist
            Vg = TAS + (u_wind * ex + v_wind * ey)
            Vg = max(EPS, Vg)
            time_edge = dist / Vg
            edge_cost = w_d * dist + w_t * time_edge
            new_g = g[u_node] + edge_cost
            if v_node not in g or new_g < g[v_node]:
                g[v_node], g_dist[v_node], g_time[v_node], parent[v_node] = new_g, g_dist[u_node] + dist, g_time[
                    u_node] + time_edge, u_node
                f_cost = new_g + heuristic_weighted(v_node, goal)
                heappush(openh, (f_cost, v_node))
    return None, np.inf, np.inf, np.inf, closed


# --- 4. 分段规划函数 ---
EARTH_RADIUS_KM = 6371.0


def haversine_km(lon1, lat1, lon2, lat2):
    # ... (函数体与之前相同)
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2);
    dlat = math.radians(lat2 - lat1);
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def phys_km_between(a, b):
    lon1, lat1 = idx_to_phys_x(a[0]), idx_to_phys_y(a[1]);
    lon2, lat2 = idx_to_phys_x(b[0]), idx_to_phys_y(b[1])
    return haversine_km(lon1, lat1, lon2, lat2)


def path_length_km(path):
    if not path or len(path) < 2: return 0.0
    return sum(phys_km_between(path[i - 1], path[i]) for i in range(1, len(path)))


def nearest_free_cell_around(xf, yf, zf, mask, max_radius=3):
    # ... (函数体与之前相同)
    xi, yi, zi = int(round(xf)), int(round(yf)), int(round(zf))
    for r in range(max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    nx, ny, nz = xi + dx, yi + dy, zi + dz
                    if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask): return (nx, ny, nz)
    return None


def choose_leg_goal(current, final_goal, leg_km, mask):
    # ... (函数体与之前相同)
    total_km = phys_km_between(current, final_goal)
    if total_km <= leg_km * 1.1: return final_goal
    t = leg_km / total_km
    xf, yf, zf = current[0] + t * (final_goal[0] - current[0]), current[1] + t * (final_goal[1] - current[1]), current[
        2] + t * (final_goal[2] - current[2])
    subgoal = nearest_free_cell_around(xf, yf, zf, mask, max_radius=5)
    return subgoal if subgoal is not None else final_goal


def plan_by_legs(start, final_goal, mask, w_d, w_t, leg_km=800.0, max_legs=10):
    curr = start
    total_path, explored_union = [curr], set()
    total_cost, total_dist, total_time = 0.0, 0.0, 0.0
    for leg_idx in range(1, max_legs + 1):
        if curr == final_goal: break
        subgoal = choose_leg_goal(curr, final_goal, leg_km, mask)
        path, cost, dist, time, explored = astar_3d(curr, subgoal, mask, w_d, w_t)
        if not path:
            return None, np.inf, np.inf, None
        total_cost += cost
        total_dist += dist
        total_time += time
        explored_union.update(explored)
        total_path.extend(path[1:])
        curr = subgoal
        if curr == final_goal: break
    return total_path, total_dist, total_time, explored_union


# --- 5. 帕累托前沿分析函数 ---
def find_knee_point(points):
    # ... (函数体与之前相同)
    normalized_points = np.array(points)
    min_vals, max_vals = normalized_points.min(axis=0), normalized_points.max(axis=0)
    range_vals = max_vals - min_vals
    if np.any(range_vals < 1e-6): return 0
    normalized_points = (normalized_points - min_vals) / range_vals
    line_vec = normalized_points[-1] - normalized_points[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = normalized_points - normalized_points[0]
    scalar_proj = vec_from_first.dot(line_vec_norm)
    vec_from_first_parallel = scalar_proj[:, np.newaxis] * line_vec_norm
    vec_to_line = vec_from_first - vec_from_first_parallel
    distances = np.linalg.norm(vec_to_line, axis=1)
    return np.argmax(distances)


# --- 6. 主程序 ---
blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)
start_point, goal_point = random_free_cell(blocked, min_dist=max(Nx, Ny) * 0.8)

weights_to_test = [(w / 20.0, 1.0 - w / 20.0) for w in range(21)][::-1]
results = []

print("--- Phase 1: Running Pareto Weight Scan on Segmented Planner ---")
for wD, wT in weights_to_test:
    print(f"  Testing weights: wD={wD:.2f}, wT={wT:.2f}")
    path, total_dist, total_time, _ = plan_by_legs(start_point, goal_point, blocked, wD, wT)
    if path:
        results.append({'wD': wD, 'wT': wT, 'dist': total_dist, 'time': total_time})

if not results:
    print("Pathfinding failed for all weights.")
else:
    results.sort(key=lambda r: r['dist'])
    pareto_points = np.array([[r['dist'], r['time']] for r in results])
    knee_index = find_knee_point(pareto_points)
    knee_result = results[knee_index]
    optimal_weights = (knee_result['wD'], knee_result['wT'])

    print("\n" + "=" * 40)
    print("          Weight Optimization Analysis Complete")
    print("=" * 40)
    print(f"Optimal balance (Knee Point) found with weights:")
    print(f"  - Recommended Weights (wD, wT): ({optimal_weights[0]:.2f}, {optimal_weights[1]:.2f})")
    print(f"  - Resulting Path Distance: {knee_result['dist']:.2f} grid units")
    print(f"  - Resulting Path Time: {knee_result['time']:.2f} time units")
    print("=" * 40 + "\n")

    plt.figure(figsize=(10, 7))
    plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'o-', label='Pareto Frontier')
    plt.scatter([knee_result['dist']], [knee_result['time']], color='red', s=200, marker='*', zorder=10,
                label=f'Knee Point\nwD={optimal_weights[0]:.2f}, wT={optimal_weights[1]:.2f}')
    plt.title('Pareto Frontier for Distance vs. Time (Segmented Planner with Wind)')
    plt.xlabel('Total Distance (grid units)');
    plt.ylabel('Total Time (time units)')
    plt.legend();
    plt.grid(True)
    plt.show()

    print("\n--- Phase 2: Re-running simulation with optimal weights for final visualization ---")
    final_path, final_dist, final_time, final_explored = plan_by_legs(
        start_point, goal_point, blocked, optimal_weights[0], optimal_weights[1]
    )

    if final_path:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Final Optimal Path (wD={optimal_weights[0]:.2f}, wT={optimal_weights[1]:.2f})")
        u, v = np.linspace(0, np.pi, 15), np.linspace(0, 2 * np.pi, 30)
        uu, vv = np.meshgrid(u, v)
        for cx, cy, cz, r in spheres_idx:
            Xs = cx + r * np.sin(uu) * np.cos(vv);
            Ys = cy + r * np.sin(uu) * np.sin(vv);
            Zs = cz + r * np.cos(uu)
            ax.plot_surface(idx_to_phys_x(Xs), idx_to_phys_y(Ys), idx_to_phys_z(Zs), alpha=0.2, color='r', linewidth=0)

        ax.plot([idx_to_phys_x(x) for x, y, z in final_path], [idx_to_phys_y(y) for x, y, z in final_path],
                [idx_to_phys_z(z) for x, y, z in final_path],
                linewidth=3, color='blue', label=f"Final Path (Dist={final_dist:.2f}, Time={final_time:.2f})")

        sx, sy, sz = start_point
        gx, gy, gz = goal_point
        ax.scatter([idx_to_phys_x(sx)], [idx_to_phys_y(sy)], [idx_to_phys_z(sz)], s=100, marker='o', label='Start',
                   color='green', depthshade=False)
        ax.scatter([idx_to_phys_x(gx)], [idx_to_phys_y(gy)], [idx_to_phys_z(gz)], s=100, marker='*', label='Goal',
                   color='purple', depthshade=False)

        ax.set_xlabel("Longitude (deg)");
        ax.set_ylabel("Latitude (deg)");
        ax.set_zlabel("Flight level (ft)")
        ax.legend(loc='upper left');
        ax.view_init(elev=25, azim=-110)
        plt.show()