import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.interpolate import interp1d


np.random.seed(53)
Nx, Ny, Nz = 15, 15, 5
TAS= 0.5  # Normalized TAS (grids/time unit, e.g., half grid per unit time)
k = 0.5 + 0.5 / TAS # Combined weight: 0.5 dist + 0.5 time, k=1.5
lon_min, lon_max = -70.0, -10.0
lat_min, lat_max = 40.0, 60.0
alt_min, alt_max = 30000.0, 38000.0
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)


def idx_to_phys_x(ix): return np.interp(ix, np.arange(Nx), lon_vals)
def idx_to_phys_y(iy): return np.interp(iy, np.arange(Ny), lat_vals)
def idx_to_phys_z(iz): return np.interp(iz, np.arange(Nz), alt_vals)


def generate_spherical_mask_index(Nx, Ny, Nz, k_min=2, k_max=4):
    if Nx < 4 or Ny < 4 or Nz < 2: raise ValueError("Grid too small")
    k = np.random.randint(k_min, k_max + 1)
    Z, Y, X = np.ogrid[:Nz, :Ny, :Nx]
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx, cy, cz = np.random.randint(Nx // 3, 2 * Nx // 3), np.random.randint(Ny // 3,
                                                                                2 * Ny // 3), np.random.randint(1,
                                                                                                                Nz - 1)
        r = np.random.randint(1, max(2, min(Nx, Ny) // 5))
        mask |= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r ** 2
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx


blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)
moves = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == dy == dz == 0: continue
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)  # Grid units
            moves.append((dx, dy, dz, dist))


def in_bounds(ix, iy, iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix, iy, iz, mask): return not mask[iz, iy, ix]


def neighbors(ix, iy, iz, mask):
    for dx, dy, dz, dist in moves:
        nx, ny, nz = ix + dx, iy + dy, iz + dz
        if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask):
            yield nx, ny, nz, dist


def heuristic(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    return dist * k


def random_free_cell(mask, margin=1, min_dist=5):
    if Nx <= 2 * margin or Ny <= 2 * margin: margin = 0
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



def astar_3d(start, goal, mask):
    openh = []
    heappush(openh, (0.0, start))
    g = {start: 0.0}
    g_dist = {start: 0.0}
    parent = {start: None}
    closed = set()
    while openh:
        _, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, g[u], g_dist[u], closed
        ux, uy, uz = u
        for nx, ny, nz, dist in neighbors(ux, uy, uz, mask):
            v = (nx, ny, nz)


            time = dist / TAS
            edge_cost = 0.5 * dist + 0.5 * time


            new_g = g[u] + edge_cost

            if v not in g or new_g < g[v]:
                g[v] = new_g
                g_dist[v] = g_dist[u] + dist
                parent[v] = u
                f_cost = new_g + heuristic(v, goal)
                heappush(openh, (f_cost, v))
    return None, np.inf, np.inf, closed



blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)
start_point, goal_point = random_free_cell(blocked, min_dist=5)
path_obs, cost_obs, dist_obs, explored = astar_3d(start_point, goal_point, blocked)
empty_mask = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, dist_free, _ = astar_3d(start_point, goal_point, empty_mask)

# Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D A*: Dual Objective (Dist+Time) with Spherical No-fly Zones")
u = np.linspace(0, np.pi, 15)  # Lower sphere resolution
v = np.linspace(0, 2 * np.pi, 30)
uu, vv = np.meshgrid(u, v)
for cx, cy, cz, r in spheres_idx:
    Xs = cx + r * np.sin(uu) * np.cos(vv)
    Ys = cy + r * np.sin(uu) * np.sin(vv)
    Zs = cz + r * np.cos(uu)
    ax.plot_surface(idx_to_phys_x(Xs), idx_to_phys_y(Ys), idx_to_phys_z(Zs), alpha=0.28, linewidth=0, cmap='Reds')
if explored:
    ex = list(explored)
    if len(ex) > 1000: ex = random.sample(ex, 1000)
    ax.scatter([idx_to_phys_x(x) for x, y, z in ex], [idx_to_phys_y(y) for x, y, z in ex],
               [idx_to_phys_z(z) for x, y, z in ex],
               s=3, c='gray', alpha=0.1, label="Explored Nodes")
if path_free:
    ax.plot([idx_to_phys_x(x) for x, y, z in path_free], [idx_to_phys_y(y) for x, y, z in path_free],
            [idx_to_phys_z(z) for x, y, z in path_free],
            linestyle='--', linewidth=2, label=f"Free-space (cost={cost_free:.2f}, dist={dist_free:.2f})")
if path_obs:
    ax.plot([idx_to_phys_x(x) for x, y, z in path_obs], [idx_to_phys_y(y) for x, y, z in path_obs],
            [idx_to_phys_z(z) for x, y, z in path_obs],
            linewidth=3, label=f"With spheres (cost={cost_obs:.2f}, dist={dist_obs:.2f})")
sx, sy, sz = start_point
gx, gy, gz = goal_point
ax.scatter([idx_to_phys_x(sx)], [idx_to_phys_y(sy)], [idx_to_phys_z(sz)], s=60, marker='o', label='Start',
           color='green', depthshade=False)
ax.scatter([idx_to_phys_x(gx)], [idx_to_phys_y(gy)], [idx_to_phys_z(gz)], s=80, marker='*', label='Goal',
           color='purple', depthshade=False)
ax.set_xlabel("Longitude (deg)");
ax.set_ylabel("Latitude (deg)");
ax.set_zlabel("Flight level (ft)")
ax.legend(loc='upper left')
plt.savefig("astar_3d_dual.png", dpi=170, bbox_inches='tight')
plt.show()

# Quantify savings
# Quantify results (phrased as an increase for clarity)
if path_obs and path_free:
    increase_dist = (dist_obs - dist_free) / dist_free * 100 if dist_free > 0 else 0
    increase_cost = (cost_obs - cost_free) / cost_free * 100 if cost_free > 0 else 0
    print(f"Free-space Path: Cost={cost_free:.2f}, Distance={dist_free:.2f} grid units")
    print(f"Obstacle Path:   Cost={cost_obs:.2f} (+{increase_cost:.1f}%), Distance={dist_obs:.2f} grid units ({increase_dist:+.1f}% longer)")
