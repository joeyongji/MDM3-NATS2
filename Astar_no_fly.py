# 3D A* with (1) obstacle-aware path and (2) free-space reference path,
# and solid 3D spherical no-fly volumes rendered as surfaces.
#
# - Grid axes: longitude (x), latitude (y), altitude (z)
# - Movement: 26-neighbor (dx,dy,dz in {-1,0,1} \ {(0,0,0)})
# - Edge cost: Euclidean step length sqrt(dx^2 + dy^2 + dz^2)
# - No extra penalty for altitude changes (per user request)
# - Obstacles: multiple solid spheres in physical coordinates
#
# The plot will show:
#   * Obstacle-aware A* path (solid line)
#   * Free-space A* path (dashed line) — same 26-neighbor rules, but without obstacles
#   * Solid spheres for no-fly volumes
#   * Start/Goal markers

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

# ------------------------------
# Config
# ------------------------------
# random.seed(243)      # reproducibility
np.random.seed(4)

# Domain (for labels)
lon_min, lon_max = -70.0, -10.0
lat_min, lat_max =  40.0,  60.0
alt_min, alt_max = 30000.0, 38000.0  # feet

# Grid resolution
Nx, Ny, Nz = 60, 40, 7
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

dx = (lon_max - lon_min) / (Nx - 1) if Nx > 1 else 1.0
dy = (lat_max - lat_min) / (Ny - 1) if Ny > 1 else 1.0
dz = (alt_max - alt_min) / (Nz - 1) if Nz > 1 else 1.0

# ------------------------------
# Solid spherical no-fly volumes (in physical space)
# ------------------------------
def generate_spherical_mask(Nx, Ny, Nz, k_min=3, k_max=5):
    k = np.random.randint(k_min, k_max + 1)
    # Create full coordinate grids in physical units for fast broadcasting
    LX = lon_vals[np.newaxis, np.newaxis, :]          # shape (1,1,Nx)
    LY = lat_vals[np.newaxis, :, np.newaxis]          # shape (1,Ny,1)
    LZ = alt_vals[:, np.newaxis, np.newaxis]          # shape (Nz,1,1)

    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres = []  # store centers/radii in physical units
    for _ in range(k):
        # Random physical center within margins
        cx = np.random.uniform(lon_min + 5*dx, lon_max - 5*dx)
        cy = np.random.uniform(lat_min + 3*dy, lat_max - 3*dy)
        cz = np.random.uniform(alt_min + 1.2*dz, alt_max - 1.2*dz)
        # Reasonable radius based on domain size
        R = np.random.uniform(3*min(dx,dy), 8*min(dx,dy))  # a few grid cells wide in horizontal
        # Build boolean mask for this sphere: (x-cx)^2+(y-cy)^2+(z-cz)^2 <= R^2
        sphere = (LX - cx)**2 + (LY - cy)**2 + (LZ - cz)**2 <= R**2
        mask |= sphere
        spheres.append((cx, cy, cz, R))
    return mask, spheres

blocked, spheres = generate_spherical_mask(Nx, Ny, Nz)

# ------------------------------
# A* in 3D
# ------------------------------
moves = []
for dxs in (-1, 0, 1):
    for dys in (-1, 0, 1):
        for dzs in (-1, 0, 1):
            if dxs == dys == dzs == 0:
                continue
            cost = np.sqrt(dxs*dxs + dys*dys + dzs*dzs)
            moves.append((dxs, dys, dzs, cost))

def in_bounds(ix, iy, iz):
    return (0 <= ix < Nx) and (0 <= iy < Ny) and (0 <= iz < Nz)

def passable(ix, iy, iz, mask):
    return not mask[iz, iy, ix]

def neighbors(ix, iy, iz, mask):
    for dxs, dys, dzs, c in moves:
        nx, ny, nz = ix + dxs, iy + dys, iz + dzs
        if in_bounds(nx, ny, nz) and passable(nx, ny, nz, mask):
            yield nx, ny, nz, c

def heuristic(a, b):
    # 3D Euclidean in index space (consistent with edge costs)
    dxs = a[0] - b[0]
    dys = a[1] - b[1]
    dzs = a[2] - b[2]
    return (dxs*dxs + dys*dys + dzs*dzs) ** 0.5

def random_free_cell(mask):
    for _ in range(10000):
        ix = np.random.randint(0, Nx)
        iy = np.random.randint(0, Ny)
        iz = np.random.randint(0, Nz)
        if passable(ix, iy, iz, mask):
            return (ix, iy, iz)
    raise RuntimeError("No free 3D cell found.")

def astar_3d(start, goal, mask):
    openh = []
    heappush(openh, (0.0, start))
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    while openh:
        _, u = heappop(openh)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            # reconstruct
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, g[u], closed
        ux, uy, uz = u
        for nx, ny, nz, c in neighbors(ux, uy, uz, mask):
            v = (nx, ny, nz)
            newg = g[u] + c
            if (v not in g) or (newg < g[v]):
                g[v] = newg
                parent[v] = u
                f = newg + heuristic(v, goal)
                heappush(openh, (f, v))
    return None, np.inf, closed

# ------------------------------
# Start/Goal & Paths
# ------------------------------
start = random_free_cell(blocked)
goal  = random_free_cell(blocked)

# (1) With spheres (weather)
path_obs, cost_obs, explored = astar_3d(start, goal, blocked)

# (2) Free space reference (same rules, no spheres)
empty_mask = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, _ = astar_3d(start, goal, empty_mask)

# ------------------------------
# Visualization
# ------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D A*: with spherical no-fly volumes vs. free-space reference")

# Draw solid spheres (as surfaces)
phi = np.linspace(0, np.pi, 24)
theta = np.linspace(0, 2*np.pi, 48)
phi, theta = np.meshgrid(phi, theta)

for cx, cy, cz, R in spheres:
    X = cx + R * np.sin(phi) * np.cos(theta)
    Y = cy + R * np.sin(phi) * np.sin(theta)
    Z = cz + R * np.cos(phi)
    ax.plot_surface(X, Y, Z, alpha=0.25, linewidth=0)

# Explored nodes (subsample for clarity)
ex = list(explored)
if len(ex) > 4000:
    ex = random.sample(ex, 4000)
ax.scatter([lon_vals[x] for (x, y, z) in ex],
           [lat_vals[y] for (x, y, z) in ex],
           [alt_vals[z] for (x, y, z) in ex],
           s=3, alpha=0.2, label="Explored (with spheres)")

# Free-space A* reference (dashed line)
if path_free:
    ax.plot([lon_vals[x] for (x, y, z) in path_free],
            [lat_vals[y] for (x, y, z) in path_free],
            [alt_vals[z] for (x, y, z) in path_free],
            linestyle='--', linewidth=2, label=f"Free-space A* (cost={cost_free:.2f})")

# Obstacle-aware A* path (solid)
if path_obs:
    ax.plot([lon_vals[x] for (x, y, z) in path_obs],
            [lat_vals[y] for (x, y, z) in path_obs],
            [alt_vals[z] for (x, y, z) in path_obs],
            linewidth=3, label=f"With spheres (cost={cost_obs:.2f})")

# Start / Goal
sx, sy, sz = start; gx, gy, gz = goal
ax.scatter([lon_vals[sx]], [lat_vals[sy]], [alt_vals[sz]], s=60, marker='o', label='Start')
ax.scatter([lon_vals[gx]], [lat_vals[gy]], [alt_vals[gz]], s=80, marker='*', label='Goal')

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Flight level (ft)")
ax.legend(loc='upper left')


plt.show()

