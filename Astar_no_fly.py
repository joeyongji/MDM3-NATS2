import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random


random.seed(7)
np.random.seed(7)

lon_min, lon_max = -70.0, 0.0          #北大西洋经纬度就是这个范围 我才选择这个范围 :)
lat_min, lat_max = 40.0, 60.0

Nx, Ny = 150, 80
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)


# Multiple circular no‑fly zones
def generate_circles_mask(Nx, Ny, k_min=3, k_max=6):  #随即生成3-6个圆形禁飞区
    k = np.random.randint(k_min, k_max+1)
    Y, X = np.ogrid[:Ny, :Nx]
    mask = np.zeros((Ny, Nx), dtype=bool) #mask用来标记那些网格点在禁飞区
    circles = []
    for _ in range(k):
        cx = np.random.randint(int(0.15*Nx), int(0.85*Nx))
        cy = np.random.randint(int(0.15*Ny), int(0.85*Ny))
        r  = np.random.randint(int(0.05*min(Nx, Ny)), int(0.15*min(Nx, Ny)))
        mask |= (X - cx)**2 + (Y - cy)**2 <= r**2
        circles.append((cx, cy, r))
    return mask, circles

no_fly_mask, circles = generate_circles_mask(Nx, Ny) #最终 no_fly_mask[iy, ix] = True 表示该点不可通行


# A* helpers
moves = [
    ( 1,  0, 1.0), (-1,  0, 1.0), (0,  1, 1.0), (0, -1, 1.0),
    ( 1,  1, np.sqrt(2)), ( 1, -1, np.sqrt(2)), (-1,  1, np.sqrt(2)), (-1, -1, np.sqrt(2))
]                     #这个就是 上下左右对角线八个的移动方向和代价 对角线代价就是根号2

def in_bounds(ix, iy):
    return 0 <= ix < Nx and 0 <= iy < Ny

def passable(ix, iy):
    return not no_fly_mask[iy, ix]

def corner_ok(ix, iy, nx, ny):
    # Disallow "corner cutting": for diagonal moves, require both adjacent cells to be free
    dx, dy = nx - ix, ny - iy
    if abs(dx) == 1 and abs(dy) == 1:
        if not passable(ix + dx, iy):
            return False
        if not passable(ix, iy + dy):
            return False
    return True

def neighbors(ix, iy):
    for dx, dy, cost in moves:
        nx, ny = ix + dx, iy + dy
        if in_bounds(nx, ny) and passable(nx, ny) and corner_ok(ix, iy, nx, ny):
            yield (nx, ny, cost)

def heuristic(a, b):
    # Octile distance (admissible for 8-connected grid)
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D, D2 = 1.0, np.sqrt(2)
    return D*(dx + dy) + (D2 - 2*D)*min(dx, dy)

def random_cell():
    return (np.random.randint(0, Nx), np.random.randint(0, Ny))

def pick_valid_point():
    for _ in range(10000):
        ix, iy = random_cell()
        if passable(ix, iy):
            return (ix, iy)
    raise RuntimeError("Could not find a valid free cell.")

def astar(start, goal):
    open_heap = []
    heappush(open_heap, (0.0, start))
    g_cost = {start: 0.0}
    parent = {start: None}
    closed = set()

    while open_heap:
        _, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, g_cost[current], closed

        cx, cy = current
        for nx, ny, step_cost in neighbors(cx, cy):
            new_cost = g_cost[current] + step_cost
            nxt = (nx, ny)
            if nxt not in g_cost or new_cost < g_cost[nxt]:
                g_cost[nxt] = new_cost
                f = new_cost + heuristic(nxt, goal)
                parent[nxt] = current
                heappush(open_heap, (f, nxt))

    return None, np.inf, closed

# ===========================
# Run
# ===========================
start = pick_valid_point()
goal  = pick_valid_point()
path, total_cost, explored = astar(start, goal)

# Straight-line (octile) lower bound as a reference
straight_lower_bound = heuristic(start, goal)

# ===========================
# Visualization
# ===========================
plt.figure(figsize=(11, 6))
plt.title("A* shortest path with multiple no‑fly zones (8-direction moves)")

# 1) No-fly zones
plt.imshow(no_fly_mask, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], alpha=0.35)

# 2) Explored nodes (to show A* actually searched and found optimal)
if explored:
    ex_lon = [lon_vals[x] for (x, y) in explored]
    ex_lat = [lat_vals[y] for (x, y) in explored]
    plt.scatter(ex_lon, ex_lat, s=3, alpha=0.35, label="Explored nodes")

# 3) Draw a straight segment from start to goal (not a valid path if crossing obstacles)
sx, sy = start; gx, gy = goal
plt.plot([lon_vals[sx], lon_vals[gx]], [lat_vals[sy], lat_vals[gy]], linestyle='--', label=f"Straight line (LB≈{straight_lower_bound:.1f})")

# 4) A* path
if path is not None:
    path_lon = [lon_vals[ix] for ix, iy in path]
    path_lat = [lat_vals[iy] for ix, iy in path]
    plt.plot(path_lon, path_lat, linewidth=3, label=f"A* path (cost={total_cost:.1f})")
else:
    plt.text((lon_min+lon_max)/2, (lat_min+lat_max)/2, "No path found", ha='center', va='center')

# 5) Start/Goal markers
plt.scatter([lon_vals[sx]], [lat_vals[sy]], s=80, marker='o', label='Start')
plt.scatter([lon_vals[gx]], [lat_vals[gy]], s=80, marker='*', label='Goal')

plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.legend(loc='upper right')

# fig_path = "/mnt/data/astar_multi_nofly.png"
# plt.savefig(fig_path, dpi=170, bbox_inches='tight')
plt.show()

# fig_path
