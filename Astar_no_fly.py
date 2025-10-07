import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

random.seed(8)
np.random.seed(8)

lon_min, lon_max = -70.0, 0.0
lat_min, lat_max = 40.0, 60.0
Nx, Ny = 150, 80
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)

def generate_circles_mask(Nx, Ny, k_min=3, k_max=6):
    k = np.random.randint(k_min, k_max+1)
    Y, X = np.ogrid[:Ny, :Nx]
    mask = np.zeros((Ny, Nx), dtype=bool)
    for _ in range(k):
        cx = np.random.randint(int(0.2*Nx), int(0.8*Nx))
        cy = np.random.randint(int(0.2*Ny), int(0.8*Ny))
        r  = np.random.randint(int(0.05*min(Nx, Ny)), int(0.13*min(Nx, Ny)))
        mask |= (X - cx)**2 + (Y - cy)**2 <= r**2
    return mask

no_fly = generate_circles_mask(Nx, Ny)

moves = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
         (1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(-1,-1,np.sqrt(2))]

def in_bounds(ix, iy): return 0 <= ix < Nx and 0 <= iy < Ny
def passable(ix, iy, mask): return not mask[iy, ix]
def corner_ok(ix, iy, nx, ny, mask):
    dx, dy = nx-ix, ny-iy
    if abs(dx)==1 and abs(dy)==1:
        if not passable(ix+dx, iy, mask): return False
        if not passable(ix, iy+dy, mask): return False
    return True
def neighbors(ix, iy, mask):
    for dx, dy, c in moves:
        nx, ny = ix+dx, iy+dy
        if in_bounds(nx, ny) and passable(nx, ny, mask) and corner_ok(ix, iy, nx, ny, mask):
            yield nx, ny, c
def heuristic(a, b):
    dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1])
    D, D2 = 1.0, np.sqrt(2)
    return D*(dx+dy) + (D2-2*D)*min(dx, dy)
def random_free(mask):
    for _ in range(10000):
        ix = np.random.randint(0, Nx); iy = np.random.randint(0, Ny)
        if passable(ix, iy, mask): return (ix, iy)
    raise RuntimeError("no free cell")
def astar(start, goal, mask):
    openh=[]; heappush(openh,(0.0,start))
    g={start:0.0}; parent={start:None}; closed=set()
    while openh:
        _, u = heappop(openh)
        if u in closed: continue
        closed.add(u)
        if u==goal:
            path=[]; cur=u
            while cur is not None: path.append(cur); cur=parent[cur]
            path.reverse(); return path, g[u], closed
        ux, uy = u
        for vx, vy, c in neighbors(ux, uy, mask):
            v=(vx,vy); new=g[u]+c
            if v not in g or new<g[v]:
                g[v]=new; parent[v]=u; heappush(openh,(new+heuristic(v,goal), v))
    return None, np.inf, closed

start = random_free(no_fly)
goal  = random_free(no_fly)

path_obs, cost_obs, explored = astar(start, goal, no_fly)

empty_mask = np.zeros_like(no_fly, dtype=bool)
path_free, cost_free, _ = astar(start, goal, empty_mask)

plt.figure(figsize=(11,6))
plt.title("A* in free space vs with weather (same 8-direction rules)")
plt.imshow(no_fly, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], alpha=0.35)

ex_lon = [lon_vals[x] for (x,y) in explored]
ex_lat = [lat_vals[y] for (x,y) in explored]
plt.scatter(ex_lon, ex_lat, s=3, alpha=0.25, label="Explored (with weather)")

if path_free:
    flon = [lon_vals[x] for x,y in path_free]
    flat = [lat_vals[y] for x,y in path_free]
    plt.plot(flon, flat, linestyle='--', linewidth=2, label=f"A* in free space (cost={cost_free:.1f})")

if path_obs:
    olon = [lon_vals[x] for x,y in path_obs]
    olat = [lat_vals[y] for x,y in path_obs]
    plt.plot(olon, olat, linewidth=3, label=f"A* with weather (cost={cost_obs:.1f})")

sx, sy = start; gx, gy = goal
plt.scatter([lon_vals[sx]],[lat_vals[sy]], s=80, marker='o', label='Start')
plt.scatter([lon_vals[gx]],[lat_vals[gy]], s=80, marker='*', label='Goal')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Longitude (deg)"); plt.ylabel("Latitude (deg)")
plt.legend(loc='upper right')

# fig_path = "/mnt/data/astar_free_vs_weather.png"
# plt.savefig(fig_path, dpi=170, bbox_inches='tight')
plt.show()

# fig_path
