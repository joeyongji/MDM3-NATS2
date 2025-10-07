
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import random

# random.seed(42)
np.random.seed(53)

lon_min, lon_max = -70.0, -10.0
lat_min, lat_max =  40.0,  60.0
alt_min, alt_max = 30000.0, 38000.0

Nx, Ny, Nz = 70, 45, 7
lon_vals = np.linspace(lon_min, lon_max, Nx)
lat_vals = np.linspace(lat_min, lat_max, Ny)
alt_vals = np.linspace(alt_min, alt_max, Nz)

def idx_to_phys_x(ix): return np.interp(ix, np.arange(Nx), lon_vals)
def idx_to_phys_y(iy): return np.interp(iy, np.arange(Ny), lat_vals)
def idx_to_phys_z(iz): return np.interp(iz, np.arange(Nz), alt_vals)

def generate_spherical_mask_index(Nx, Ny, Nz, k_min=3, k_max=5):
    k = np.random.randint(k_min, k_max+1)
    Z, Y, X = np.ogrid[:Nz, :Ny, :Nx]
    mask = np.zeros((Nz, Ny, Nx), dtype=bool)
    spheres_idx = []
    for _ in range(k):
        cx = np.random.randint(5, Nx-5)
        cy = np.random.randint(5, Ny-5)
        cz = np.random.randint(1, Nz-2)
        r  = np.random.randint(4, min(10, max(5, min(Nx,Ny)//5)))
        mask |= (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 <= r**2
        spheres_idx.append((cx, cy, cz, r))
    return mask, spheres_idx

blocked, spheres_idx = generate_spherical_mask_index(Nx, Ny, Nz)

moves = []
for dx in (-1,0,1):
    for dy in (-1,0,1):
        for dz in (-1,0,1):
            if dx==dy==dz==0: continue
            cost = np.sqrt(dx*dx + dy*dy + dz*dz)
            moves.append((dx,dy,dz,cost))

def in_bounds(ix, iy, iz): return 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
def passable(ix, iy, iz, mask): return not mask[iz, iy, ix]
def neighbors(ix, iy, iz, mask):
    for dx,dy,dz,c in moves:
        nx, ny, nz = ix+dx, iy+dy, iz+dz
        if in_bounds(nx,ny,nz) and passable(nx,ny,nz, mask):
            yield nx, ny, nz, c
def heuristic(a, b):
    dx=a[0]-b[0]; dy=a[1]-b[1]; dz=a[2]-b[2]
    return (dx*dx + dy*dy + dz*dz)**0.5
def random_free_cell(mask):
    for _ in range(20000):
        ix=np.random.randint(0,Nx); iy=np.random.randint(0,Ny); iz=np.random.randint(0,Nz)
        if passable(ix,iy,iz,mask): return (ix,iy,iz)
    raise RuntimeError("No free 3D cell found.")
def astar_3d(start, goal, mask):
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
        ux,uy,uz=u
        for nx,ny,nz,c in neighbors(ux,uy,uz,mask):
            v=(nx,ny,nz); new=g[u]+c
            if v not in g or new<g[v]:
                g[v]=new; parent[v]=u; heappush(openh,(new+heuristic(v,goal), v))
    return None, np.inf, closed

start = random_free_cell(blocked)
goal  = random_free_cell(blocked)
path_obs, cost_obs, explored = astar_3d(start, goal, blocked)
empty_mask = np.zeros_like(blocked, dtype=bool)
path_free, cost_free, _ = astar_3d(start, goal, empty_mask)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D A*: Spherical no-fly (INDEX) vs Free-space reference")

u = np.linspace(0, np.pi, 28); v = np.linspace(0, 2*np.pi, 56)
uu, vv = np.meshgrid(u, v)
for cx,cy,cz,r in spheres_idx:
    Xs = cx + r*np.sin(uu)*np.cos(vv)
    Ys = cy + r*np.sin(uu)*np.sin(vv)
    Zs = cz + r*np.cos(uu)
    ax.plot_surface(idx_to_phys_x(Xs), idx_to_phys_y(Ys), idx_to_phys_z(Zs), alpha=0.28, linewidth=0)

ex = list(explored)
if len(ex) > 4000: ex = random.sample(ex, 4000)
ax.scatter([idx_to_phys_x(x) for (x,y,z) in ex],
           [idx_to_phys_y(y) for (x,y,z) in ex],
           [idx_to_phys_z(z) for (x,y,z) in ex], s=3, alpha=0.25, label="Explored (with spheres)")

if path_free:
    ax.plot([idx_to_phys_x(x) for (x,y,z) in path_free],
            [idx_to_phys_y(y) for (x,y,z) in path_free],
            [idx_to_phys_z(z) for (x,y,z) in path_free],
            linestyle='--', linewidth=2, label=f"Free-space A* (cost={cost_free:.2f})")
if path_obs:
    ax.plot([idx_to_phys_x(x) for (x,y,z) in path_obs],
            [idx_to_phys_y(y) for (x,y,z) in path_obs],
            [idx_to_phys_z(z) for (x,y,z) in path_obs],
            linewidth=3, label=f"With spheres (cost={cost_obs:.2f})")

sx,sy,sz=start; gx,gy,gz=goal
ax.scatter([idx_to_phys_x(sx)],[idx_to_phys_y(sy)],[idx_to_phys_z(sz)], s=60, marker='o', label='Start')
ax.scatter([idx_to_phys_x(gx)],[idx_to_phys_y(gy)],[idx_to_phys_z(gz)], s=80, marker='*', label='Goal')

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_zlabel("Flight level (ft)")
ax.legend(loc='upper left')
plt.show()
