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
