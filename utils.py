import heapq
import random
from values import *

def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


#A*算法
def a_star_search(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < MAP_SIZE and 0 <= neighbor[1] < MAP_SIZE):
                continue
            if any(math.hypot(neighbor[0] - ox, neighbor[1] - oy) <= radius for ox, oy, radius in obstacles):
                continue
            tentative_g_score = g_score[current] + (DIAGONAL_COST if dx != 0 and dy != 0 else STRAIGHT_COST)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


# 人工势场法实现1
def potential_field_planning(current, goal, obstacles, uavs, uav_index, use_potential_field=False, path=None,
                             path_index=0):
    if use_potential_field:
        target = goal
    else:
        if path and path_index < len(path):
            target = path[path_index]
        else:
            target = goal

    fx, fy = 0.0, 0.0
    dx, dy = target[0] - current[0], target[1] - current[1]
    distance_to_target = math.sqrt(dx ** 2 + dy ** 2)
    if distance_to_target > 0:
        fx += ATTRACTIVE_WEIGHT * dx / distance_to_target
        fy += ATTRACTIVE_WEIGHT * dy / distance_to_target

    if use_potential_field:
        for ox, oy, radius in obstacles:
            dx, dy = current[0] - ox, current[1] - oy
            distance_to_obstacle = math.sqrt(dx ** 2 + dy ** 2)
            if distance_to_obstacle < INFLUENCE_RADIUS and distance_to_obstacle > 0:
                repulsion = REPULSIVE_WEIGHT * (1.0 / distance_to_obstacle - 1.0 / INFLUENCE_RADIUS) / (
                            distance_to_obstacle ** 2)
                fx += repulsion * dx / distance_to_obstacle
                fy += repulsion * dy / distance_to_obstacle

        for i, other_uav in enumerate(uavs):
            if i != uav_index:
                dx, dy = current[0] - other_uav[0], current[1] - other_uav[1]
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < INFLUENCE_RADIUS and dist > 0:
                    repulsion = REPULSIVE_WEIGHT * (1.0 / dist - 1.0 / INFLUENCE_RADIUS) / (dist ** 2)
                    fx += repulsion * dx / dist
                    fy += repulsion * dy / dist

    # 检测合力大小
    force_magnitude = math.sqrt(fx ** 2 + fy ** 2)

    # 平衡检测模块
    if use_potential_field and force_magnitude < MIN_FORCE_THRESHOLD:
        print(f"UAV {uav_index} detected force balance at ({current[0]:.2f}, {current[1]:.2f}), escaping...")
        # 随机选择左或右
        direction = random.choice([-1, 1])  # -1: 左, 1: 右
        # 计算当前航向（目标方向）
        heading_x, heading_y = target[0] - current[0], target[1] - current[1]
        heading_norm = math.sqrt(heading_x ** 2 + heading_y ** 2)
        if heading_norm > 0:
            heading_x, heading_y = heading_x / heading_norm, heading_y / heading_norm
            # 计算垂直于航向的向量
            perpendicular_x, perpendicular_y = -heading_y * direction, heading_x * direction
            # 移动一段距离以摆脱平衡
            next_x = current[0] + perpendicular_x * ESCAPE_DISTANCE
            next_y = current[1] + perpendicular_y * ESCAPE_DISTANCE
        else:
            next_x, next_y = current[0], current[1]  # 如果无航向，默认不动
    else:
        # 正常移动
        next_x = current[0] + fx
        next_y = current[1] + fy
        step_size = math.sqrt((next_x - current[0]) ** 2 + (next_y - current[1]) ** 2)
        if step_size > MAX_STEP_SIZE:
            scale = MAX_STEP_SIZE / step_size
            next_x = current[0] + fx * scale
            next_y = current[1] + fy * scale

    next_x = max(0, min(MAP_SIZE - 1, int(round(next_x))))
    next_y = max(0, min(MAP_SIZE - 1, int(round(next_y))))

    print(
        f"UAV {uav_index}: Pos = ({current[0]:.2f}, {current[1]:.2f}), Force = ({fx:.2f}, {fy:.2f}), Next = ({next_x:.2f}, {next_y:.2f}), Potential = {use_potential_field}")

    return (next_x, next_y)


# 人工势场法实现2
def potential_field_planning_formation(current, goal, obstacles, uavs, uav_index):
    if distance(current, goal) <= FORMATION_THRESHOLD:
        return current

    fx, fy = 0.0, 0.0
    dx, dy = goal[0] - current[0], goal[1] - current[1]
    distance_to_goal = math.sqrt(dx ** 2 + dy ** 2)
    if distance_to_goal > 0:
        fx += ATTRACTIVE_WEIGHT * dx / distance_to_goal
        fy += ATTRACTIVE_WEIGHT * dy / distance_to_goal

    for ox, oy, radius in obstacles:
        dx, dy = current[0] - ox, current[1] - oy
        distance_to_obstacle = math.sqrt(dx ** 2 + dy ** 2)
        if distance_to_obstacle < INFLUENCE_RADIUS and distance_to_obstacle > 0:
            repulsion = REPULSIVE_WEIGHT * (1.0 / distance_to_obstacle - 1.0 / INFLUENCE_RADIUS) / (
                        distance_to_obstacle ** 2)
            fx += repulsion * dx / distance_to_obstacle
            fy += repulsion * dy / distance_to_obstacle

    for i, other_uav in enumerate(uavs):
        if i != uav_index:
            dx, dy = current[0] - other_uav[0], current[1] - other_uav[1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < INFLUENCE_RADIUS and dist > 0:
                repulsion = (REPULSIVE_WEIGHT / 4) * (1.0 / dist - 1.0 / INFLUENCE_RADIUS) / (dist ** 2)
                fx += repulsion * dx / dist
                fy += repulsion * dy / dist

    next_x = current[0] + fx
    next_y = current[1] + fy
    step_size = math.sqrt((next_x - current[0]) ** 2 + (next_y - current[1]) ** 2)

    if step_size > MAX_STEP_SIZE:
        scale = MAX_STEP_SIZE / step_size
        next_x = current[0] + fx * scale
        next_y = current[1] + fy * scale

    next_x = max(0, min(MAP_SIZE - 1, int(round(next_x))))
    next_y = max(0, min(MAP_SIZE - 1, int(round(next_y))))

    print(
        f"UAV {uav_index}: Pos = ({current[0]:.2f}, {current[1]:.2f}), Force = ({fx:.2f}, {fy:.2f}), Next = ({next_x:.2f}, {next_y:.2f})")

    return (next_x, next_y)


# 动态障碍物更新
def update_obstacles(obstacles):
    updated_obstacles = []
    for ox, oy, radius, vx, vy in obstacles:
        new_x = ox + vx
        new_y = oy + vy
        new_x = max(0, min(MAP_SIZE - 1, new_x))
        new_y = max(0, min(MAP_SIZE - 1, new_y))
        updated_obstacles.append((new_x, new_y, radius, vx, vy))
    return updated_obstacles


# 检查障碍物是否在后方
def all_obstacles_behind(uavs, goals, obstacles):
    for i, uav in enumerate(uavs):
        dir_x, dir_y = goals[i][0] - uav[0], goals[i][1] - uav[1]
        dir_norm = math.sqrt(dir_x ** 2 + dir_y ** 2)
        if dir_norm == 0:
            continue
        dir_x, dir_y = dir_x / dir_norm, dir_y / dir_norm

        for ox, oy, _ in obstacles:
            obst_x, obst_y = ox - uav[0], oy - uav[1]
            dot_product = obst_x * dir_x + obst_y * dir_y
            if dot_product >= 0:
                return False
    return True

