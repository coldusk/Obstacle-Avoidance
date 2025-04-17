
from utils import *
import matplotlib.pyplot as plt

def main():
    start_base = (5, 5)
    goals = [(80, 90), (90, 80), (70, 70), (80, 60)]
    #初始化障碍物
    obstacles = [
        (40, 40, OBSTACLE_RADIUS, 0.3, -0.4),
        (60, 60, OBSTACLE_RADIUS, -0.5, -0.3),
        (90, 80, OBSTACLE_RADIUS, -0.4, -0.4),
        (50, 30, OBSTACLE_RADIUS, -0.2, -0.5),
        (70, 50, OBSTACLE_RADIUS, -0.3, 0.4),
        (50, 70, OBSTACLE_RADIUS, 0.4, -0.2)
    ]
    heading = (1, 1)
    uavs = [(start_base[0] + i * FORMATION_SPACING * heading[0] / math.sqrt(2),
             start_base[1] + i * FORMATION_SPACING * heading[1] / math.sqrt(2)) for i in range(NUM_UAVS)]
    static_obstacles = [(ox, oy, radius) for ox, oy, radius, _, _ in obstacles]

    plt.figure()
    plt.grid(True)
    plt.axis([0, MAP_SIZE, 0, MAP_SIZE])

    for ox, oy, radius, _, _ in obstacles:
        circle = plt.Circle((ox, oy), radius, color='b', fill=True)
        plt.gca().add_patch(circle)

    uav_points = [plt.plot(uav[0], uav[1], 'go', label=f'UAV {i}' if i == 0 else "")[0] for i, uav in enumerate(uavs)]

    # 阶段2：垂直脱离
    heading_normalized = (heading[0] / math.sqrt(2), heading[1] / math.sqrt(2))
    left_turn = (-heading_normalized[1], heading_normalized[0])
    right_turn = (heading_normalized[1], -heading_normalized[0])
    directions = [random.choice([left_turn, right_turn]) for _ in range(NUM_UAVS)]
    dispersion_targets = [(uavs[i][0] + directions[i][0] * DISPERSION_DISTANCE,
                           uavs[i][1] + directions[i][1] * DISPERSION_DISTANCE) for i in range(NUM_UAVS)]
    while any(distance(uavs[i], dispersion_targets[i]) > 1 for i in range(NUM_UAVS)):
        for i in range(NUM_UAVS):
            uavs[i] = potential_field_planning(uavs[i], dispersion_targets[i], static_obstacles, uavs, i,
                                               use_potential_field=True)
            uav_points[i].set_data([uavs[i][0]], [uavs[i][1]])
        obstacles = update_obstacles(obstacles)
        static_obstacles = [(ox, oy, radius) for ox, oy, radius, _, _ in obstacles]
        for patch in plt.gca().patches:
            patch.remove()
        for ox, oy, radius, _, _ in obstacles:
            circle = plt.Circle((ox, oy), radius, color='b', fill=True)
            plt.gca().add_patch(circle)
        plt.pause(0.2)

    # 阶段3：避障（A* 路径规划 + 人工势场法）
    paths = [a_star_search(uavs[i], goals[i], static_obstacles) for i in range(NUM_UAVS)]
    path_lines = [None] * NUM_UAVS
    path_indices = [0] * NUM_UAVS
    use_potential_field = [False] * NUM_UAVS
    for i, path in enumerate(paths):
        if path:
            px, py = zip(*path)
            line, = plt.plot(px, py, '-r', alpha=0.5, label=f'UAV {i} Path' if i == 0 else "")
            path_lines[i] = line

    step_count = 0
    reorganized = False
    while any(uavs[i] != goals[i] for i in range(NUM_UAVS)) and not reorganized:
        for i in range(NUM_UAVS):
            if not use_potential_field[i]:
                if any(distance(uavs[i], (ox, oy)) < OBSTACLE_AVOIDANCE_THRESHOLD for ox, oy, _ in static_obstacles):
                    use_potential_field[i] = True
                    print(f"UAV {i} switched to potential field at position ({uavs[i][0]:.2f}, {uavs[i][1]:.2f})")
                else:
                    while path_indices[i] < len(paths[i]) - 1 and distance(uavs[i], paths[i][path_indices[i]]) < 1:
                        path_indices[i] += 1
                    uavs[i] = potential_field_planning(uavs[i], goals[i], static_obstacles, uavs, i,
                                                       use_potential_field=False, path=paths[i],
                                                       path_index=path_indices[i])
            else:
                uavs[i] = potential_field_planning(uavs[i], goals[i], static_obstacles, uavs, i,
                                                   use_potential_field=True)
            uav_points[i].set_data([uavs[i][0]], [uavs[i][1]])

        obstacles = update_obstacles(obstacles)
        static_obstacles = [(ox, oy, radius) for ox, oy, radius, _, _ in obstacles]
        for patch in plt.gca().patches:
            patch.remove()
        for ox, oy, radius, _, _ in obstacles:
            circle = plt.Circle((ox, oy), radius, color='b', fill=True)
            plt.gca().add_patch(circle)

        step_count += 1
        if step_count % REPLAN_INTERVAL == 0:
            for i in range(NUM_UAVS):
                if not use_potential_field[i]:
                    if path_lines[i]:
                        path_lines[i].remove()
                    paths[i] = a_star_search(uavs[i], goals[i], static_obstacles)
                    path_indices[i] = 0
                    if paths[i]:
                        px, py = zip(*paths[i])
                        line, = plt.plot(px, py, '-r', alpha=0.5)
                        path_lines[i] = line

        if all_obstacles_behind(uavs, goals, static_obstacles):
            print("All obstacles are behind UAVs, starting reorganization.")
            formation_center = uavs[0]
            formation_targets = [(formation_center[0] + i * FORMATION_SPACING * heading[0] / math.sqrt(2),
                                  formation_center[1] + i * FORMATION_SPACING * heading[1] / math.sqrt(2))
                                 for i in range(NUM_UAVS)]
            while any(distance(uavs[i], formation_targets[i]) > FORMATION_THRESHOLD for i in range(NUM_UAVS)):
                for i in range(NUM_UAVS):
                    uavs[i] = potential_field_planning_formation(uavs[i], formation_targets[i], static_obstacles, uavs,
                                                                 i)
                    uav_points[i].set_data([uavs[i][0]], [uavs[i][1]])
                obstacles = update_obstacles(obstacles)
                static_obstacles = [(ox, oy, radius) for ox, oy, radius, _, _ in obstacles]
                for patch in plt.gca().patches:
                    patch.remove()
                for ox, oy, radius, _, _ in obstacles:
                    circle = plt.Circle((ox, oy), radius, color='b', fill=True)
                    plt.gca().add_patch(circle)
                plt.pause(0.2)
            reorganized = True

        plt.pause(0.2)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

