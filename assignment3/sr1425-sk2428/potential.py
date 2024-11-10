import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse
import json
import matplotlib.animation as animation # type: ignore
from matplotlib.animation import  FFMpegWriter # type: ignore
from matplotlib.patches import Polygon as MplPolygon # type: ignore
import time

ALPHA = 0.06
MAX_DISTANCE = 7.0
REPULSIVE_COEFF = 8
ATTRACTIVE_COEFF = 0.5

def load_map(filename):
    file = open(filename, "r")
    environment = json.load(file)
    file.close()
    return environment["obstacles"]

def attractive_gradient(config, goal):
    return 2 * ATTRACTIVE_COEFF * (config - goal)

def repulsive_gradient(config, obstacles):
    grad = np.zeros_like(config)
    for obs in obstacles:
        dist = np.linalg.norm(config - obs)
        if dist < MAX_DISTANCE:
            grad += -2 * REPULSIVE_COEFF*(config - obs) * (1 / dist - 1 / MAX_DISTANCE) / dist**3
    return grad

def potential_gradient(config, goal, obstacles):
    return attractive_gradient(config, goal) + repulsive_gradient(config, obstacles)

def configurations_intersect_obstacles(config, obstacles):
        
    def check_collision(corners1, corners2):
        corners1 = np.array(corners1)
        corners2 = np.array(corners2)

        edges1 = [corners1[i] - corners1[i - 1] for i in range(4)]
        edges2 = [corners2[i] - corners2[i - 1] for i in range(4)]

        edges = np.vstack([edges1, edges2])

        for edge in edges:
            normal = np.array([edge[1], -edge[0]])
            normal /= np.linalg.norm(normal)

            minA = np.inf
            maxA = -np.inf

            for corner in np.array(corners1):
                projection = np.dot(corner, normal)

                minA = np.min([projection, minA])
                maxA = np.max([projection, maxA])

            minB = np.inf
            maxB = -np.inf

            for corner in np.array(corners2):
                projection = np.dot(corner, normal)

                minB = np.min([projection, minB])
                maxB = np.max([projection, maxB])

            if maxA < minB or maxB < minA:
                return False

        return True
    
    corners1 = [[config[0]-0.25, config[1]-0.15],
                [config[0]-0.25, config[1]+0.15],
                [config[0]+0.25, config[1]-0.15],
                [config[0]+0.25, config[1]+0.15]]
    for obstacle in obstacles:
        if check_collision(obstacle["corners"], corners1):
            return True
    return False

def path_collision_check(node, next_node, obstacles):

    def do_intersect(p1, q1, p2, q2):

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2

        def on_segment(p, q, r):
            if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
                return True
            return False

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True

        if o2 == 0 and on_segment(p1, q2, q1):
            return True

        if o3 == 0 and on_segment(p2, p1, q2):
            return True

        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    for obstacle in obstacles:
        c1, c2, c3, c4 = obstacle['corners']

        edges = [(c1,c2), (c1,c3), (c1,c4), (c2,c3), (c2,c4), (c3,c4)]

        for edge in edges:
            if do_intersect(node[:2], next_node[:2], edge[0], edge[1]):
                return True
    
    return False

def check_collision(config, config_new, obstacles):
    return path_collision_check(config,config_new, obstacles) or configurations_intersect_obstacles(config, obstacles)

def gradient_descent(start, goal, obstacles_centers, obstacles):
    config = np.array(start, dtype=float)
    path = [config.copy()]
    iteration = 0
    
    while np.linalg.norm(potential_gradient(config, goal, obstacles_centers)) > 1e-3:
        grad = potential_gradient(config, goal, obstacles_centers)
        config_new = config - ALPHA * grad
        if not check_collision(config,config_new, obstacles):
            path.append(config_new.copy())
            print(config_new)
            iteration += 1
            config = config_new
        else:
            break
    
    return np.array(path)

def visualize_path_freebody(path, obstacles, start, goal,obstacle_centers, name="visualize.mp4"):
        figure, axes = plt.subplots()

        obstacles_corners = []

        for obs in obstacles:
            corners = obs["corners"]
            obstacles_corners.append(corners)

        for polygon in obstacles_corners:
            poly = np.array(polygon)
            mpl_poly = MplPolygon(poly, closed=True, color='gray')
            axes.add_patch(mpl_poly)

        axes.set_xlim([0, 20])
        axes.set_ylim([0, 20])

        obstacle_centers_x = list(map(lambda o: o[0], obstacle_centers))
        obstacle_centers_y = list(map(lambda o: o[1], obstacle_centers))

        plt.scatter([start[0], goal[0]], [start[1], goal[1]], [10,10],marker="D", zorder=2, c="blue")
        plt.scatter(obstacle_centers_x, obstacle_centers_y)

        for i in range(len(path)-1):
            prev = path[i]
            next = path[i+1]
            plt.plot([prev[0], next[0]], [prev[1], next[1]], c ="red", zorder=3)
    
        length, width = (0.5, 0.3)
    
        robot = plt.Rectangle((length/2, width/2), length, width, fill=True, color='b')
        axes.add_patch(robot)
    
        def update(num):
            x, y = path[num]
            robot.set_xy([x - length/2, y - width/2])
            return robot,
    
        ani = animation.FuncAnimation(figure, update, frames=len(path), interval=10, blit=True)
    
        ani.save(name, writer=FFMpegWriter(fps=1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, nargs=2)
    parser.add_argument("--goal", type=float, nargs=2)
    args = parser.parse_args()
    
    start = np.array(args.start)
    goal = np.array(args.goal)
    

    obstacles = load_map("./environments/25_obstacles_env.json")

    obstacle_centers = list(map(lambda o: o["center"], obstacles))
    
    start_time = time.time()
    path = gradient_descent(start, goal, obstacle_centers, obstacles)
    elapsed = time.time() - start_time
    print(elapsed)
    visualize_path_freebody(path, obstacles, start, goal, obstacle_centers, "100_obstacles_(17,17)_(2,2).mp4")
