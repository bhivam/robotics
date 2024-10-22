import argparse
import numpy as np # type: ignore
import json
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Polygon as MplPolygon, Circle # type: ignore
import heapq
import matplotlib.animation as animation # type: ignore
from matplotlib.animation import  FFMpegWriter # type: ignore
import math
from datetime import datetime

def load_map(filename):
    file = open(filename, "r")
    environment = json.load(file)
    file.close()
    return environment["obstacles"], environment["dimensions"]

def get_xy_arm_coordinates(config):
        angle_1, angle_2 = config

        width = 0.4

        return [
            (width / 2) * np.cos(angle_1),
            (width / 2) * np.sin(angle_1),
            2 * np.cos(angle_1) + (width / 2) * np.cos(angle_2),
            2 * np.sin(angle_1) + (width / 2) * np.sin(angle_2),
        ]

def get_distance(robot, origin, config):
    if robot == 'arm':
        xy_coordinates = get_xy_arm_coordinates(origin)
        xy_config_coordinates = get_xy_arm_coordinates(config)
        return np.linalg.norm(
            np.array(xy_config_coordinates) - np.array(xy_coordinates)
        )
    else:
        pos_dist = np.linalg.norm(np.array(config[:2]) - np.array(origin[:2]))
        angle_diff = min(abs(config[2] - origin[2]), 2*np.pi - abs(config[2] - origin[2]))
        return pos_dist + angle_diff


def get_k_nearest_neighbors(robot, configs, origin, k):
    
    def get_configurations_from_tuple(distances):
        return [distance[1] for distance in distances]
    
    if k == 0:
        return []

    distances = []
    for config in configs:
        distance = get_distance(robot, origin, config)
        distances.append((distance, config))
    distances.sort(key=lambda x: x[0])
    return get_configurations_from_tuple(distances)[:k]
    

def random_config(robot_type):
    if robot_type == "arm":
        return [np.pi * np.random.random(), np.pi * np.random.random()]
    elif robot_type == "freeBody":
        return [20 * np.random.random(), 20 * np.random.random(), 2 * np.pi * np.random.random()]

def configurations_intersect_obstacles(config, robot_type, obstacles):

    def generate_arm(config):
        link1_w = 0.04
        link1_l = 2

        link2_w = 0.04
        link2_l = 1.5

        theta1 = config[0]

        theta2 = config[1]

        link1_corners = np.array(
            [
                [-link1_l / 2, -link1_w / 2],
                [-link1_l / 2, link1_w / 2],
                [link1_l / 2, link1_w / 2],
                [link1_l / 2, -link1_w / 2],
            ]
        )

        link2_corners = np.array(
            [
                [-link2_l / 2, -link2_w / 2],
                [-link2_l / 2, link2_w / 2],
                [link2_l / 2, link2_w / 2],
                [link2_l / 2, -link2_w / 2],
            ]
        )

        rotm1 = np.array(
            [
                [np.cos(theta1), -np.sin(theta1)],
                [np.sin(theta1), np.cos(theta1)],
            ],
        )

        rotm2 = np.array(
            [
                [np.cos(theta2), -np.sin(theta2)],
                [np.sin(theta2), np.cos(theta2)],
            ],
        )

        link1_rot_corners = np.dot(link1_corners, rotm1.T)

        link2_rot_corners = np.dot(link2_corners, rotm2.T)

        link1_rot_corners += np.array([0, 0])

        link2_rot_corners += np.array([link1_l * np.cos(theta1), link1_l * np.sin(theta1)])

        return [
            {
                "center": (0, 0),
                "width": link1_w,
                "length": link1_l,
                "theta": theta1,
                "corners": link1_rot_corners.tolist(),
            },
            {
                "center": (link1_l * np.cos(theta1), link1_l * np.sin(theta1)),
                "width": link2_w,
                "length": link2_l,
                "theta": theta2,
                "corners": link2_rot_corners.tolist(),
            },
        ]


    def generate_robot_freebody(config):
        w = 0.5
        h = 0.3

        theta = config[2]

        corners = np.array(
            [
                [-w / 2, -h / 2],
                [-w / 2, h / 2],
                [w / 2, h / 2],
                [w / 2, -h / 2],
            ]
        )

        rotm = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
        )

        rot_corners = np.dot(corners, rotm.T)

        x = config[0]
        y = config[1]

        rot_corners += np.array([x, y])

        return [
            {
                "center": (x, y),
                "width": w,
                "height": h,
                "theta": theta,
                "corners": rot_corners.tolist(),
            }
        ]
        
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

    generate_robot = None
    if robot_type == "arm":
        generate_robot = generate_arm(config)
    elif robot_type == "freeBody":
        generate_robot = generate_robot_freebody(config)
    else:
        raise ValueError("robot_type must be either 'freeBody' or 'arm'")
    
    for obstacle in obstacles:
        for box in generate_robot:
            if check_collision(obstacle["corners"], box["corners"]):
                return True
    return False

def add_edges(graph, new_node, neighbors):
    if new_node not in graph:
        graph[new_node] = []
    for neighbor in neighbors:
        graph[new_node].append(neighbor)
        if neighbor not in graph:
            graph[neighbor] = []
        graph[neighbor].append(new_node)

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
        
def prm_star(start, goal, robot_type, obstacles, num_samples=5000):
    graph = {}
    nodes = [tuple(start), tuple(goal)]

    graph[tuple(start)] = []
    graph[tuple(goal)] = []

    while len(nodes) < num_samples:
        config = random_config(robot_type)
        if not configurations_intersect_obstacles(config, robot_type, obstacles):
            nodes.append(tuple(config))

    if robot_type == 'arm':
        dim = 2
    else:
        dim = 3

    k_prm = np.exp(1+1/dim)

    for n in range(len(nodes)):
        node = nodes[n]
        k = math.ceil(k_prm*np.log(n+1))
        neighbors = get_k_nearest_neighbors(robot_type, nodes, node, k)
        for neighbor in neighbors:
            if not path_collision_check(node, neighbor, obstacles):
                add_edges(graph, tuple(node), [neighbor])
        
    return graph

def dijkstra_search(robot, start, goal, graph):
    start = tuple(start)
    goal = tuple(goal)

    priority_queue = [(0, start)]
    
    g_score = {start: 0}
    
    came_from = {}
    
    visited = set()
    
    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1]  
        
        for neighbor in graph[current_node]:

            cost = get_distance(robot, neighbor, current_node)

            tentative_g_score = current_cost + cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                came_from[neighbor] = current_node
                heapq.heappush(priority_queue, (tentative_g_score, neighbor))
    return None

def visualize_prm(graph, obstacles, start, goal, path=None, dimensions=None, robot_type="freeBody"):
    _, axes = plt.subplots(figsize=(8, 8))

    onstacles_corners = []

    for obs in obstacles:
        corners = obs["corners"]
        onstacles_corners.append(corners)

    for polygon in onstacles_corners:
        poly = np.array(polygon)
        mpl_poly = MplPolygon(poly, closed=True, color='gray')
        axes.add_patch(mpl_poly)
    
    nodes_x_values = []
    nodes_y_values = []

    for node in graph.keys():
       nodes_x_values.append(node[0])
       nodes_y_values.append(node[1])

       for neighbor in graph[node]:
           plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], c ="red", zorder=1)
           

    plt.scatter(nodes_x_values, nodes_y_values, c= "blue", zorder=2)

    plt.scatter([start[0], goal[0]], [start[1], goal[1]], [200,200],marker="D", zorder=3, c="blue")


    axes.set_xlim(0, 20)
    axes.set_ylim(0, 20)

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_title("PRM Path Planning")
    plt.grid(True)
    plt.savefig("prm_visualization.png")
    plt.close()


def visualize_path(path, obstacles, start, goal, robot, name="visualize.mp4"):

    def visualize_path_freebody(path, obstacles, start, goal, name="visualize.mp4"):
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

        plt.scatter([start[0], goal[0]], [start[1], goal[1]], [200,200],marker="D", zorder=3, c="blue")

        for i in range(len(path)-1):
            prev = path[i]
            next = path[i+1]
            plt.plot([prev[0], next[0]], [prev[1], next[1]], c ="red", zorder=2)
    
        length, width = (0.5, 0.3)
    
        robot = plt.Rectangle((length/2, width/2), length, width, fill=True, color='b')
        axes.add_patch(robot)
    
        def update(num):
            x, y, theta = path[num]
            robot.set_xy([x - length/2, y - width/2])
            robot.angle = np.degrees(theta)
            return robot,
    
        ani = animation.FuncAnimation(figure, update, frames=len(path), interval=50, blit=True)
    
        ani.save(name, writer=FFMpegWriter(fps=1))

    def visualize_path_arm(path, obstacles, start, goal, name="visualize.mp4"):
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

        width = 0.04

        ground = plt.Rectangle((-5, -0.005), 10, 0.01, fill=True, color="black")

        joint_1 = plt.Circle((0, 0), width * 1.5, fill=True, color="r")
        link_1 = plt.Rectangle((0, -width / 2), 2, width, fill=True, color="b")
        joint_2 = plt.Circle((2, 0), width * 1.5, fill=True, color="r")
        link_2 = plt.Rectangle((2, -width / 2), 1.5, width, fill=True, color="b")

        axes.add_patch(ground)
        axes.add_patch(link_1)
        axes.add_patch(link_2)
        axes.add_patch(joint_1)
        axes.add_patch(joint_2)

        plt.scatter([start[0], goal[0]], [start[1], goal[1]], [200,200],marker="D", zorder=3, c="blue")

        for i in range(len(path)-1):
            prev = path[i]
            next = path[i+1]
            plt.plot([prev[0], next[0]], [prev[1], next[1]], c ="red", zorder=2)

        def update(num):
            angle_1, angle_2 = path[num]
            link_1.set_xy(
                [
                    (width / 2) * np.cos(angle_1 - 0.5 * np.pi),
                    (width / 2) * np.sin(angle_1 - 0.5 * np.pi),
                ]
            )

            link_2.set_xy(
                [
                    2 * np.cos(angle_1) + (width / 2) * np.cos(angle_2 - 0.5 * np.pi),
                    2 * np.sin(angle_1) + (width / 2) * np.sin(angle_2 - 0.5 * np.pi),
                ]
            )

            joint_2.set_center([2 * np.cos(angle_1), 2 * np.sin(angle_1)])

            link_1.angle = np.degrees(angle_1)
            link_2.angle = np.degrees(angle_2)
            return link_1, link_2, joint_2
        
        ani = animation.FuncAnimation(
        figure, update, frames=len(path), interval=50, blit=True
        )

        ani.save(name, writer=FFMpegWriter(fps=10))
    
    
    if robot == 'arm':
        visualize_path_arm(path, obstacles, start, goal)
    else:
        visualize_path_freebody(path, obstacles, start, goal)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str)
    parser.add_argument("--start", nargs="+", type=float)
    parser.add_argument("--goal", nargs="+", type=float)
    parser.add_argument("--map", type=str)

    args = parser.parse_args()

    obstacles, dimensions = load_map(args.map)

    graph = prm_star(args.start, args.goal, args.robot, obstacles)

    path = dijkstra_search(args.robot, args.start, args.goal, graph)

    visualize_prm(graph, obstacles, args.start, args.goal, args.robot)

    visualize_path(path, obstacles, args.start, args.goal, args.robot)