import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation


def generate_robot_freebody():
    w = 0.5
    h = 0.3

    theta = np.random.uniform(0, 2 * np.pi)

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

    max_x = max(rot_corners[:, 0])
    min_x = min(rot_corners[:, 0])
    max_y = max(rot_corners[:, 1])
    min_y = min(rot_corners[:, 1])

    x_buf = (max_x - min_x) / 2
    y_buf = (max_y - min_y) / 2

    x = np.random.uniform(x_buf, 20 - x_buf)
    y = np.random.uniform(y_buf, 20 - y_buf)

    rot_corners += np.array([x, y])

    return {
        "center": (x, y),
        "width": w,
        "height": h,
        "theta": theta,
        "corners": rot_corners.tolist(),
    }


# we are going to use seprability by axis theorem
def check_collision(corners1, corners2):
    corners1 = np.array(corners1)
    corners2 = np.array(corners2)

    edges1 = [corners1[i] - corners1[i - 1] for i in range(4)]
    edges2 = [corners2[i] - corners2[i - 1] for i in range(4)]

    edges = np.vstack([edges1, edges2])

    for edge in edges:
        normal = np.array([edge[1], -edge[0]])
        normal /= np.linalg.norm(normal)

        # now we have to project both polygons on the normal

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


def check_collision_environment(environment, robot):
    for obstacle in environment["obstacles"]:
        obstacle["collision"] = check_collision(obstacle["corners"], robot["corners"])


def visualize_scene(environment, filename=None):
    fig, ax = plt.subplots()

    ax.set_xlim(0, environment["dimensions"][0])
    ax.set_ylim(0, environment["dimensions"][1])

    robot = generate_robot_freebody()
    corners = robot["corners"]
    robot_patch = Polygon(
        corners, closed=True, edgecolor="red", facecolor="green", alpha=0.6
    )
    ax.add_patch(robot_patch)

    x, y = robot["center"]
    ax.plot(x, y, "ro")

    check_collision_environment(environment, robot)

    for obstacle in environment["obstacles"]:
        corners = obstacle["corners"]

        if obstacle["collision"]:
            obstacle["polygon"] = Polygon(
                corners, closed=True, edgecolor="red", facecolor="red", alpha=0.6
            )
        else:
            obstacle["polygon"] = Polygon(
                corners, closed=True, edgecolor="blue", facecolor="cyan", alpha=0.6
            )

        ax.add_patch(obstacle["polygon"])

        x, y = obstacle["center"]
        ax.plot(x, y, "ro")

    def update(num):
        # skip fram 0
        if num == 0:
            return []

        # skip frame if num is even
        if num % 2 == 0:
            return []

        robot = generate_robot_freebody()
        robot_patch.set_xy(robot["corners"])

        check_collision_environment(environment, robot)

        obstacles = []

        for obstacle in environment["obstacles"]:
            if obstacle["collision"]:
                obstacle["polygon"].set_edgecolor("red")
                obstacle["polygon"].set_facecolor("red")
            else:
                obstacle["polygon"].set_edgecolor("blue")
                obstacle["polygon"].set_facecolor("cyan")
            obstacles.append(obstacle["polygon"])

        return [robot_patch, *obstacles]

    ax.set_aspect("equal")
    plt.grid(True)
    plt.title(f"Environment with {len(environment['obstacles'])} Obstacles")

    ani = animation.FuncAnimation(fig, update, frames=100, blit=True)

    ani.save("test.gif", writer=PillowWriter(fps=1))


def scene_from_file(filename: str):
    file = open(filename, "r")
    environment = json.load(file)
    file.close()
    return environment

if __name__ == "__main__":
    environment = scene_from_file("environments/100_obstacles_env.json")
    visualize_scene(environment)
