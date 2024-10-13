import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

def generate_arm():
    link1_w = 0.04
    link1_l = 2

    link2_w = 0.04
    link2_l = 1.5

    theta1 = np.random.uniform(0, np.pi)
    link1_h = np.sin(theta1) * link1_l

    x = np.random.uniform(0, 1)

    phi = np.arccos(link1_h / 1.5)
    theta2 = (2 * x * (np.pi - phi) + ((3 / 2) * np.pi + phi)) % (2 * np.pi)

    # get corners of link1 and link2 base based on width and length and theta respectively
    # you know that the base of link2 is at the end of link1 so you can calculate the corners of link2 based on the end of link1 after you've calculated the end of link1's corners

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

    return [
        {
            "center": (x, y),
            "width": w,
            "height": h,
            "theta": theta,
            "corners": rot_corners.tolist(),
        }
    ]


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
        for box in robot:
            obstacle["collision"] = check_collision(obstacle["corners"], box["corners"])
            if obstacle["collision"]:
                break


def visualize_scene(environment, robot_type="freebody", filename=None):
    
    generate_robot = None
    if robot_type == "arm":
        generate_robot = generate_arm
    elif robot_type == "freebody":
        generate_robot = generate_robot_freebody
    else:
        raise ValueError("robot_type must be either 'freebody' or 'arm'")

    fig, ax = plt.subplots()

    ax.set_xlim(0, environment["dimensions"][0])
    ax.set_ylim(0, environment["dimensions"][1])

    robot_patches = []
    robot = generate_robot()

    for box in robot:
        corners = box["corners"]
        robot_patches.append(
            Polygon(corners, closed=True, edgecolor="red", facecolor="green", alpha=0.6)
        )
        ax.add_patch(robot_patches[-1])

        x, y = box["center"]
        ax.plot(x, y, "ro", markersize=2)

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
        ax.plot(x, y, "ro", markersize=2)
    
    def update(num):
        # skip fram 0
        if num == 0:
            return []
        
        nonlocal robot

        for box in robot:
            x, y = box["center"]
            ax.plot(x, y, marker=None)


        robot = generate_robot()
        for patch, box in zip(robot_patches, robot):
            patch.set_xy(box["corners"])
            x, y = box["center"]
            ax.plot(x, y, "ro", markersize=2)

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

        return [*robot_patches, *obstacles]

    ax.set_aspect("equal")
    plt.grid(True)
    plt.title(f"Environment with {len(environment['obstacles'])} Obstacles")

    ani = animation.FuncAnimation(fig, update, frames=10, blit=True)

    ani.save("test.gif", writer=PillowWriter(fps=1))


def scene_from_file(filename: str):
    file = open(filename, "r")
    environment = json.load(file)
    file.close()
    return environment


if __name__ == "__main__":
    environment = scene_from_file("environments/100_obstacles_env.json")
    visualize_scene(environment, robot_type="arm")
