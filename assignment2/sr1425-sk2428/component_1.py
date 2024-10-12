import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json


def generate_valid_obstacle():
    w = np.random.uniform(0.5, 2.0)
    h = np.random.uniform(0.5, 2.0)

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


def generate_environment(number_of_obstacles: int):
    obstacles = [generate_valid_obstacle() for _ in range(number_of_obstacles)]
    return {"dimensions": (20, 20), "obstacles": obstacles}


def scene_to_file(environment, filename: str):
    file = open(filename, "w")
    json.dump(environment, file)
    file.close()


def scene_from_file(filename: str):
    file = open(filename, "r")
    environment = json.load(file)
    file.close()
    return environment


def visualize_scene(environment, filename=None):
    _, ax = plt.subplots()

    ax.set_xlim(0, environment["dimensions"][0])
    ax.set_ylim(0, environment["dimensions"][1])

    for obstacle in environment["obstacles"]:
        corners = obstacle["corners"]
        polygon = Polygon(
            corners, closed=True, edgecolor="blue", facecolor="cyan", alpha=0.6
        )
        ax.add_patch(polygon)

        x, y = obstacle["center"]
        ax.plot(x, y, "ro")

    ax.set_aspect("equal")
    plt.grid(True)
    plt.title(f"Environment with {len(environment['obstacles'])} Obstacles")
    
    if filename is None:
        plt.show()
    else:
        print("saving to", filename)
        plt.savefig(filename)


if __name__ == "__main__":
    obstacle_progression = [3, 5, 10, 25, 100]

    for number_of_obstacles in obstacle_progression:
        env = generate_environment(number_of_obstacles)
        scene_to_file(env, f"./environments/{number_of_obstacles}_obstacles_env.json")
        visualize_scene(env, filename=f"./component_1_visualizations/{number_of_obstacles}_obstacles_env.jpg")

    
