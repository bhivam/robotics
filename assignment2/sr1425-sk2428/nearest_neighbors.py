import argparse
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.animation as animation  # type: ignore
from matplotlib.animation import PillowWriter  # type: ignore


def get_configurations_from_tuple(distances):
    return [distance[1] for distance in distances]


def get_k_nearest_neighbors_freebody(k, configs, origin):
    distances = []
    for config in configs:
        distance = get_difference_freebody(config, origin)
        distances.append((distance, config))
    distances.sort(key=lambda x: x[0])
    return get_configurations_from_tuple(distances)[:k]

def get_difference_freebody(config, origin):
    pos_dist = np.linalg.norm(np.array(config[:2]) - np.array(origin[:2]))
    angle_diff = min(abs(config[2] - origin[2]), 2*np.pi - abs(config[2] - origin[2]))
    return pos_dist + angle_diff

def get_xy_arm_coordinates(config):
    angle_1, angle_2 = config

    width = 0.4

    return [
        (width / 2) * np.cos(angle_1),
        (width / 2) * np.sin(angle_1),
        2 * np.cos(angle_1) + (width / 2) * np.cos(angle_2),
        2 * np.sin(angle_1) + (width / 2) * np.sin(angle_2),
    ]


def get_k_nearest_neighbors_arm(k, configs, origin):
    distances = []
    xy_coordinates = get_xy_arm_coordinates(origin)
    for config in configs:
        xy_config_coordinates = get_xy_arm_coordinates(config)
        distance = np.linalg.norm(
            np.array(xy_config_coordinates) - np.array(xy_coordinates)
        )
        distances.append((distance, config))
    distances.sort(key=lambda x: x[0])
    return get_configurations_from_tuple(distances)[:k]


def get_configurations(file_name, N):
    configs = []
    with open(file_name, "r") as file:
        for line in file:
            configs.append(list(map(float, line.strip().split())))
    return configs


def visualize_arm_neighbors(target, configs, name="visualize.png"):
    figure, axes = plt.subplots()
    axes.set_xlim([-5, 5])
    axes.set_ylim([-1, 5])

    width = 0.04

    ground = plt.Rectangle((-5, -0.005), 10, 0.01, fill=True, color="black")
    axes.add_patch(ground)

    joint_1 = plt.Circle((0, 0), width * 1.5, fill=True, color="r")
    axes.add_patch(joint_1)

    angle_1_target, angle_2_target = target

    link_1_target = plt.Rectangle((0, -width / 2), 2, width, fill=True, color="r")
    joint_2_target = plt.Circle((2, 0), width * 1.5, fill=True, color="r")
    link_2_target = plt.Rectangle((2, -width / 2), 1.5, width, fill=True, color="r")

    link_1_target.set_xy(
        [
            (width / 2) * np.cos(angle_1_target),
            (width / 2) * np.sin(angle_1_target),
        ]
    )

    link_2_target.set_xy(
        [
            2 * np.cos(angle_1_target) + (width / 2) * np.cos(angle_2_target),
            2 * np.sin(angle_1_target) + (width / 2) * np.sin(angle_2_target),
        ]
    )

    joint_2_target.set_center([2 * np.cos(angle_1_target), 2 * np.sin(angle_1_target)])

    link_1_target.angle = np.degrees(angle_1_target)
    link_2_target.angle = np.degrees(angle_2_target)

    axes.add_patch(link_1_target)
    axes.add_patch(link_2_target)
    axes.add_patch(joint_2_target)

    for config in configs:

        angle_1, angle_2 = config

        link_1 = plt.Rectangle((0, -width / 2), 2, width, fill=True, color="b")
        joint_2 = plt.Circle((2, 0), width * 1.5, fill=True, color="b")
        link_2 = plt.Rectangle((2, -width / 2), 1.5, width, fill=True, color="b")

        link_1.set_xy(
            [
                (width / 2) * np.cos(angle_1),
                (width / 2) * np.sin(angle_1),
            ]
        )

        link_2.set_xy(
            [
                2 * np.cos(angle_1) + (width / 2) * np.cos(angle_2),
                2 * np.sin(angle_1) + (width / 2) * np.sin(angle_2),
            ]
        )

        joint_2.set_center([2 * np.cos(angle_1), 2 * np.sin(angle_1)])

        link_1.angle = np.degrees(angle_1)
        link_2.angle = np.degrees(angle_2)

        axes.add_patch(link_1)
        axes.add_patch(link_2)
        axes.add_patch(joint_2)

    plt.savefig(name)
    plt.close()


def visualize_robot_neighbors(target, configs, name="visualize.png"):
    figure, axes = plt.subplots()
    axes.set_xlim([-10, 10])
    axes.set_ylim([-10, 10])

    length, width = (0.5, 0.3)

    x_target, y_target, theta_target = target

    robot_target = plt.Rectangle(
        (length / 2, width / 2), length, width, fill=True, color="r"
    )
    axes.add_patch(robot_target)

    robot_target.set_xy([x_target - length / 2, y_target - width / 2])
    robot_target.angle = np.degrees(theta_target)

    for config in configs:
        x, y, theta = config

        robot = plt.Rectangle(
            (length / 2, width / 2), length, width, fill=True, color="b"
        )
        axes.add_patch(robot)

        robot.set_xy([x - length / 2, y - width / 2])
        robot.angle = np.degrees(theta)

    plt.savefig(name)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str)
    parser.add_argument("--target", nargs="+", type=float)
    parser.add_argument("-k", type=int)
    parser.add_argument("--configs", type=str)

    args = parser.parse_args()

    configs = get_configurations(args.configs, args.target)

    if args.robot == "arm":

        nearest_neighbors = get_k_nearest_neighbors_arm(args.k, configs, args.target)

        visualize_arm_neighbors(args.target, nearest_neighbors, "visualize.png")
    else:

        nearest_neighbors = get_k_nearest_neighbors_freebody(
            args.k, configs, args.target
        )

        visualize_robot_neighbors(args.target, nearest_neighbors, "visualize.png")
