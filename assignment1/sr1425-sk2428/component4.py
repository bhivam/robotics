import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from typing import List, Tuple


def interpolate_arm(
    start: Tuple[int], goal: Tuple[int]
) -> List[Tuple[int]]:  # type hint later
    angle_start_1, angle_start_2 = start
    angle_goal_1, angle_goal_2 = goal

    angle_diff_1 = angle_goal_1 - angle_start_1
    angle_diff_2 = angle_goal_2 - angle_start_2

    n = 100
    path = []
    for t_step in range(n):
        angle_1_t = angle_start_1 + (angle_diff_1 / n) * t_step
        angle_2_t = angle_start_2 + (angle_diff_2 / n) * t_step

        path.append((angle_1_t, angle_2_t))

    path.append((angle_goal_1, angle_goal_2))

    return path


def forward_propogate_arm(
    start_pose: Tuple[int], plan: List[Tuple[int]]
) -> List[Tuple[int]]:
    path = [start_pose]

    # potentially stuff this with frames proportional to velocity
    for w_1, w_2, duration in plan:
        angle_1, angle_2 = path[-1]

        angle_1 += (w_1 * duration) % (2 * np.pi)
        angle_2 += (w_2 * duration) % (2 * np.pi)

        path.append((angle_1, angle_2))

    return path


def visualize_arm_path(path: List[Tuple[int]], name="component3.gif"):
    figure, axes = plt.subplots()
    axes.set_xlim([-5, 5])
    axes.set_ylim([-1, 5])

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

    ani.save(name, writer=PillowWriter(fps=10))


if __name__ == "__main__":

    visualize_arm_path(
        interpolate_arm((np.pi / 2, np.pi * 1.5), (np.pi * (2 / 3), 1.5 * np.pi)),
        name="interpolate_arm_1.gif"
    )
    
    visualize_arm_path(
        interpolate_arm((np.pi / 2, np.pi * 1), (np.pi * (1 / 3), 1.5 * np.pi)),
        name="interpolate_arm_2.gif"
    )
    
    visualize_arm_path(
        interpolate_arm((np.pi / 2, np.pi * 1), (np.pi * (1 / 3), 1.5 * np.pi)),
        name="interpolate_arm_3.gif"
    )
    
    visualize_arm_path(
        forward_propogate_arm((0, 0), [(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1),(0.4, 2, 0.1), (0.3, 1, .2),(0.3, 1, .2),(0.3, 1, .2),(0.3, -1, .2),(-0.3, -1, .2),(-0.3, -1, .2),(-0.3, -1, .2),(-0.3, 1, .2),(-0.3, 1, .2),(-0.3, 1, .2), (5, 0, 0.1)]),
        name="forward_propogate_arm_1.gif"
    )
    
    visualize_arm_path(
        forward_propogate_arm((0, 0), [(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2),(0.4, 2, .2), (0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4),(0.1, 0.4, .4), (2, 0, 0.1)]),
        name="forward_propogate_arm_2.gif"
    )

