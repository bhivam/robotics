import numpy as np # type: ignore
import gtsam # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse


def trajectory_with_intermediate_states_SE2(start, goal, T, intermediate_states):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1])) 
    motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    intermediate_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    goal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

    symbols = [gtsam.symbol("x", t) for t in range(T + 1)]

    graph.add(gtsam.PriorFactorPose2(symbols[0], gtsam.Pose2(*start), prior_noise))
    values.insert(symbols[0], gtsam.Pose2(*start))

    num_intermediates = len(intermediate_states)
    intermediate_indices = [
        T * (i + 1) // (num_intermediates + 1) for i in range(num_intermediates)
    ]

    for idx, intermediate in zip(intermediate_indices, intermediate_states):
        graph.add(
            gtsam.PriorFactorPose2(
                symbols[idx], gtsam.Pose2(*intermediate), intermediate_noise
            )
        )

    for t in range(1, T + 1):
        prev_state = np.array(start) + (np.array(goal) - np.array(start)) * (t - 1) / T
        next_state = np.array(start) + (np.array(goal) - np.array(start)) * t / T
        values.insert(symbols[t], gtsam.Pose2(*next_state))

        delta_pose = gtsam.Pose2(*((np.array(next_state) - np.array(prev_state)).tolist()))
        graph.add(gtsam.BetweenFactorPose2(symbols[t - 1], symbols[t], delta_pose, motion_noise))

    graph.add(gtsam.PriorFactorPose2(symbols[-1], gtsam.Pose2(*goal), goal_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    trajectory = [result.atPose2(symbol) for symbol in symbols]
    return trajectory


def visualize_trajectory_SE2(trajectory, start, goal, intermediate_states):
    plt.figure(figsize=(10, 10))
    plt.title("SE(2) Trajectory Optimization with Intermediate States")

    positions = np.array([[pose.x(), pose.y()] for pose in trajectory])
    plt.plot(positions[:, 0], positions[:, 1], "-o", label="Optimized Trajectory")

    plt.scatter(start[0], start[1], color="green", label="Start")
    plt.scatter(goal[0], goal[1], color="red", label="Goal")

    if intermediate_states:
        intermediates = np.array([[s[0], s[1]] for s in intermediate_states])
        plt.scatter(
            intermediates[:, 0],
            intermediates[:, 1],
            color="orange",
            label="Intermediate States",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.savefig("result_se2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, nargs=3)
    parser.add_argument("--goal", type=float, nargs=3, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--x0", type=float, nargs=3, required=False)
    parser.add_argument("--x1", type=float, nargs=3, required=False)
    args = parser.parse_args()

    intermediate_states = []
    if args.x0:
        intermediate_states.append(tuple(args.x0))
    if args.x1:
        intermediate_states.append(tuple(args.x1))

    trajectory = trajectory_with_intermediate_states_SE2(tuple(args.start), tuple(args.goal), args.T, intermediate_states)

    print("Optimized Trajectory:")
    for t, pose in enumerate(trajectory):
        print(f"t={t}: {pose}")

    visualize_trajectory_SE2(trajectory, tuple(args.start), tuple(args.goal), intermediate_states)
