import numpy as np
import gtsam
import matplotlib.pyplot as plt
import argparse


def trajectory_with_intermediate_states(start, goal, T, intermediate_states, dt=0.1):
    """
    Constructs a factor graph for trajectory optimization with intermediate states.

    :param start: Start state (x, y) as a tuple.
    :param goal: Goal state (x, y) as a tuple.
    :param T: Total number of time steps.
    :param intermediate_states: List of intermediate states [(x1, y1), (x2, y2), ...].
    :param dt: Time step duration.
    :return: Optimized trajectory.
    """
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))
    motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2]))
    intermediate_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))
    goal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))

    
    symbols = [gtsam.symbol("x", t) for t in range(T + 1)]

    
    graph.add(gtsam.PriorFactorVector(symbols[0], np.array(start), prior_noise))
    values.insert(symbols[0], np.array(start))

    
    num_intermediates = len(intermediate_states)
    intermediate_indices = [
        T * (i + 1) // (num_intermediates + 1) for i in range(num_intermediates)
    ]

    
    for idx, intermediate in zip(intermediate_indices, intermediate_states):
        graph.add(
            gtsam.PriorFactorVector(
                symbols[idx], np.array(intermediate), intermediate_noise
            )
        )

    
    for t in range(1, T + 1):
        values.insert(
            symbols[t], np.array(start) + (np.array(goal) - np.array(start)) * t / T
        )
        graph.add(
            gtsam.BetweenFactorVector(
                symbols[t - 1], symbols[t], np.zeros(2), motion_noise
            )
        )

    
    graph.add(gtsam.PriorFactorVector(symbols[-1], np.array(goal), goal_noise))

    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    
    trajectory = [result.atVector(symbol) for symbol in symbols]
    return trajectory


def visualize_trajectory(trajectory, start, goal, intermediate_states):
    """
    Visualizes the trajectory with intermediate states.
    """
    trajectory = np.array(trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-o", label="Optimized Trajectory")
    plt.scatter(*start, color="green", label="Start")
    plt.scatter(*goal, color="red", label="Goal")

    
    if intermediate_states:
        intermediates = np.array(intermediate_states)
        plt.scatter(
            intermediates[:, 0],
            intermediates[:, 1],
            color="orange",
            label="Intermediate States",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Optimization with Intermediate States")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Trajectory Optimization with Intermediate States"
    )
    parser.add_argument(
        "--start", type=float, nargs=2, required=True, help="Start state (x, y)"
    )
    parser.add_argument(
        "--goal", type=float, nargs=2, required=True, help="Goal state (x, y)"
    )
    parser.add_argument("--T", type=int, required=True, help="Number of time steps")
    parser.add_argument(
        "--x0",
        type=float,
        nargs=2,
        help="First intermediate state (x, y)",
        required=False,
    )
    parser.add_argument(
        "--x1",
        type=float,
        nargs=2,
        help="Second intermediate state (x, y)",
        required=False,
    )
    args = parser.parse_args()

    
    intermediate_states = []
    if args.x0:
        intermediate_states.append(tuple(args.x0))
    if args.x1:
        intermediate_states.append(tuple(args.x1))

    
    trajectory = trajectory_with_intermediate_states(
        tuple(args.start), tuple(args.goal), args.T, intermediate_states
    )

    
    print("Optimized Trajectory:")
    for t, state in enumerate(trajectory):
        print(f"t={t}: {state}")

   
    visualize_trajectory(
        trajectory, tuple(args.start), tuple(args.goal), intermediate_states
    )
