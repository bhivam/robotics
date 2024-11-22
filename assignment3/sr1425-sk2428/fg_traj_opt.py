import numpy as np # type: ignore
import gtsam # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse


def trajectory_factor_graph(start, goal, T, dt=0.1):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

 
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))
    motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2]))
    goal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))

    
    start_symbol = gtsam.symbol('x', 0)
    graph.add(gtsam.PriorFactorVector(start_symbol, np.array(start), prior_noise))
    values.insert(start_symbol, np.array(start))

    
    symbols = [start_symbol]
    for t in range(1, T + 1):
        current_symbol = gtsam.symbol('x', t)
        symbols.append(current_symbol)
        values.insert(current_symbol, np.array(start) + (np.array(goal) - np.array(start)) * t / T)
        graph.add(gtsam.BetweenFactorVector(symbols[t - 1], current_symbol, np.zeros(2), motion_noise))

    
    graph.add(gtsam.PriorFactorVector(symbols[-1], np.array(goal), goal_noise))

    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    
    trajectory = [result.atVector(symbol) for symbol in symbols]
    return trajectory


def visualize_trajectory(trajectory, start, goal):

    trajectory = np.array(trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', label='Optimized Trajectory')
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*goal, color='red', label='Goal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Basic Trajectory Optimization')
    plt.legend()
    plt.grid()
    plt.axis('equal')

    plt.savefig("trajectory.png")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, nargs=2, required=True)
    parser.add_argument("--goal", type=float, nargs=2, required=True)
    parser.add_argument("--T", type=int, required=True)
    args = parser.parse_args()

    
    trajectory = trajectory_factor_graph(tuple(args.start), tuple(args.goal), args.T)

    
    print("Optimized Trajectory:")
    for t, state in enumerate(trajectory):
        print(f"t={t}: {state}")

    
    visualize_trajectory(trajectory, tuple(args.start), tuple(args.goal))

