import numpy as np # type: ignore
import gtsam # type: ignore
import matplotlib.pyplot as plt # type: ignore
import argparse


def trajectory_2link_arm(start, goal, T, dt=0.1):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))
    motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2]))
    goal_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))

    
    symbols = [gtsam.symbol("x", t) for t in range(T + 1)]

    
    graph.add(gtsam.PriorFactorVector(symbols[0], np.array(start), prior_noise))
    values.insert(symbols[0], np.array(start))

    
    for t in range(1, T + 1):
        current_symbol = symbols[t]
        previous_symbol = symbols[t - 1]
        guess = np.array(start) + (np.array(goal) - np.array(start)) * t / T
        values.insert(current_symbol, guess)
        graph.add(
            gtsam.BetweenFactorVector(
                previous_symbol, current_symbol, np.zeros(2), motion_noise
            )
        )

    
    graph.add(gtsam.PriorFactorVector(symbols[-1], np.array(goal), goal_noise))

    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    
    trajectory = [result.atVector(symbol) for symbol in symbols]
    return trajectory


def visualize_trajectory(trajectory, start, goal, link_length=1.0):
    plt.figure(figsize=(10, 8))
    plt.title("2-Link Robot Arm Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")

    for i, (theta1, theta2) in enumerate(trajectory):
        
        x1, y1 = link_length * np.cos(theta1), link_length * np.sin(
            theta1
        )  
        x2 = x1 + link_length * np.cos(theta1 + theta2)  
        y2 = y1 + link_length * np.sin(theta1 + theta2)


        plt.plot([0, x1, x2], [0, y1, y2], "-o", label=f"Step {i}" if i == 0 else None)


    plt.grid()
    plt.axis("equal")
    plt.legend(["Robot Arm Configuration"], loc="upper right")
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Trajectory Optimization for 2-Link Robot Arm")
    parser.add_argument("--start", type=float, nargs=2, required=True)
    parser.add_argument("--goal", type=float, nargs=2, required=True)
    parser.add_argument("--T", type=int, required=True)
    args = parser.parse_args()

    
    trajectory = trajectory_2link_arm(tuple(args.start), tuple(args.goal), args.T)

    
    print("Optimized Trajectory:")
    for t, state in enumerate(trajectory):
        print(f"t={t}: {state}")

    
    visualize_trajectory(trajectory, tuple(args.start), tuple(args.goal))
