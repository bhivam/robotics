import gtsam
import numpy as np
import matplotlib.pyplot as plt
import argparse


def trajectory_2link_arm_rot2(start, goal, T, dt=0.1):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Strong prior noise model
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.001)  # Enforce initial angles strongly
    motion_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.2 * dt)  # Smooth motion
    goal_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)  # Reasonable goal tolerance

    symbols_theta1 = [gtsam.symbol('x', t) for t in range(T + 1)]
    symbols_theta2 = [gtsam.symbol('y', t) for t in range(T + 1)]

    # Add prior factors for the start state
    graph.add(gtsam.PriorFactorRot2(symbols_theta1[0], gtsam.Rot2.fromAngle(start[0]), prior_noise))
    graph.add(gtsam.PriorFactorRot2(symbols_theta2[0], gtsam.Rot2.fromAngle(start[1]), prior_noise))

    # Ensure initial values match the start exactly
    values.insert(symbols_theta1[0], gtsam.Rot2.fromAngle(start[0]))
    values.insert(symbols_theta2[0], gtsam.Rot2.fromAngle(start[1]))

    for t in range(1, T + 1):
        current_symbol_theta1 = symbols_theta1[t]
        current_symbol_theta2 = symbols_theta2[t]
        previous_symbol_theta1 = symbols_theta1[t - 1]
        previous_symbol_theta2 = symbols_theta2[t - 1]

        # Set initial guesses for intermediate states
        initial_guess_theta1 = gtsam.Rot2.fromAngle(start[0] + (goal[0] - start[0]) * t / T)
        initial_guess_theta2 = gtsam.Rot2.fromAngle(start[1] + (goal[1] - start[1]) * t / T)

        values.insert(current_symbol_theta1, initial_guess_theta1)
        values.insert(current_symbol_theta2, initial_guess_theta2)

        # Add motion factors
        graph.add(gtsam.BetweenFactorRot2(previous_symbol_theta1, current_symbol_theta1, gtsam.Rot2(), motion_noise))
        graph.add(gtsam.BetweenFactorRot2(previous_symbol_theta2, current_symbol_theta2, gtsam.Rot2(), motion_noise))

    # Add prior factors for the goal state
    graph.add(gtsam.PriorFactorRot2(symbols_theta1[-1], gtsam.Rot2.fromAngle(goal[0]), goal_noise))
    graph.add(gtsam.PriorFactorRot2(symbols_theta2[-1], gtsam.Rot2.fromAngle(goal[1]), goal_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    trajectory = [(result.atRot2(symbols_theta1[t]).theta(), result.atRot2(symbols_theta2[t]).theta()) for t in range(T + 1)]
    return trajectory


def visualize_2link_arm(trajectory, link_length=1.0):
    plt.figure(figsize=(10, 8))
    plt.title("2-Link Robot Arm Trajectory with Rot2")
    plt.xlabel("X")
    plt.ylabel("Y")

    for i, (theta1, theta2) in enumerate(trajectory):
        x1, y1 = link_length * np.cos(theta1), link_length * np.sin(theta1)
        x2 = x1 + link_length * np.cos(theta1 + theta2)
        y2 = y1 + link_length * np.sin(theta1 + theta2)

        if i == 0:
            plt.plot([0, x1, x2], [0, y1, y2], '-o', label='Start Configuration', color='green')
        elif i == len(trajectory) - 1:
            plt.plot([0, x1, x2], [0, y1, y2], '-o', label='Goal Configuration', color='red')
        else:
            plt.plot([0, x1, x2], [0, y1, y2], '-o', color='blue', alpha=0.3)

    plt.grid()
    plt.axis('equal')
    plt.legend()

    plt.savefig("2link_arm_rot2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Link Robot Arm Optimization with Rot2")
    parser.add_argument("--start", type=float, nargs=2, required=True, help="Start joint angles (θ1, θ2) in radians")
    parser.add_argument("--goal", type=float, nargs=2, required=True, help="Goal joint angles (θ1, θ2) in radians")
    parser.add_argument("--T", type=int, required=True, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step duration")
    args = parser.parse_args()

    trajectory = trajectory_2link_arm_rot2(tuple(args.start), tuple(args.goal), args.T, args.dt)

    print("Optimized Trajectory:")
    for t, state in enumerate(trajectory):
        print(f"t={t}: θ1={state[0]:.4f}, θ2={state[1]:.4f}")

    visualize_2link_arm(trajectory)

