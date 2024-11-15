from functools import partial
import numpy as np
import gtsam
from typing import List, Optional
import argparse

# "True" function with its respective parameters
def f(x, a=0.045, b = 0.2, c =0.7, d=4.86):
    return a * x**3 + b * x**2 + c * x + d

def error_func(y: np.ndarray, x: np.ndarray, this: gtsam.CustomFactor, v:
    gtsam.Values, H: List[np.ndarray]):
    """
    :param y: { Given data point at x: y = f(x) }
    :type y: { array of one element }
    :param x: { Value that produces y for some function f: y = f(x) }
    :type x: { Array of one element }
    :param this: The factor
    :type this: { CustomFactor }
    :param v: { Set of Values, accessed via a key }
    :type v: { Values }
    :param H: { List of Jacobians: dErr/dInput. The inputs of THIS
    factor (the values) }
    :type H: { List of matrices }
    """
    # First, get the keys associated to THIS factor. The keys are in the same order as when the factor is constructed
    key_a = this.keys()[0]
    key_b = this.keys()[1]
    key_c = this.keys()[2]
    key_d = this.keys()[3]
    # Access the values associated with each key. Useful function include: tDouble, atVector, atPose2, atPose3...
    a = v.atDouble(key_a)
    b = v.atDouble(key_b)
    c = v.atDouble(key_c)
    d = v.atDouble(key_d)
    # Compute the prediction (the function h(.))
    yp = a * x[0]**3 + b * x[0]**2 + c * x[0] + d
    # Compute the error: H(.) - zi. Notice that zi here is "fixed" per factor
    error = yp - y[0]
    # For comp. efficiency, only compute jacobians when requested
    if H is not None:
    # GTSAM always expects H[i] to be matrices. For this simple problem, each J is a 1x1 matrix
        H[0] = np.array([[x[0]**3]])  # derr / da
        H[1] = np.array([[x[0]**2]])  # derr / db
        H[2] = np.array([[x[0]]])    # derr / dc
        H[3] = np.array([[1]])     # derr / dd
    return np.array([error])

if __name__ == "__main__":
    graph = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    T = 100
    GT = [] # The ground truth, for comparison
    Z = [] # GT + Normal(0, Sigma)
    # The initial guess values
    # These need to be taken from the input
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=float, nargs=4)
    parser.add_argument("--goal", type=float, nargs=2)
    args = parser.parse_args()
    a, b, c, d = args.initial
    # Create the key associated to m
    ka = gtsam.symbol('a', 0)
    kb = gtsam.symbol('b', 0)
    kc = gtsam.symbol('c', 0)
    kd = gtsam.symbol('d', 0)
    # Insert the initial guess of each key
    v.insert(ka, a)
    v.insert(kb, b)
    v.insert(kc, c)
    v.insert(kd, d)
    # Create the \Sigma (a n x n matrix, here n=1)
    sigma = 1
    noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
    for i in range(T):
        GT.append(f(i))
        Z.append(f(i) + np.random.normal(0.0, sigma)) # Produce the noisy data
        # This are the keys associate to each factor.
        # Notice that for this simple example, the keys do not depend on T, but this may not always be the case
        keys = gtsam.KeyVector([ka, kb, kc, kd])
        # Create the factor:
        # Noise model - The Sigma associated to the factor
        # Keys - The keys associated to the neighboring Variables of the factor
        # Error function - The function that computes the error: h(.) - z
        # The function expected by CustomFactor has the signature
        # F(this: gtsam.CustomFactor, v: gtsam.Values, H: List[np.ndarray])
        # Because our function has more parameters (z and i), we need to *fix* this
        # which can be done via partial.
        gf = gtsam.CustomFactor(noise_model, keys, partial(error_func, np.array([Z[i]]), np.array([i]) ))
        # add the factor to the graph.
        graph.add(gf)
    # Construct the optimizer and call with default parameters
    result = gtsam.LevenbergMarquardtOptimizer(graph, v).optimize()
    # We can print the graph, the values and evaluate the graph given some values:
    # result.print()
    # graph.print()
    # graph.printErrors(result)
    # Query the resulting values for m and b
    a = result.atDouble(ka)
    b = result.atDouble(kb)
    c = result.atDouble(kc)
    d = result.atDouble(kd)
    print("a: ", a, " b: ", b, "c: ", c, " d: ", d)
    # Print the data for plotting.
    # Should be further tested that the resulting m, b actually fit the data
    for i in range(T):
        print(i, GT[i], Z[i])