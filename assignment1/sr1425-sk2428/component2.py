import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def random_rotation_matrix(naive):
    if naive:
        x_rotation = 2 * np.pi * np.random.random()
        y_rotation = 2 * np.pi * np.random.random()
        z_rotation = 2 * np.pi * np.random.random()

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(x_rotation), -np.sin(x_rotation)],
            [0, np.sin(x_rotation), np.cos(x_rotation)]
        ])

        Ry = np.array([
            [np.cos(y_rotation), 0, np.sin(y_rotation)],
            [0, 1, 0],
            [-np.sin(y_rotation), 0, np.cos(y_rotation)]
        ])

        Rz = np.array([
        [np.cos(z_rotation), -np.sin(z_rotation), 0],
        [np.sin(z_rotation), np.cos(z_rotation), 0],
        [0, 0, 1]
        ])

        return np.matmul(Rx, np.matmul(Ry, Rz))
    
    pi2 = 2 * np.pi
    x_1 = np.random.random()
    x_2 = np.random.random()
    x_3 = np.random.random()

    R = [[np.cos(pi2 * x_1), np.sin(pi2 * x_1), 0],
         [-np.sin(pi2 * x_1), np.cos(pi2 * x_1), 0],
         [0, 0, 1]]

    v = np.array([np.cos(pi2 * x_2) * x_3 ** 0.5, np.sin(pi2 * x_2) * x_3 ** 0.5, (1-x_3)**0.5])
    v = v[:, np.newaxis]

    H = 2 * v @ np.transpose(v) - np.identity(3)

    M = -H @ R
    
    return M 

def random_quaternion(naive):
    if naive:
        v = np.random.random((4,0))
        
        norm_v = np.linalg.norm(v)

        v = v / norm_v

        return v
    
    vx = np.random.random()
    vy = np.random.random()
    theta = 2 * np.pi * np.random.random()

    return [np.cos(theta/2), vx*np.sin(theta/2), vy*np.sin(theta/2), np.sin(theta/2)]

def plot_sphere(ax):
    """Plot a sphere."""
    phi = np.linspace(0, np.pi, 1000)
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = np.outer(np.sin(phi), np.cos(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.cos(phi), np.ones_like(theta))

    ax.plot_surface(x, y, z, color='b', alpha=0.3, rstride=5, cstride=5)


def visualize_rotation(path="visualize.png", naive=False, n=100, show=True):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d') 
    plot_sphere(ax)
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 

    ax.axes.set_xlabel('x')
    ax.axes.set_ylabel('y')
    ax.axes.set_zlabel('z')

    for i in range(n):
        R = random_rotation_matrix(naive)
        v_0 = np.array([[0], [0], [1]])
        v_1 = np.array([[0], [0.01], [0]]) + v_0

        v_0_p = R @ v_0
        v_1_p = R @ (v_1 -  v_0)

        ax.quiver3D(*v_0_p, *v_1_p, length=np.linalg.norm(v_1_p-v_0_p)*10, color='green')

    if show:
        plt.show()
    fig.savefig(path)


if __name__ == "__main__":
    # visualize_rotation()
    visualize_rotation("component_2_uniform.png", False, 1000, False)
    visualize_rotation("component_2_naive.png", True, 1000, False)
    