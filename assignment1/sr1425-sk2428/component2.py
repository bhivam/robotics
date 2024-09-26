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
    
    theta = 2 * np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    x = np.random.random()

    V = [np.cos(phi) * np.sqrt(x), np.cos(theta) * np.sqrt(x), np.sqrt(1-x)]

    temp = 2 * np.matmul(V, np.transpose(V))
    temp2 = np.subtract(temp, np.eye(3))
    result = np.matmul(temp2, [[np.cos(theta), np.sin(theta), 0],
                               [-np.sin(theta), np.cos(theta), 0],
                               [0,0,1]])
    return result

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

def visualize_rotation(rotation_mat):
    v0 = [0,0,1]

    x_values = []
    y_values = []
    z_values = []

    for i in range(300):
        v1 = np.add([0,np.random.random(), 0], v0)
        v0_prime = np.matmul(rotation_mat, v0)
        v1_prime = np.matmul(rotation_mat, np.subtract(v1,v0))
        x, y, z = np.subtract(v0_prime,v1_prime)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)


    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x_values, y_values, z_values)

    fig.savefig("visualize.png")


#__name__
if __name__=="__main__":
    R = random_rotation_matrix(True)

    visualize_rotation(R)