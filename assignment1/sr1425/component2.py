import numpy as np # type: ignore

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
    x = 2 * np.pi * np.random.random()

    V = [np.cos(phi) * np.sqrt(x), np.cos(theta) * np.sqrt(x), np.sqrt(1-x)]

    temp = 2 * np.matmul(V, V.T)
    temp2 = np.subtract(temp, np.eye(3))
    result = np.matmul(temp2, [[np.cos(theta), np.sin(theta), 0],
                               [-np.sin(theta), np.cos(theta), 0],
                               [0,0,1]])
    return result

def random_quaternion(naive):
    if naive:
        return np.array()
    
    vx = np.random.random()
    vy = np.random.random()
    theta = 2 * np.pi * np.random.random()

    return np.array(np.cos(theta/2), vx*np.sin(theta/2), vy*np.sin(theta/2), np.sin(theta/2))

#__name__
if __name__=="__main__":
    print("Hello World!")