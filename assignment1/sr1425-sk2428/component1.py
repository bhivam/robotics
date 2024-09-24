import numpy as np # type: ignore

def check_SOn(m, epsilon = 0.01):
    isSOn = True
    
    if m.shape[0] != m.shape[1]:
        isSOn = False

    identity = np.dot(m, m.T)
    
    if  not np.allclose(identity, np.eye(m.shape[0]), atol=epsilon):
        isSOn = False

    if not np.abs(np.linalg.det(m) - 1) <= epsilon:
        isSOn = False

    return isSOn

def check_quaternion(v, epsilon = 0.01):
    isQuat = True
    
    if v.shape != [4,]:
        isQuat = False

    norm = np.linalg.norm(v)
    if np.abs(norm -1) > epsilon:
        isQuat = False

    return isQuat

def check_SEn(m, epsilon = 0.01):
    isSEn = True
    
    if m.shape[0] != m.shape[1] | (m.shape == 3 ^ m.shape == 4):
        isSEn = False

    rotation_matrix = m[:m.shape[0]-1,:m.shape[1]-1]
    if not (check_SOn(rotation_matrix, epsilon=epsilon)):
        isSEn = False
    
    last_line = m[m.shape[0]-1]
    if not (np.compare(last_line, [0,0,1]) ^ np.compare(last_line, [0,0,0,1])):
        isSEn = False

    return isSEn

def correct_SOn(m, epsilon = 0.01):
    U, s, Vt = np.linalg.svd(m)

    m = np.dot(U, Vt)

    if check_SOn(m, epsilon=epsilon):
        return m

def correct_quaternion(v, epsilon = 0.01):
    norm_v = np.linalg.norm(v)

    v = v / norm_v

    if check_quaternion(v, epsilon=epsilon):
        return v

def correct_SEn(m, epsilon = 0.01):
    rotation = m[:m.shape[0] - 1, :m.shape[0] - 1]

    corrected_rotation = correct_SOn(rotation, epsilon)

    np.hstack(corrected_rotation, m[0,:])

    m = np.vstack(corrected_rotation, [0,0,0,1])

    return m

#__name__
if __name__=="__main__":
   result = check_SOn(np.array([[1,2], [3,4]]))
   print(result)