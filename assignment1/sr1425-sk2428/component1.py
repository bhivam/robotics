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
    
    if v.shape != (4,):
        isQuat = False

    norm = np.linalg.norm(v)
    if np.abs(norm -1) > epsilon:
        isQuat = False

    return isQuat

def check_SEn(m, epsilon = 0.01):
    isSEn = True
    
    if m.shape[0] != m.shape[1] | (m.shape[0] == 3 ^ m.shape[0] == 4):
        isSEn = False

    rotation_matrix = m[:m.shape[0]-1,:m.shape[1]-1]
    if not (check_SOn(rotation_matrix, epsilon=epsilon)):
        isSEn = False
    
    last_line = m[m.shape[0]-1]
    if not ((m.shape[0] == 3 and np.allclose(last_line, [0,0,1])) ^ (m.shape[0] == 4 and np.allclose(last_line, [0,0,0,1]))):
        isSEn = False

    return isSEn

def correct_SOn(m, epsilon = 0.01):
    if check_SOn(m, epsilon=epsilon):
        return m
    
    U, s, Vt = np.linalg.svd(m)

    m = np.dot(U, Vt)

    return m

def correct_quaternion(v, epsilon = 0.01):
    if check_quaternion(v, epsilon=epsilon):
        return v
    
    norm_v = np.linalg.norm(v)

    v = v / norm_v

    return v

def correct_SEn(m, epsilon = 0.01):
    if check_SEn(m, epsilon=epsilon):
        return m

    rotation = m[:m.shape[0] - 1, :m.shape[0] - 1]
    corrected_rotation = correct_SOn(rotation, epsilon)

    last_column = m[:corrected_rotation.shape[0], -1].reshape(-1, 1)
    m = np.hstack((corrected_rotation, last_column))
    
    last_row = np.array([0, 0, 0, 1]) if corrected_rotation.shape[0] == 3 else np.array([0, 0, 0, 1])
    m = np.vstack((m, last_row))
    
    return m

def test_check_SOn():
    print("Testing check_SOn...")
    
    # Valid rotation matrix (3x3)
    R1 = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    print("R1 is SO(3):", check_SOn(R1))  # Expected: True

    # Invalid matrix (not orthogonal)
    R2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print("R2 is SO(3):", check_SOn(R2))  # Expected: False

    # Invalid determinant (not 1)
    R3 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 2]])
    print("R3 is SO(3):", check_SOn(R3))  # Expected: False

def test_check_quaternion():
    print("\nTesting check_quaternion...")
    
    # Valid quaternion
    q1 = np.array([0, 0, 0, 1])
    print("q1 is quaternion:", check_quaternion(q1))  # Expected: True

    # Invalid quaternion (norm not 1)
    q2 = np.array([1, 2, 3, 4])
    print("q2 is quaternion:", check_quaternion(q2))  # Expected: False

    # Invalid shape
    q3 = np.array([1, 0, 0])
    print("q3 is quaternion:", check_quaternion(q3))  # Expected: False

def test_check_SEn():
    print("\nTesting check_SEn...")
    
    # Valid SE(3) matrix
    SE1 = np.array([[0, -1, 0, 1],
                    [1, 0, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1]])
    print("SE1 is SE(3):", check_SEn(SE1))  # Expected: True

    # Invalid SE(3) (not orthogonal)
    SE2 = np.array([[1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1]])
    print("SE2 is SE(3):", check_SEn(SE2))  # Expected: False

    # Invalid last row
    SE3 = np.array([[1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [1, 0, 0, 1]])
    print("SE3 is SE(3):", check_SEn(SE3))  # Expected: False

def test_correct_SOn():
    print("\nTesting correct_SOn...")
    
    # Valid rotation matrix
    R1 = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
    print("Corrected R1:\n", correct_SOn(R1))  # Should return R1
    print("Corrected R1 validity: ", check_SOn(correct_SOn(R1)))  
    
    # Invalid rotation matrix
    R2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    print("Corrected R2: ", correct_SOn(R2))  # Should return a valid SO(3) matrix
    print("Corrected R2 validity: ", check_SOn(correct_SOn(R2)))  

def test_correct_quaternion():
    print("\nTesting correct_quaternion...")
    

    q1 = np.array([0, 0, 0, 1])
    print("Corrected q1: ", correct_quaternion(q1))
    print("Corrected q1 valididity: ", check_quaternion(correct_quaternion(q1)))


    q2 = np.array([1, 2, 3, 4])
    print("Corrected q2: ", correct_quaternion(q2))
    print("Corrected q2 valididity: ", check_quaternion(correct_quaternion(q2)))

def test_correct_SEn():
    print("\nTesting correct_SEn...")
    

    SE1 = np.array([[0, -1, 0, 1],
                    [1, 0, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1]])
    print("Corrected SE1: ", correct_SEn(SE1))  
    print("Corrected SE1 validity: ", check_SEn(correct_SEn(SE1)))  


    SE2 = np.array([[1, 2, 3, 1],
                    [4, 5, 6, 2],
                    [7, 8, 9, 3],
                    [0, 0, 0, 1]])
    print("Corrected SE2: ", correct_SEn(SE2))  
    print("Corrected SE2 validity: ", check_SEn(correct_SEn(SE2)))  

if __name__ == "__main__":
    test_check_SOn()
    test_check_quaternion()
    test_check_SEn()
    test_correct_SOn()
    test_correct_quaternion()
    test_correct_SEn()
    