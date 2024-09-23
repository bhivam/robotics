import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
from matplotlib.animation import  PillowWriter # type: ignore

def validate_pose(pose):
    isValid = True

    x, y, _ = pose

    if x < 0.5 or x > 9.5:
        isValid = False
    
    if y < 0.5 or y > 9.5:
        isValid = False

    return isValid



def interpolate_rigid_body(start_pose, goal_pose):
    x_start, y_start, theta_start = start_pose
    x_goal, y_goal, theta_goal = goal_pose

    n = np.random.randint(10)
    
    path = []
    for t in range(n):
        xt = x_start + (x_goal - x_start)/n *t
        yt = y_start + t * (y_goal - y_start)/n
        thetat = theta_start + t * (theta_goal - theta_start)/n

        if validate_pose((xt, yt, thetat)):
            path.append((xt, yt, thetat))
        else:
            return

    path.append((x_goal, y_goal, theta_goal))
    
    return path


def forward_propagate_rigid_body(start_pose, plan):
    x, y, theta = start_pose
    path = [start_pose]

    for (v_x, v_y, v_theta, duration) in plan:
        dx = v_x * np.cos(theta) - v_y * np.sin(theta)
        dy = v_x * np.sin(theta) + v_y * np.cos(theta)

        x += dx * duration
        y += dy * duration
        theta += v_theta * duration
        
        if validate_pose((x, y, theta)):
            path.append((x, y, theta))
        else:
            return

    return path


def visualize_path(path):
    


#__name__
if __name__=="__main__":

    start_pose = (1, 1, 0)

    plan = ((1,2,np.pi,1), (2,2,2*np.pi,0.5))

    path = interpolate_rigid_body([1,1,0], [10,10,2])

    print(path)

    print(forward_propagate_rigid_body(start_pose, plan))
