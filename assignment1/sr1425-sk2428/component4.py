import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib.animation import  PillowWriter 
from typing import List, Tuple

# It's giveen that link 1 is 2m and link 2 is 1.5m
# there shouldn't be any case where this is required if start and goal are proper, but just for peace of mind
def is_valid_pose(angle_1: int, angle_2: int):
    # first link shouldn't phase through ground to keep end effector up
    if angle_1 < 0 or angle_1 > np.pi:
        return False
    # end effector should be above ground
    if 2 * np.sin(angle_1) + 1.5 * np.sin(angle_2) < 0:
        return False
    
    return True

def interpolate_arm(start: Tuple[int], goal: Tuple[int]) -> List[Tuple[int]]: # type hint later
    angle_start_1, angle_start_2 = start
    angle_goal_1, angle_goal_2 = goal 
    
    angle_diff_1 = angle_goal_1 - angle_start_1
    angle_diff_2 = angle_goal_2 - angle_start_2
    
    n = 5
    path = []
    for t_step in range(n):
        angle_1_t = angle_start_1 + (angle_diff_1/n) * t_step
        angle_2_t = angle_start_2 + (angle_diff_2/n) * t_step
        
        if is_valid_pose(angle_1_t, angle_2_t):
            path.append((angle_1_t, angle_2_t))
        else:
            print("INVALID POSE")
            print(angle_1_t, angle_2_t)
            return
        
    path.append((angle_goal_1, angle_goal_2))
        
    return path

def forward_propogate_arm(start_pose: Tuple[int], plan: List[Tuple[int]]) -> List[Tuple[int]]: 
    path = [start_pose]
    
    if not is_valid_pose(*start_pose):
        print("INVALID POSE")
        print(angle_1, angle_2)
        return
    
    for w_1, w_2, duration in plan:
        angle_1, angle_2 = path[-1]
        
        angle_1 += (w_1 * duration) % (2*np.pi)
        angle_2 += (w_2 * duration) % (2*np.pi)
        
        if np.sign(angle_1 - path[-1][0]) != np.sign(w_1) or np.sign(angle_2 - path[-1][1]) != np.sign(w_2):
            print("INVALID POSE")
            print(angle_1, angle_2)           
            return 
        
        if is_valid_pose(angle_1, angle_2):
            path.append((angle_1, angle_2))
        else:
            print("INVALID POSE")
            print(angle_1, angle_2)
            return
    
    return path

def visualize_arm_path(path: List[Tuple[int]]): 
        figure, axes = plt.subplots()
        axes.set_xlim([-5, 5])
        axes.set_ylim([-1, 5])
        
        
        link_1 = plt.Rectangle((1, 0.1), 2, 0.2, fill=True, color='b')
        link_2 = plt.Rectangle((0.75, 0.1), 1.5, 0.2, fill=True, color='b')
        
        axes.add_patch(link_1)
        axes.add_patch(link_2)
        
        def update(num):
            angle_1, angle_2 = path[num]
            # robot.set_xy([x - length/2, y - width/2])
            # robot.angle = np.degrees(theta)
            return link_1, link_2
        
        ani = animation.FuncAnimation(figure, update, frames=len(path), interval=50, blit=True)
        
        ani.save("component3.gif", writer=PillowWriter(fps=1))


if __name__ == '__main__':
    
    print(
        interpolate_arm((0, 0), (np.pi, 0))
    )
    
    print(
        forward_propogate_arm((0, 0), [(-1.7, -1.8, 2)])
    )
    
    visualize_arm_path(interpolate_arm((0, 0), (np.pi, 0)))
    