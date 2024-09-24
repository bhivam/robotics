import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
from matplotlib.animation import  PillowWriter # type: ignore


def interpolate_rigid_body(start_pose, goal_pose):
    x, y, theta = start_pose
    x_goal, y_goal, theta_goal = goal_pose

    num_poses = 10

    delta_x = (x_goal - x)/num_poses
    delta_y = (y_goal - y)/num_poses
    delta_theta = (theta_goal - theta)/num_poses
    
    path = [start_pose]
    for t in range(num_poses):
        x += delta_x
        y += delta_y
        theta += delta_theta

        path.append((x, y, theta))

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
        
        path.append((x, y, theta))

    return path


def visualize_path(path, name="visualize.gif"):
    figure, axes = plt.subplots()
    axes.set_xlim([-10, 10])
    axes.set_ylim([-10, 10])
    
    length, width = (0.5, 0.3)
    
    robot = plt.Rectangle((length/2, width/2), length, width, fill=True, color='b')
    axes.add_patch(robot)
    
    def update(num):
        x, y, theta = path[num]
        robot.set_xy([x - length/2, y - width/2])
        robot.angle = np.degrees(theta)
        return robot,
    
    ani = animation.FuncAnimation(figure, update, frames=len(path), interval=50, blit=True)
    
    ani.save(name, writer=PillowWriter(fps=1))


#__name__
if __name__=="__main__":

    # interpolate_rigid_body

    # test case 1

    path = interpolate_rigid_body([-9,-9,0], [9,9,0])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase1.gif")

    # test case 2
    path = interpolate_rigid_body([-9,-9,0], [9,0,0])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase2.gif")
    
    # test case 3
    path = interpolate_rigid_body([-9,-9,0], [0,9,0])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase3.gif")
    
    # test case 4
    path = interpolate_rigid_body([-9,-9,0], [9,9,np.pi/2])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase4.gif")
    
    # test case 5
    path = interpolate_rigid_body([-9,-9,0], [9,9,np.pi])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase5.gif")

    # test case 6
    path = interpolate_rigid_body([0,-9,0], [9,0,np.pi])
    visualize_path(path, "component3_gifs/interpolate_rigid_body_testcase6.gif")

    # forward_propagate_rigid_body

    # test case 7
    plan = ((1,2,np.pi,1), (2,2,2*np.pi,0.5))
    path = forward_propagate_rigid_body([-9,-9,0], plan)
    visualize_path(path, "component3_gifs/forward_propagate_rigid_body_testcase7.gif")

    # test case 8
    plan = ((3,2,np.pi,1), (2,2,2*np.pi,0.5),(3,3,2*np.pi,4))
    path = forward_propagate_rigid_body([-9,-9,0], plan)
    visualize_path(path, "component3_gifs/forward_propagate_rigid_body_testcase8.gif")
    
    # test case 9
    plan = ((1,1,np.pi/2,5), (2,2,np.pi/4,0.5))
    path = forward_propagate_rigid_body([-9,-9,0], plan)
    visualize_path(path, "component3_gifs/forward_propagate_rigid_body_testcase9.gif")