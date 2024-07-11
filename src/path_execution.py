import time
import pybullet as p
import numpy as np

class PathExecutor:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller

    def execute_path(self, path, time_step=0.01, interpolation_steps=100):
        print("Executing path")
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i+1]
            
            for step in range(interpolation_steps):
                t = step / interpolation_steps
                current_config = self.interpolate(start_config, end_config, t)
                self.robot_controller.set_joint_positions(current_config)
                p.stepSimulation()
                time.sleep(time_step)
        
        self.robot_controller.set_joint_positions(path[-1])
        p.stepSimulation()
        print("Path executed")

    def interpolate(self, start_config, end_config, t):
        return start_config + t * (end_config - start_config)

    def visualize_path(self, path, color=[1, 0, 0]):
        print("Visualizing path")
        for i in range(len(path) - 1):
            start_pos = self.robot_controller.get_end_effector_position(path[i])
            end_pos = self.robot_controller.get_end_effector_position(path[i+1])
            p.addUserDebugLine(start_pos, end_pos, color, lineWidth=2, lifeTime=0)
        print("Path visualized")

    def has_reached_goal(self, goal_position, tolerance=0.15):
        current_position = self.robot_controller.get_end_effector_position()
        distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        return distance < tolerance