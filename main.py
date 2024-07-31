import pybullet as p
import pybullet_data
import numpy as np
import time
from src.informedrrtstar import InformedRRTStarPlanner
from src.controller import RobotController
from src.path_execution import PathExecutor

def main():

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    # p.setGravity(0, 0, -9.81)

    robot_controller = RobotController("DUME.urdf", useFixedBase=1)
    robot_controller.load_robot()

    obstacle1 = p.loadURDF("cube_small.urdf", [0.9, 0.9, 1.15])
    obstacle2 = p.loadURDF("cube_small.urdf", [-0.2, -0.2, 0.1])

    start_config = [0] * robot_controller.num_joints
    end_effector_goal = [1.3, 1.2, 1.2]  

    start_pos = robot_controller.get_end_effector_position()
    p.addUserDebugLine([0, 0, 0], start_pos, [1, 0, 0], 5, 0)
    p.addUserDebugLine([0, 0, 0], end_effector_goal, [0, 0, 1], 5, 0)

    planner = InformedRRTStarPlanner(robot_controller.robot_id, start_config, end_effector_goal, max_iter=140, base_step_size=0.05, min_clearance=0.1)
    path = planner.plan()

    if path:
        print("Path found")

        executor = PathExecutor(robot_controller)

        executor.visualize_path(path)

        executor.execute_path(path, time_step=0.01, interpolation_steps=100)

        if executor.has_reached_goal(end_effector_goal):
            print("Goal reached")
        else:
            print("Goal NOT reached")


        final_config = path[-1]
        robot_controller.set_joint_positions(final_config)
        print("frozen at final")

        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(0.1)  
        except KeyboardInterrupt:
            print("exited")
    else:
        print("No path found")

    p.disconnect()

if __name__ == "__main__":
    main()