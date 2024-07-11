import pybullet as p
import time

class RobotController:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot_id = None
        self.num_joints = None

    def load_robot(self):
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=1)
        self.num_joints = p.getNumJoints(self.robot_id)

    def set_joint_positions(self, positions):
        for i in range(len(positions)):
            p.resetJointState(self.robot_id, i, positions[i])
        p.stepSimulation()

    def get_end_effector_position(self, config=None):
        if config is None:
            return p.getLinkState(self.robot_id, 7)[0] 
        else:
            current_state = p.saveState()
            for i, angle in enumerate(config):
                p.resetJointState(self.robot_id, i, float(angle))
            pos = p.getLinkState(self.robot_id, 7)[0]
            p.restoreState(current_state)
            return pos

    def execute_path(self, path, time_step=0.1):
        for config in path:
            self.set_joint_positions(config)
            time.sleep(time_step)