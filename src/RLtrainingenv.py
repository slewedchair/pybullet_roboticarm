import gym
import numpy as np
import pybullet as p
import pybullet_data

class RobotArmEnv(gym.Env):
    def __init__(self):
        super(RobotArmEnv, self).__init__()
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("DUME.urdf", [0, 0, 0], useFixedBase=1)
        self.num_joints = 6  # 6 not including wrist rotation and fingers, shouldn't need those (check if this is how you do that)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints + 6,), dtype=np.float32) # number of joints + 6 (3 for current position and 3 for goal position)
        self.goal = None

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())           #help find plane urdf
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("DUME.urdf", [0, 0, 0], useFixedBase=1)
        self.goal = self.generate_random_goal()    #random goal for training
        return self.get_observation()

    def step(self, action):
        self.apply_action(action)    #apply any changes to joints
        p.stepSimulation()
        observation = self.get_observation()     
        reward = self.compute_reward()
        done = self.is_done()        
        return observation, reward, done, {}

    def apply_action(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])     #should apply each action to its respective joint in action

    def get_observation(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))    #current states of joints
        joint_positions = [state[0] for state in joint_states]               #current positions of joints
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]     #current position of hand (what needs to reach the goal position (link index -1))
        return np.array(joint_positions + list(end_effector_pos) + list(self.goal))

    def compute_reward(self):
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.goal))
        return -distance                #distance from end goal and hand position

    def is_done(self):
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.goal))
        return distance < 0.05                #complete episode if goal is reached (with tolerance)

    def generate_random_goal(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0.3, 1.0)
        return [x, y, z]