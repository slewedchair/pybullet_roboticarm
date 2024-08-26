import numpy as np
import pybullet as p
import pybullet_data
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment

#TODO: add auto gen obstacles for the robot to avoid and fix the damn base joint that wont turn 
#TODO: generate obstacles that arent right next to the robot (in the arm) because that would just infinitely make you hit the obstacles
#TODO: find max reach of arm


class RobotArmEnv(py_environment.PyEnvironment):
    def __init__(self):
        super(RobotArmEnv, self).__init__()
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("DUME.urdf", [0, 0, 0], useFixedBase=1) #use fixed base might not work
        self.num_joints = 6  # 6 not including wrist rotation and fingers, shouldn't need those (check if this is how you do that)
        self.action_space = array_spec.BoundedArraySpec(shape=(self.num_joints,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self.observation_space = array_spec.BoundedArraySpec(shape=(self.num_joints + 6,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation') # number of joints + 6 (3 for current position and 3 for goal position)
        self.goal = None
        self._episode_ended = False
        self.obstacles = []
        self.max_reach = 1.5  #maximum reach of the arm might not be 1.5 just a guess

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())           #help find plane urdf
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("DUME.urdf", [0, 0, 0], useFixedBase=1)
        self.generate_obstacles()
        self.goal = self.generate_random_goal()     #random goal for training
        return ts.restart(self.get_observation())
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def step(self, action):
        if self._episode_ended:
            return self.reset() 
        self.apply_action(action)    #apply any changes to joints
        p.stepSimulation()
        observation = self.get_observation()     
        reward = self.compute_reward()
        if self.check_collision_with_obstacles():
            reward -= 10  #penalty for collision
            done = True
        done = self.is_done()       
        if self._episode_ended:
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward)
        
    def apply_action(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])     #should apply each action to its respective joint in action

    def get_observation(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))    #current states of joints
        joint_positions = [state[0] for state in joint_states]               #current positions of joints
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]     #current position of hand (what needs to reach the goal position (link index -1))
        return np.array(joint_positions + list(end_effector_pos) + list(self.goal), dtype=np.float32)

    def compute_reward(self):
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.goal))
        return -distance                #distance from end goal and hand position

    def is_done(self):
        end_effector_pos = p.getLinkState(self.robot, self.num_joints - 1)[0]
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.goal))
        self._episode_ended = distance < 0.05
        return distance < 0.05                #complete episode if goal is reached (with tolerance)

    def generate_random_goal(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0.3, 1.0)
        return [x, y, z]
    
    def generate_obstacles(self):
        self.obstacles = []
        num_obstacles = np.random.randint(2, 8)
        for _ in range(num_obstacles):
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            z = np.random.uniform(0, 1.5)              
            obstacle = p.loadURDF("sphere2red.urdf", [x, y, z], globalScaling=0.1)
            self.obstacles.append(obstacle)

    def generate_random_goal(self):
        while True:
            x = np.random.uniform(-self.max_reach, self.max_reach)
            y = np.random.uniform(-self.max_reach, self.max_reach)
            z = np.random.uniform(0, self.max_reach)
            if self.is_reachable([x, y, z]) and abs(x) > 0.1 and abs(y) > 0.1:                 #check how close x and y the robot can go
                return [x, y, z]
            
    def is_reachable(self, position):
        return np.linalg.norm(position) <= self.max_reach             #check if total distance within reach
    
    def check_collision_with_obstacles(self):
        for obstacle in self.obstacles:
            closest_points = p.getClosestPoints(self.robot, obstacle, distance=0.05)
            if len(closest_points) > 0:
                return True
        return False