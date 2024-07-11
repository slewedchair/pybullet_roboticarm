import numpy as np
import random
import pybullet as p

class Node:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0

class RRTStarPlanner:
    def __init__(self, robot_id, start_config, end_effector_goal, max_iter=10000, base_step_size=0.1):
        self.robot_id = robot_id
        self.start = Node(start_config)
        self.goal = Node(start_config)  
        self.end_effector_goal = np.array(end_effector_goal)
        self.max_iter = max_iter
        self.base_step_size = base_step_size
        self.nodes = [self.start]
        self.joint_ranges = self.get_joint_ranges()
        self.goal_bias = 0.1
        self.goal_threshold = 0.05  

    def get_joint_ranges(self):
        ranges = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[3] > -1:  #non-fixed joints
                ranges.append((info[8], info[9]))  # (lower limit, upper limit)
        return ranges

    def set_goal_config(self):
        goal_config = p.calculateInverseKinematics(self.robot_id, 7, self.end_effector_goal)
        self.goal.config = np.clip(np.array(goal_config[:len(self.joint_ranges)]), [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges])

    def plan(self):
        self.set_goal_config()
        
        print("Checking for direct path")
        if self.check_direct_path():
            print("Direct path found")
            return [self.start.config, self.goal.config]
        
        print("No direct path found starting planning")
        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f"Iteration: {i}/{self.max_iter}")
            
            if random.random() < self.goal_bias:
                random_node = Node(self.goal.config)
            else:
                random_node = self.get_random_node()
            
            nearest_node = self.get_nearest_node(random_node)
            new_node = self.steer(nearest_node, random_node)
            
            if not self.check_collision(new_node.config):
                near_nodes = self.get_near_nodes(new_node)
                print(f"Found {len(near_nodes)} near nodes")
                self.connect_node(new_node, near_nodes)
                self.rewire(new_node, near_nodes)
                self.nodes.append(new_node)
                
                if self.is_goal_reached(new_node):
                    print(f"Goal reached at iteration {i}")
                    path = self.get_path(new_node)
                    smoothed_path = self.smooth_path(path)
                    return smoothed_path
            else:
                print(f"Collision detected at iteration {i}")
        
        print("No path found after maximum iterations")
        return None

    def check_direct_path(self):
        steps = 100
        for i in range(steps + 1):
            t = i / steps
            config = self.start.config * (1 - t) + self.goal.config * t
            if self.check_collision(config):
                return False
        return True

    def get_random_node(self):
        if random.random() < 0.1: 
            return Node(np.clip(self.goal.config + np.random.normal(0, 0.1, len(self.goal.config)), [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges]))
        elif random.random() < 0.1:  
            return Node(np.clip(self.start.config + np.random.normal(0, 0.1, len(self.start.config)), [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges]))
        else:
            return Node([random.uniform(r[0], r[1]) for r in self.joint_ranges])

    def steer(self, from_node, to_node):
        direction = to_node.config - from_node.config
        distance = np.linalg.norm(direction)
        step_size = self.adaptive_step_size(from_node)
        if distance > step_size:
            new_config = from_node.config + (direction / distance) * step_size
        else:
            new_config = to_node.config
        new_config = np.clip(new_config, [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges])
        return Node(new_config)

    def adaptive_step_size(self, node):
        min_dist = max(self.get_min_distance_to_obstacle(node.config), 0.01)
        return max(min(self.base_step_size * min_dist, self.base_step_size * 2), 0.01)

    def get_min_distance_to_obstacle(self, config):
        #placeholder
        return 1.0

    def check_collision(self, config):
        config = np.clip(config, [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges])
        current_state = p.saveState()
        for i, angle in enumerate(config):
            p.resetJointState(self.robot_id, i, float(angle))
        collision = len(p.getContactPoints(self.robot_id)) > 1
        p.restoreState(current_state)
        return collision

    def get_near_nodes(self, node):
        num_nodes = len(self.nodes)
        radius = min(self.base_step_size * 5, (np.log(num_nodes) / num_nodes) ** (1/len(self.joint_ranges)))
        near_nodes = [n for n in self.nodes if self.distance(n, node) < radius]
        
        if not near_nodes:
            #return nearest node if not found
            return [min(self.nodes, key=lambda n: self.distance(n, node))]
        
        return near_nodes

    def connect_node(self, node, near_nodes):
        if not near_nodes:
            print(f"No near nodes found for node at {node.config}")
            node.parent = self.start
            node.cost = self.distance(self.start, node)
        else:
            node.parent = min(near_nodes, key=lambda n: n.cost + self.distance(n, node))
            node.cost = node.parent.cost + self.distance(node.parent, node)

    def rewire(self, node, near_nodes):
        for near_node in near_nodes:
            new_cost = node.cost + self.distance(node, near_node)
            if new_cost < near_node.cost:
                if self.is_collision_free(node.config, near_node.config):
                    near_node.parent = node
                    near_node.cost = new_cost

    def is_collision_free(self, config1, config2):
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            config = config1 * (1 - t) + config2 * t
            if self.check_collision(config):
                return False
        return True

    def is_goal_reached(self, node):
        end_effector_pos = self.get_end_effector_position(node.config)
        return np.linalg.norm(end_effector_pos - self.end_effector_goal) < self.goal_threshold

    def get_path(self, node):
        path = []
        while node:
            path.append(node.config)
            node = node.parent
        return list(reversed(path))

    def smooth_path(self, path):
        smoothed_path = path.copy()
        for _ in range(100):  
            if len(smoothed_path) <= 2:
                break
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            i, j = min(i, j), max(i, j)
            if self.is_collision_free(smoothed_path[i], smoothed_path[j]):
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        return smoothed_path

    def get_end_effector_position(self, config):
        current_state = p.saveState()
        for i, angle in enumerate(config):
            p.resetJointState(self.robot_id, i, float(angle))
        pos = p.getLinkState(self.robot_id, 7)[0]
        p.restoreState(current_state)
        return pos

    @staticmethod
    def distance(node1, node2):
        return np.linalg.norm(node1.config - node2.config)