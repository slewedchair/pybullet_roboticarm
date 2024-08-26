import numpy as np
import random
import pybullet as p
from scipy.spatial.transform import Rotation as R

class Node:
    def __init__(self, config):
        self.config = np.array(config)
        self.parent = None
        self.cost = 0

class InformedRRTStarPlanner:
    def __init__(self, robot_id, start_config, end_effector_goal, max_iter=10000, base_step_size=0.1, min_clearance=0.05):
        self.robot_id = robot_id
        self.start = Node(start_config)
        self.goal = Node(start_config)  
        self.end_effector_goal = np.array(end_effector_goal)
        self.max_iter = max_iter
        self.base_step_size = base_step_size
        self.nodes = [self.start]
        self.joint_ranges = self.get_joint_ranges()
        self.goal_bias = 0.1
        self.goal_threshold = 0.1  
        self.best_cost = float('inf')
        self.c_best = None
        self.c_min = None
        self.min_clearance = min_clearance



    def get_joint_ranges(self):
        ranges = []
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[3] > -1:  # non-fixed joints
                ranges.append((info[8], info[9]))  # (lower limit, upper limit)
                print(f"Joint {i}: {info[1]}, Range: {info[8]} to {info[9]}")
        return ranges

    def set_goal_config(self):
        goal_config = p.calculateInverseKinematics(self.robot_id, 7, self.end_effector_goal)
        self.goal.config = np.clip(np.array(goal_config[:len(self.joint_ranges)]), [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges])

    def plan(self):
        self.set_goal_config()
        self.c_min = self.distance(self.start, self.goal)
        
        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f"Iteration: {i}/{self.max_iter}, Nodes: {len(self.nodes)}, Best cost: {self.c_best}")
            
            if self.c_best is None:
                x_rand = self.sample_free()
            else:
                x_rand = self.informed_sample()
            
            nearest_node = self.get_nearest_node(x_rand)
            new_node = self.steer(nearest_node, x_rand)
            
            if not self.check_collision(new_node.config):
                near_nodes = self.get_near_nodes(new_node)
                self.connect_node(new_node, near_nodes)
                self.rewire(new_node, near_nodes)
                self.nodes.append(new_node)
                
                if self.is_goal_reached(new_node):
                    print(f"Goal reached at iteration {i}")
                    path = self.get_path(new_node)
                    cost = self.path_cost(path)
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.c_best = cost
                        best_path = path
                    
                    if i > self.max_iter / 2:  #Early termination i get bored
                        return self.smooth_path(best_path)
        
        if self.c_best is not None:
            return self.smooth_path(best_path)
        print("No path found")
        return None

    def sample_free(self):
        if random.random() < self.goal_bias:
            return Node(self.goal.config)
        
        config = []
        for i, r in enumerate(self.joint_ranges):
            if i == 0:  # Assuming joint 0 is the base rotation
                config.append(random.uniform(-np.pi, np.pi))
            else:
                config.append(random.uniform(r[0], r[1]))
        return Node(config)

    def informed_sample(self):
        if self.c_best is None:
            return self.sample_free()
        
        c_min = self.c_min
        c_max = self.c_best
        
        for _ in range(100):  #sampling attempts 100 times 
            x_ball = self.sample_unit_ball()
            x_rand = self.transform_to_ellipsoid(x_ball, c_min, c_max)
            if self.is_in_range(x_rand):
                return Node(x_rand)
        
        return self.sample_free() 

    def sample_unit_ball(self):
        while True:
            x = np.random.uniform(-1, 1, len(self.start.config))
            if np.linalg.norm(x) <= 1:
                return x

    def transform_to_ellipsoid(self, x_ball, c_min, c_max):
        r1 = c_max / 2
        r2 = np.sqrt(c_max**2 - c_min**2) / 2
        
        C = np.diag([r1] + [r2] * (len(self.start.config) - 1))
        center = (self.start.config + self.goal.config) / 2
        rotation = self.rotation_to_world()
        
        return rotation.dot(C.dot(x_ball)) + center

    def rotation_to_world(self):
        a1 = (self.goal.config - self.start.config) / np.linalg.norm(self.goal.config - self.start.config)
        M = np.eye(len(self.start.config))
        M[:, 0] = a1
        Q, _ = np.linalg.qr(M)
        return Q

    def is_in_range(self, config):
        return all(r[0] <= c <= r[1] for c, r in zip(config, self.joint_ranges))

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
        return max(min(self.base_step_size * min_dist, self.base_step_size * 3), 0.02)

    def get_min_distance_to_obstacle(self, config):
        current_state = p.saveState()
        for i, angle in enumerate(config):
            p.resetJointState(self.robot_id, i, float(angle))
        
        min_distance = float('inf')
        contact_points = p.getClosestPoints(self.robot_id, -1, self.min_clearance * 2)
        for point in contact_points:
            min_distance = min(min_distance, point[8])
        
        p.restoreState(current_state)
        return max(min_distance, self.min_clearance)
    
    def check_collision(self, config):
        config = np.clip(config, [r[0] for r in self.joint_ranges], [r[1] for r in self.joint_ranges])
        current_state = p.saveState()
        for i, angle in enumerate(config):
            p.resetJointState(self.robot_id, i, float(angle))
        
        #Check clearance
        collision = False
        contact_points = p.getClosestPoints(self.robot_id, -1, self.min_clearance)
        for point in contact_points:
            if point[8] < self.min_clearance:  #point[8] = distance between objects
                collision = True
                break
        
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
                    self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, node):
        for child in [n for n in self.nodes if n.parent == node]:
            child.cost = node.cost + self.distance(node, child)
            self.propagate_cost_to_leaves(child)

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

    def path_cost(self, path):
        return sum(self.distance(Node(path[i]), Node(path[i+1])) for i in range(len(path)-1))

    def smooth_path(self, path):
        smoothed_path = path.copy()
        for _ in range(100):  # Number of smoothing iterations
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

    def get_nearest_node(self, node):
        return min(self.nodes, key=lambda n: self.distance(n, node))