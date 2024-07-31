import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.utils import common
import pybullet as p
import numpy as np
from src.controller import RobotController
from src.path_execution import PathExecutor

#TODO: train PPO agent

"""
TODO: (I think)
Create environment thru RLtrainingenv.py - should be a function call or two with py_environment
Actor and Value networks thru tf_agents.networks - again read docs but should be function calls with actor_distribution_network and value_network
Create + initialize PPO agent - 1 function call with tf_agents.agents.ppo (ppo_agent)

Still to figure out:
Data collection, training, and eval?


"""