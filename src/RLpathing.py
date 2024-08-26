import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.utils import common
import pybullet as p
import numpy as np
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver
from .controller import RobotController
from .path_execution import PathExecutor
from .RLtrainingenv import RobotArmEnv



#TODO: train PPO agent

"""
TODO: (I think)
Create environment thru RLtrainingenv.py - should be a function call or two with py_environment
Actor and Value networks thru tf_agents.networks - again read docs but should be function calls with actor_distribution_network and value_network
Create + initialize PPO agent - 1 function call with tf_agents.agents.ppo (ppo_agent)

Still to figure out:
Data collection, training, and eval?

Do ddpg and dqn after get this one working
"""

env = RobotArmEnv()
tf_env = tf_py_environment.TFPyEnvironment(env)


actor_net = actor_distribution_network.ActorDistributionNetwork(tf_env.observation_spec(), tf_env.action_spec(), fc_layer_params=(200, 100))
value_net = value_network.ValueNetwork(tf_env.observation_spec(), fc_layer_params=(200, 100))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(tf_env.time_step_spec(), tf_env.action_spec(), actor_net=actor_net, value_net=value_net, optimizer=optimizer, train_step_counter=train_step_counter)
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=1000)
collect_driver = dynamic_step_driver.DynamicStepDriver(tf_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=1)

num_iterations = 20000
collect_steps_per_iteration = 1
batch_size = 64

for i in range(num_iterations):
    collect_driver.run()
    
    experience = replay_buffer.gather_all()
    train_loss = agent.train(experience)
    replay_buffer.clear()
    
    if i % 1000 == 0:
        print(f'step = {i}: loss = {train_loss.loss}')  #print every 1000 iterations


saver = policy_saver.PolicySaver(agent.policy)
saver.save('ppo_policy')
