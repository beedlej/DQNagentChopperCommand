#%%
# https://www.tensorflow.org/agents/tutorials/0_intro_rl
# my reference for this project
#%%
from __future__ import absolute_import, division, print_function
import retro
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import random
import base64
import imageio
import IPython
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

from tf_agents.networks import q_network

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

#%%
tf.compat.v1.enable_v2_behavior()

#%%
num_iterations = 100000 # @param {type:"integer"}
# how many training iterations
initial_collect_steps = 50000  # @param {type:"integer"} 
# before we train model, how many random steps will we record into the buffer?
collect_steps_per_iteration = 10  # @param {type:"integer"}
# for each iteration in num_iteration, how many action/reward samples will we take from the step?
replay_buffer_max_length = 1000000  # @param {type:"integer"}
# max length of the replay buffer
batch_size = 512  # @param {type:"integer"}
# size of our small batches for neural network training
learning_rate = 0.00025  # @param {type:"number"}

log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 4  # @param {type:"integer"}
eval_interval = 5000  # @param {type:"integer"}

#%%
# load chopper command with the atari suite. 
# the suite provides important information such as shape of action space and the location of pixels
env_name = suite_atari.game(name="ChopperCommand", mode='')

#%%
# Load of a training, and testing environment suite
train_py_env = suite_atari.load(env_name)
eval_py_env = suite_atari.load(env_name)
#%%
# Converts environment variables to tensors for more efficient processing
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#%%
# Creating our Q network layer
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
#%%
# We call our optimizer with learning rate declared from above
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
# initialize the agent with the network and other declarations 
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    gamma = .99
    )

agent.initialize()
#%%
# function that calculates avg return on an environment for 5 episodes to calculate policy improvement
def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

#%%
# We initialize a replay buffer that keeps track of data gathered from our env
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
#%%
# initialize a random policy, we are going to need this to gather our initial data
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())
                                                
#%%

#@test {"skip": true}
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, steps=1000)
# We are going to run some steps randomly to gather data for our learner

# This loop is so common in RL, that we provide standard implementations. 
# For more details see the drivers module.
# https://github.com/tensorflow/agents/blob/master/tf_agents/docs/python/tf_agents/drivers.md
#%%
# cast the replay buffer to data. num_steps=2 because we are comparing a first and second observation
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)
#%
iterator = iter(dataset)

#%% TRAINING
# This is the main test loop that takes hyperparameters and uses TF method calls to train
#@test {"skip": true}


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  #if step % log_interval == 0:
  #  print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    
#%% Graph our performance over the steps, comparing average returns

#@test {"skip": true}

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=4000)
#%%
# function to turn replay pictures into a video
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)
#%%
# Function that takes the learned policy from training, records the play into MP4
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
  return embed_mp4(filename)

#%%


create_policy_eval_video(agent.policy, "trained-agent_1")
create_policy_eval_video(agent.policy, "trained-agent_2")


