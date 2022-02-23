import random

import gym
import gym_partially_observable_grid
import numpy as np

# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed). Only necessary if is_partially_obs is set to True AND you want
# the underlying system to behave like deterministic MDP.
indicate_slip = True
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = False
# If one_time_rewards is set to True, reward in single location will be obtained only once per episode.
# Otherwise, reward will be given every time
one_time_rewards = True

env = gym.make(id='poge-v1',
               world_file_path='worlds/world0.txt',
               force_determinism=force_determinism,
               indicate_slip=indicate_slip,
               indicate_wall=True,
               is_partially_obs=is_partially_obs,
               one_time_rewards=one_time_rewards,
               step_penalty=-0.1)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

num_training_episodes = 10000

# For plotting metrics
all_epochs = []

for i in range(1, num_training_episodes + 1):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    while not done:

        action = env.action_space.sample() if random.random() < epsilon else np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -1:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")

total_epochs = 0
episodes = 100

goals_reached = 0
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == env.goal_reward and done:
            goals_reached += 1

        epochs += 1

    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Total Number of Goal reached: {goals_reached}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
