# Q learning demonstration

import gym
import numpy as np

"""
The goal is 
There are three actions:
0: Left
1: Nothing
2: Right
"""

LEARNING_RATE = 0.1 #gamma
DISCOUNT = 0.95 # measure of how important the future rewards are over current rewards
EPISODES = 25000
SHOW_EVERY = 1000
MAX_EPISODES = 100

env = gym.make("MountainCar-v0", render_mode='human')


"""
Create a Q table of a manageable sized buckets using the largest range [0.6, 0.07]
usually you will need to keep this as a dynamic size and run a few 100 episodes to tweak the values
 | Action 1 | Action 2 | Action 3 
"""
# DISCRETE_OS_SIZE = [20,20] # in most cases this depends on the environment
# discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE # size of each bucket
# print(discrete_os_win_size)

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/ (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(env.observation_space)
# print(env.action_space.n)
# q_table = np.random.rand(env.observation_space.n, env.action_space.n)
# q_table[-1,:] = np.zeros(env.action_space.n)

# def get_discrete_state(state):
#     discrete_state = (state[0] - env.observation_space.low) / discrete_os_win_size
#     if discrete_state[1] < 0:
#         discrete_state[1] = 0
#     return tuple(discrete_state.astype(np.int64))
#

# print(discrete_State)
# print(np.argmax(q_table[discrete_State])) # gets max value
# discrete_state = get_discrete_state(env.reset()) #returns the initial state


# iterate through environment
"""
Implement Q learning
"""
for episode in range(EPISODES):
    # discrete_state = get_discrete_state(env.reset()) #returns the initial state
    s, _ = env.reset()

    # done = False
    # t = 0
    # if episode % SHOW_EVERY == 0:
    #     print(episode)
    #     render = True
    # else:
    #     render = False

    # while not done:
    for max_step in range(MAX_EPISODES):
        # epsilon greedy
        if np.random.random() > episode:
            action = np.argmax(q_table[discrete_state]) # gets max value
        else:
            action = np.random.randint(0, env.action_space.n)

        # get new state and reward and check if goal state
        # action = np.argmax(q_table[discrete_state])  # gets max value

        new_state, reward, done, truncated, _ = env.step(action)
        # new_discrete_state = get_discrete_state(new_state)


        # if render: # if a showable render
        #     env.render()

        # if not done: # get new q values
        # max_future_q = np.max(q_table[discrete_state]) # Maximum possible Q value in next step (for new state)
        # current_q = q_table[discrete_state + (action, )] # Current Q value (for current state and performed action)

        # new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[s,a] = q_table[s,a] + LEARNING_RATE*(reward + DISCOUNT * np.max(q_table[new_state, :]) - q_table[s, a])
            # update the q_table with new value
        # q_table[discrete_state+(action, )] = new_q
        # elif new_state[0] >= env.goal_position: # if position is larger than goal position
        #     print(f"We made it on episode {episode}")
        #     q_table[discrete_State + (action, )] = 0
        s = new_state
        if done:
            print(f"We made it on episode {episode}")
            q_table[discrete_State + (action,)] = 0
            break


        # discrete_State = new_discrete_state
        # t += 1


    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()