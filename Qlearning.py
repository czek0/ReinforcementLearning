# Q learning demonstration

import gym

"""
There are three actions:
0: Left
1: Nothing
2: Right
"""
env = gym.make("MountainCar-v0", render_mode='human')
env.reset()

# iterate through environment
done = False

while not done:
    action = 2
    new_state, reward, done, truncated, _ = env.step(action)

env.close()