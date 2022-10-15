# objective is to get the cart to the flag.
# for now, let's just move randomly:

import gym
import numpy as np

trainingEnv = gym.make("MountainCar-v0")
executionEnv = gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 3000
MAX_EPISODE = 250
DISCRETE_OS_SIZE = [20, 20]
N_STATES= 40

discrete_os_win_size = (trainingEnv.observation_space.high - trainingEnv.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [trainingEnv.action_space.n]))


def get_discrete_state(state):
    # individual bounding
    a = int((state[0] - trainingEnv.observation_space.low[0]) / discrete_os_win_size[0])
    b = int((state[1] - trainingEnv.observation_space.low[1]) / discrete_os_win_size[1])
    return (a,b)


def training(epsilon):
    for episode in range(EPISODES):
        discrete_state = get_discrete_state(trainingEnv.reset()[0])
        done = False
        print(episode)

        while not done:
            # Epsilon greedy
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, trainingEnv.action_space.n)


            new_state, reward, done, truncated, _ = trainingEnv.step(action)
            new_discrete_state = get_discrete_state(new_state)

            # If simulation did not end yet after last step - update Q table
                # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

            # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            if new_state[0] >= trainingEnv.goal_position:
                #q_table[discrete_state + (action,)] = reward
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
            discrete_state = get_discrete_state(trainingEnv.reset()[0])


if __name__ == "__main__":
    print("beginning training")
    ep = 1
    training(ep)
    discrete_state = get_discrete_state(executionEnv.reset()[0])
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncated, _ = executionEnv.step(action)
        new_discrete_state = get_discrete_state(new_state)
        discrete_state = new_discrete_state