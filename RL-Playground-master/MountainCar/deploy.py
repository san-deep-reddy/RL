import gym
import numpy as np
import random
from numpy import genfromtxt

env = gym.make("MountainCar-v0").env
env.reset()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

#min_pos = -1.2,	max_pos = 0.6
#min_vel = -0.07,	max_vel = 0.07
#q_table = np.zeros((19, 15, 3))
q_table = genfromtxt('mountaincar.csv', delimiter=',').reshape((19, 15, 3))


total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        pos = int(round(state[0], 1)*10 + 12)
        vel = int(round(state[1], 2)*100 + 7)
        action = np.argmax(q_table[pos][vel])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1
        env.render()

    total_penalties += penalties
    total_epochs += epochs
    print("Episodes : ", _, "    Epochs : ", epochs)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

env.close()
