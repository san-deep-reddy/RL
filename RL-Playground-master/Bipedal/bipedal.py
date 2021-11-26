import gym
import numpy as np
import random
import Box2D

env = gym.make("BipedalWalkerHardcore-v3").env
env.reset()
print("Action Space : ", env.action_space[0])
print("State Space : ", env.observation_space)

#min_pos = -1.2,	max_pos = 0.6
#min_vel = -0.07,	max_vel = 0.07
q_table = np.zeros((19, 15, 3))

alpha = 0.1
gamma = 0.6
epsilon = 0.1
state = env.reset()
print("Starting Training")
index = 0
action = env.action_space.sample()
for i in range(1, 20):
    new_state, reward, done, info = env.step(action)
    #print(action[0])
    #print(action[1])
    #print(action[2])
    env.render()
    print("Action : ", action[0].shape)
    print("State : ", state)
    action = new_state[0:4]
    print(action)
    #break
	

print("Training finished.\n")
env.close()
