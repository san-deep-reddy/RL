import gym
import numpy as np
import random

env = gym.make("CartPole-v0").env
env.reset()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

#min_pos = -1.2,	max_pos = 0.6
#min_vel = -0.07,	max_vel = 0.07
q_table = np.zeros((19, 15, 3))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

print("Starting Training")
for i in range(1, 100001):
	done = False
	
	state = env.reset()
	
	epochs, penalties, reward = 0, 0, 0
    
	while not done:
		#action = env.action_space.sample()
		pos = int(round(state[0], 1)*10 + 12)
		vel = int(round(state[1], 2)*100 + 7)
		#print(pos, vel)
		if random.uniform(0,1)<epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(q_table[pos][vel])
		
		new_state, reward, done, info = env.step(action)
		pos_new = int(round(new_state[0], 1)*10 + 12)
		vel_new = int(round(new_state[1], 2)*100 + 7)
		#print("Action : ",action, "	State : ", pos_new, "	", vel_new)
		old_val = q_table[pos][vel][action]
		new_max = np.max(q_table[pos_new][vel_new])
		new_val = (1 - alpha)*old_val + alpha*(reward + gamma*new_max)
		q_table[pos][vel][action] = new_val
		
		state = new_state
		epochs+=1
		env.render()	
		
	print("Iteration : ", i,"	Epochs : ", epochs)
	np.savetxt("mountaincar.csv", q_table.reshape((q_table.shape[0]*q_table.shape[1]), q_table.shape[2]), delimiter=",")

print("Training finished.\n")
env.close()
