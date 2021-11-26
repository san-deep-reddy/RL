import gym
import numpy as np
import random
import Box2D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.initializers import RandomUniform
from keras.regularizers import l2
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import os
import time

class BipedalDQN(keras.Model):
    def __init__(self, n_obs, n_action, gamma, lr, batch_size, s_link):
        super(BipedalDQN, self).__init__()
        self.nx = n_obs
        self.ny = n_action
        self.gamma = gamma
        #self.observation_list = deque(maxlen=10000)	
        self.obs_list = []
        self.reward_list = [] 
        self.obs_new_list = []
        self.done_list = []
        self.s_link =s_link
        self.lr = lr 
        self.los = []
        self.ep_rewards = []
        self.epsilon = 1.0
        #self.epsilon = 0.053
        self.e_= 0.01
        self.dc= 0.995
        self.max_length = 100000
        self.batch_size = batch_size
        self.loss_function = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam()
        if os.path.isfile('./' + self.s_link):
            print("LOAD existing keras model....")
            self.model = load_model(self.s_link)
            self.model_target = load_model(self.s_link)
            print(self.model.summary())
        else:
            # Call function model to build the model   
            print("Build a new model")
            self.model = self.create_model()
            self.model_target = self.create_model() 
        print(self.model.summary())
        #self.model = self.create_model()


    def store_sample(self, observation, action, reward, observation_new, done):
        #self.observation_list.append((observation, action, reward, observation_new, flags ))
        self.obs_list.append(observation)
        self.obs_new_list.append(observation_new)
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.ep_rewards.append(reward)
        if len(self.done_list) > self.max_length:
            self.obs_list = self.obs_list[1:]
            self.obs_new_list = self.obs_new_list[1:]
            self.reward_list = self.reward_list[1:]
            self.done_list = self.done_list[1:]
        #print(self.reward_list)


    def choose_action(self,observation):
        if np.random.rand() <= self.epsilon : 
            action = np.random.uniform(-1,1,4)
            return action    
        action = self.model(observation, training = False)    
        #print(action.shape)
        #print(self.nx, self.ny)
        return action[0]
        
    def create_model(self):
        inputs = layers.Input(shape=(self.nx,))
        # Convolutions on the frames on the screen
        x = layers.Dense(72, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(48, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(24, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(12, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        action = layers.Dense(self.ny, activation="tanh")(x)
        model = keras.Model(inputs=inputs, outputs=action)
        #model.compile(loss='mse', optimizer="adam")
        return model

    def train(self):
        indices = np.random.choice(range(len(self.done_list)), size=self.batch_size)
        """reward = np.array([np.repeat(self.reward_list[i], 4) for i in indices])
        observation = np.array([self.obs_list[i] for i in indices]).reshape(self.batch_size, self.nx)
        observation_new = np.array([self.obs_new_list[i] for i in indices]).reshape(self.batch_size, self.nx)
        print(reward.shape)
        print(observation.shape)
        print(observation_new.shape)
        #quit()
        predictions = self.model.predict(observation_new)
        target = reward + self.gamma*predictions
        print(target.shape)
        #quit()
        history = self.model.fit(x=observation, y=target, verbose=0, epochs=5, batch_size=32)"""
        reward = []
        observation_new = []
        observation = []
        done = []
        for i in indices:
            reward.append(np.repeat(self.reward_list[i], 4).reshape(1, -1))
            observation.append(self.obs_list[i])
            observation_new.append(self.obs_new_list[i])
            done.append(np.repeat(float(self.done_list[i]), 4).reshape(1, -1))
            #predictions = self.model.predict(observation_new)
            #target = reward
            #if not self.done_list[i]:
                #target = reward + self.gamma*(predictions)
                #target = target.reshape(1,-1)
        target = np.ones([256, 4])
        observation_new = np.array(observation_new).reshape(self.batch_size, self.nx)
        observation = np.array(observation).reshape(self.batch_size, self.nx)
        reward = np.array(reward).reshape(self.batch_size, self.ny)
        done = np.array(done).reshape(self.batch_size, self.ny)
        predictions = self.model_target.predict(observation_new)
        target = reward + (self.gamma * predictions * (1 - done))
        #print(observation.shape)
        #print(observation_new.shape)
        #print(reward.shape)
        #print(target.shape)
        #print(done.shape)
        #quit()
            
            #print(observation)
            #print(target)
        #history = self.model.fit(x=observation, y=target, batch_size=1, epochs=10)
        #history = self.model.fit(observation, target, batch_size=32, epochs=50)
        with tf.GradientTape() as tape:
            print("Gradient Tape")
            target_pred = self.model(observation)
            loss = self.loss_function(target, target_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        #print(loss)
        #quit()
        self.ep_rewards = []
        self.model_target.set_weights(self.model.get_weights())
        #print(history)
        #self.los.append(history.history['loss'])
        #mm = np.mean(self.los)
        if self.epsilon >= self.e_:
            self.epsilon *= self.dc
        #return history, mm

if __name__ == "__main__":
    s_link = "BipedalWalker_model_custom.h5"
    env = gym.make("BipedalWalkerHardcore-v3").env
    env.reset()
    print("------------------------------------------------------------")
    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))
    print("------------------------------------------------------------")
    n_action = env.action_space.shape[0]
    n_obs = env.observation_space.shape[0]
    episodes = 100000
    alpha = 0.001
    gamma = 0.99
    epsilon = 0.05
    batch_size = 512
    train_counter = 40
    lr = 0.0005
    agent = BipedalDQN(n_obs, n_action, gamma, lr, batch_size, s_link)
    r_list = []
    win = 0

    for i in range(episodes):
        done = False
        observation = env.reset()	
        observation = observation.reshape(1, -1)
        counter = 0
        epochs = 0
        start = time.time()
        while True:
            flag = False
            #print("Shape : ", observation)
            env.render()
            action = agent.choose_action(np.array(observation))
            #print(action)
            observation_new, reward, done, inf = env.step(action)
            #print(reward)
            observation_new = observation_new.reshape(1, -1)
            agent.store_sample(observation, action, reward, observation_new, done)
            observation = observation_new
            
            epochs+=1
            total = sum(agent.ep_rewards)
            end = time.time()
            time_space = end - start
            
            if time_space > 12:
                flag = True
            if total>=300:
                print("Winner")
                win+=1
            if total<-300:
                flag = True
            if flag:
                print("--------------------Training-------------------")
                print("Max reward till now : ", max(agent.ep_rewards))
                print("Total Rewards : ", total)
                print("Epsilon : ", agent.epsilon)
                print("Times won : ", win)
                agent.train()
                print("-----------------------------------------------")
                #print("+++++++++++++++++++++++++++++++++++++++++++++++")
                #print("MM : ", mm)
                #print("History : ", history)
                #print("+++++++++++++++++++++++++++++++++++++++++++++++")
                counter = 0
                agent.model.save(agent.s_link)
                if i%250 == 0:
                    agent.update_weight = True
                break
        print("Iteration : ", i,"	Epochs : ", epochs)
    env.close()
    
