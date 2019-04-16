import gym
from gym import wrappers
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers
from keras import backend as K
import tensorflow as tf
import time
from statistics import mean
import matplotlib.pyplot as plt


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

class QNetwork:
    def __init__(self, learning_rate=0.0001, state_size=2, action_size=3, hidden_size=50):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 2))
        targets = np.zeros((batch_size, 3))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if (next_state_b[0] != fin).any():
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target

        self.score.append(self.model.evaluate(inputs, targets, verbose=0))
        self.model.fit(inputs, targets, epochs=1, verbose=0)

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    def len(self):
        return len(self.buffer)

class Actor:
    def get_action(self, state, episode, mainQN):
        epsilon = 0.001 + 0.9 / (1.0+episode)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)

        else:
            action = np.random.choice([0, 1, 2])

        return action

DQN_MODE = 1
LENDER_MODE = 1
learning_rate = 0.0001
env = gym.make('MountainCar-v0')
fin = [0.52606228, 0.02842538]
num_episodes = 4000
gamma = 0.99
isrender = 0
hidden_size = 50
memory_size = 10000
batch_size = 32

mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
memory = Memory(max_size=memory_size)
actor = Actor()
t_list = []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    targetQN.model.set_weights(mainQN.model.get_weights())
    done = 0
    state = np.reshape(state, [1, 2])
    t = 0

    while done == 0:
        mainQN.score = []
        action = actor.get_action(state, episode, mainQN)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 2])

        episode_reward += reward

        memory.add((state, action, reward, next_state))
        state = next_state

        if memory.len() > batch_size:
            mainQN.replay(memory, batch_size, gamma, targetQN)

        if done:
            print('episode', episode, 'reward', episode_reward, 't', t, 'loss', mean(mainQN.score))
            t_list.append(t)
        t += 1

done = 0
observation = env.reset()
state = np.reshape(observation, [1, 2])
targetQN.model.set_weights(mainQN.model.get_weights())
while done == 0:
    env.render()
    action = np.argmax(mainQN.model.predict(state)[0])
    next_state, reward, done, info = env.step(action)
    next_state = np.reshape(next_state, [1, 2])
    state = next_state
    time.sleep(0.01)

plt.plot(t_list)
plt.show()
