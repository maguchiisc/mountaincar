import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1: -1]

def digitize_state(observation):
    cart_pos, cart_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-1.2, 0.6, num_digitized)),
        np.digitize(cart_v, bins=bins(-0.07, 0.07, num_digitized))
    ]
    return digitized

def get_action(car_pos, car_v, episode):
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[int(car_pos)][int(car_v)])
    else:
        next_action = np.random.choice([0, 1, 2])
    return next_action

def update_Qtable(q_table, car_pos, car_v, action, reward, next_car_pos, next_car_v):
    gamma = 0.9
    alpha = 0.5
    q_table[int(car_pos)][int(car_v)][action] += alpha*(reward + gamma * max(q_table[int(next_car_pos)][int(next_car_v)]) - q_table[int(car_pos)][int(car_v)][action])
    return q_table

env = gym.make('MountainCar-v0')
num_episodes = 4000
num_digitized = 10
q_table = np.random.uniform(low=-1, high=1, size=(num_digitized, num_digitized, env.action_space.n))
final_x = np.zeros((num_episodes, 1))
islearned = 0
isrender = 0

for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    action = get_action(state[0], state[1], episode)
    episode_reward = 0
    done = 0
    t = 0

    while done == 0:
        observation, reward, done, info = env.step(action)
        next_state = digitize_state(observation)

        if done:
            if t < 198:
                reward = 3000
            else:
                if state[1] > 5:
                    reward = -abs(next_state[1]-10)**2
                else:
                    reward = -next_state[1]**2
        else:
            if state[1] > 5:
                reward = -abs(state[1]-10)**2
            else:
                reward = -state[1]**2

        episode_reward += reward


        q_table = update_Qtable(q_table, state[0], state[1], action, reward, next_state[0], next_state[1])

        action = get_action(next_state[0], next_state[1], episode)

        state = next_state

        if done:
            print('学習終了', episode, '報酬', episode_reward, 't', t)
        t += 1

done = 0
t_list = []
for i in range(200):
    observation = env.reset()
    t = 0
    done = 0
    while done == 0:
        state = digitize_state(observation)
        action = np.argmax(q_table[state[0]][state[1]])
        observation, reward, done, info = env.step(action)
        if done:
            t_list.append(t)
        t += 1

plt.plot(t_list)
plt.show()
