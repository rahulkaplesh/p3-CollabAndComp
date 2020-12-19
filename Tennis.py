from unityagents import UnityEnvironment
import numpy as np
import logging

'''Adding the environment This is the start point for training'''

env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

import torch
import pickle

from maddpg import MADDPG

from collections import deque
import matplotlib.pyplot as plt
import time, os

maddpg = MADDPG(24, 2, 2, 1976)
scores_max_hist = []
scores_mean_hist = []

logger = logging.getLogger(__name__)

f_handle = logging.FileHandler("Log_File.txt")
f_format = logging.Formatter('%(levelname)s: %(asctime)s %(message)s')
f_handle.setFormatter(f_format)
f_handle.setLevel(logging.INFO)

logger.addHandler(f_handle)

def maddpg_train(n_episodes=2500):

    scores_deque = deque(maxlen=100)
    solved = False

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        scores = np.zeros(num_agents)
        maddpg.reset()
        step = 0
        while True:
            step += 1
            action = maddpg.act(state, i_episode, add_noise=True)
            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            scores += reward

            maddpg.step(i_episode, state, action, reward, next_state, done)

            if np.any(done):
                break

            state = next_state

        score_max = np.max(scores)
        scores_deque.append(score_max)
        score_mean = np.mean(scores_deque)

        scores_max_hist.append(score_max)
        scores_mean_hist.append(score_mean)

        logger.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, score_mean))
        if solved == False and score_mean >= 0.5:
            logger.info('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_mean))
            maddpg.save()
            solved = True

        if i_episode % 500 == 0:
            print()

scores = maddpg_train()

with open('scores.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(scores, filehandle)

env.close()