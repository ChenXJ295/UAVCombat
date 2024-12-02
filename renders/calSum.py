# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

dfr5 = pd.read_csv('dp//dp4.csv')
data = dfr5
scalar = data['reward'].values
episode = []
sum_reward = []
reward = []
j = 0
for i in scalar:
    j = j + 1
    episode.append(j)
    # if i > 0:
    #     reward.append(-i)
    # else:
    #     reward.append(i)
    reward.append(i)
    sum_reward.append(np.sum(reward))
result = np.column_stack((np.array(episode, dtype=int), sum_reward))
data=pd.DataFrame(result)
data.columns=['timestep','reward']
pd.DataFrame(data).to_csv('dp//sum4.csv')