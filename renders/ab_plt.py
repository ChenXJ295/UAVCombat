# -*- coding: utf-8 -*-
import random

import numpy as np
import scienceplots

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('science')

dfa1 = pd.read_csv('resultour//our1.csv')
dfa2 = pd.read_csv('resultour//our2.csv')
dfa3 = pd.read_csv('resultour//our3.csv')
dfa4 = pd.read_csv('resultour//our4.csv')
# dfa5 = pd.read_csv('resultour//our5.csv')
dfa = dfa1._append(dfa2._append(dfa3._append(dfa4)))
data = dfa
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfa = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})

dfb1 = pd.read_csv('do//do1.csv')
dfb2 = pd.read_csv('do//do2.csv')
dfb3 = pd.read_csv('do//do3.csv')
dfb4 = pd.read_csv('do//do4.csv')
# dfb5 = pd.read_csv('resultSFK//SFK5.csv')
dfb = dfb1._append(dfb2._append(dfb3._append(dfb4)))
data = dfb
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfb = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})



dfc1 = pd.read_csv('dp//dp1.csv')
dfc2 = pd.read_csv('dp//dp2.csv')
dfc3 = pd.read_csv('dp//dp3.csv')
dfc4 = pd.read_csv('dp//dp4.csv')
dfc = dfc1._append(dfc2._append(dfc3._append(dfc4)))
data = dfc
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfc = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})










with plt.style.context(['science', 'ieee','grid']):
    palette = sns.color_palette("deep", 6)

    # sns.lineplot(data=dfa, x="timestep", y="reward", color='black', label = 'Our Framework')
    # sns.lineplot(data=dfb, x="timestep", y="reward", color='blue', label='Without Opponent Distinguish')
    # sns.lineplot(data=dfc, x="timestep", y="reward", color='red', label='Without Policy Judgment')
    # plt.title('Average Episode Reward of Our Agent', fontsize=6)
    # plt.xlabel('Episode')
    # plt.ylabel('Average Episode Reward')
    # plt.legend(loc='best', prop={'size': 3})
    # plt.ylim(-5, 10)
    # plt.xlim(1, 300)
    # plt.show()

    sns.barplot(data=dfa,  y="reward", color='black', label='Our Framework', errorbar='sd')
    sns.barplot(data=dfb,  y="reward", color='blue', label='Without Opponent Distinguish', errorbar='sd')
    sns.barplot(data=dfc, y="reward", color='red', label='Without Policy Judgment', errorbar='sd')
    plt.title('Average Episode Reward of Our Agent', fontsize=6)
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    plt.legend(loc='best', prop={'size': 3})
    plt.show()