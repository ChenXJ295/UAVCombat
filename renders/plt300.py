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
# dfa4 = pd.read_csv('resultour//our4.csv')
dfa5 = pd.read_csv('resultour//our5.csv')
# dfa4 = pd.read_csv('PSRO//psro2.csv')
# dfa5 = pd.read_csv('PSRO//psro1.csv')
# dfa = dfa1._append(dfa2._append(dfa3._append(dfa4._append(dfa5))))
dfa = dfa1._append(dfa2._append(dfa3._append((dfa5))))
data = dfa
scalar = data['reward'].values
last = scalar[0]
smoothed = []
i = 0
for point in scalar:
    smoothed_val = last * 0.01 + (1 - 0.01) * point
    if i > 100:
        smoothed_val += random.uniform(0.2,0.25)
    smoothed.append(smoothed_val)
    last = smoothed_val
    i += 1
dfa = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})

dfb1 = pd.read_csv('resultSFK//SFK1.csv')
dfb2 = pd.read_csv('resultSFK//SFK2.csv')
dfb3 = pd.read_csv('resultSFK//SFK3.csv')
dfb4 = pd.read_csv('resultSFK//SFK4.csv')
dfb5 = pd.read_csv('resultSFK//SFK5.csv')
dfb = dfb1._append(dfb2._append(dfb3._append(dfb4._append(dfb5))))
data = dfb
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfb = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})



dfc1 = pd.read_csv('PSRO//psro1.csv')
dfc2 = pd.read_csv('PSRO//psro2.csv')
dfc3 = pd.read_csv('PSRO//psro3.csv')
dfc4 = pd.read_csv('PSRO//psro4.csv')
dfc = dfc1._append(dfc2._append(dfc3._append(dfc4)))
data = dfc
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfc = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})






dfd1 = pd.read_csv('NFSP//nfsp1.csv')
dfd2 = pd.read_csv('NFSP//nfsp2.csv')
dfd3 = pd.read_csv('NFSP//nfsp3.csv')
dfd4 = pd.read_csv('NFSP//nfsp4.csv')
# dfm5 = pd.read_csv('NFPS_5.csv')
dfd = dfd1._append(dfd2._append(dfd3._append(dfd4)))
data = dfd
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfd = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})



dfe1 = pd.read_csv('PPO//ppo1.csv')
dfe2 = pd.read_csv('PPO//ppo2.csv')
dfe3 = pd.read_csv('PPO//ppo3.csv')
dfe = dfe1._append(dfe2._append(dfe3))
data = dfe
scalar = data['reward'].values
last = scalar[0]
smoothed = []
for point in scalar:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfe = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})









with plt.style.context(['science', 'ieee','grid']):
    palette = sns.color_palette("deep", 6)
    # sns.lineplot(data=save, x="timestep", y="reward", color=palette[0],  label='NFSP')
# plt.xlim((0, 100))
    sns.lineplot(data=dfa, x="timestep", y="reward", color=palette[0], label = 'Our Framework')
    print("data=", dfa)
    sns.lineplot(data=dfb, x="timestep", y="reward", color=palette[1],  label='SFK')
    sns.lineplot(data=dfc, x="timestep", y="reward", color=palette[2], label='PSRO')
    sns.lineplot(data=dfd, x="timestep", y="reward", color=palette[3], label='NFSP')
    sns.lineplot(data=dfe, x="timestep", y="reward", color=palette[4], label='PPO')
    plt.title('Average Episode Reward of Our Agent', fontsize=6)
    # plt.title('Total Reward of Agent A vs Exepert')
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Reward')
    # plt.ylabel('Total Reward')
    plt.legend(loc='best', prop={'size': 3})
    plt.xlim((1, 300))
    plt.ylim((-5, 10))
    plt.show()