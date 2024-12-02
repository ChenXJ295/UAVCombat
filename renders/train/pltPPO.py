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
def map(data, MIN, MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = np.max(data)  # 当前数据最大值
    d_max = np.min(data)  # 当前数据最小值
    return MIN + (MAX - MIN) / (d_max - d_min) * (data - d_min)

data = pd.read_csv('nashvsppo.csv')

scalar = data['reward'].values
a = map(scalar, 10,-20)
last = a[0]
smoothed1 = []
for point in scalar:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed1.append(smoothed_val)
    last = smoothed_val
dfa = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed1})
print(dfa)
data2 = pd.read_csv('trainPPO1.csv',index_col=0)
print(data2)
scalar2 = data2['reward'].values
a = map(scalar2, 10,-20)
last = a[0]
# last = scalar2[0]
smoothed = []
baseline = []
x = []
i = 0
for point in a:
    i = 0
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
    baseline.append(0)
    x.append(i)
dfb = pd.DataFrame({'timestep': data2['timestep'].values, 'reward': smoothed})
print(dfb)
print(data2['timestep'].values)

dfc = pd.read_csv('trainPPO.csv',index_col=0)
# 随机策略vsNE
data3= pd.read_csv('1.csv')

scalar3 = data3['reward'].values
a = map(scalar3, 10,-20)
last = a[0]
smoothed3 = []
for point in scalar3:
    smoothed_val = last * 0.6 + (1 - 0.6) * point
    smoothed3.append(smoothed_val)
    last = smoothed_val
dfd = pd.DataFrame({'timestep': data3['timestep'].values, 'reward': smoothed3})
with plt.style.context(['science', 'ieee','grid']):
    palette = sns.color_palette("deep", 6)
    fig, ax1 = plt.subplots()
    # sns.lineplot(data=save, x="timestep", y="reward", color=palette[0],  label='NFSP')
    # plt.xlim((1, 250))
#     plt.xscale('log')
#     sns.lineplot(data=dfa, x="timestep", y="reward", color=palette[0],label='Expert Policy')
    # plt.plot(x, baseline, color=palette[1],label='Expected Reward')
    # sns.lineplot(data=dfa, x="timestep", y="reward", color=palette[0])
    # sns.lineplot(data=dfc, x="timestep", y="reward", color=palette[2],  label='Expected Reward')
    sns.lineplot(data=dfb, x="timestep", y="reward", color=palette[1])

    # sns.lineplot(data=dfc, x="timestep", y="reward", color=palette[2], label='PSRO')
    # sns.lineplot(data=dfd, x="timestep", y="reward", color=palette[3], label='Stochastic Policy')
    # sns.lineplot(data=dfe, x="timestep", y="reward", color=palette[4], label='PPO')
    # x = [1, 2, 3, 4, 5, 6, 7,8]
    # x_index = ['1', '10', '100', '1000000', '2000000', '3000000','4000000','5000000']
    # ax1.plot(data2['timestep'].values, smoothed, color=palette[1])
    # plt.plot(data['timestep'].values, smoothed1, color=palette[0])
    # ax2 =ax1.twiny()
    # ax2.plot(data2['timestep'].values, smoothed, color=palette[1])

    # plt.title('Episode Reward', fontsize=6)
    # plt.title('Total Reward of Agent A vs Exepert')
    plt.xlabel('Episode')
    # plt.legend(loc='best', prop={'size': 3})
    plt.xlim((0, 4500000))

    # plt.plot(data2['timestep'].values, smoothed, color=palette[1])
    # _ = plt.xticks(x, x_index)
    plt.ylabel('Episode Reward')
    # plt.ylabel('Total Reward')
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.show()