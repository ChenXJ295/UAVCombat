# -*- coding: utf-8 -*-
import random

import numpy as np
import scienceplots

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import  matplotlib
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
label = []
test = []
for point in scalar:
    test.append('UAV Aerial Combat')
    label.append('Our Framework')
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfa = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed, 'label':label,'test':test})

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
label = []
test = []
for point in scalar:
    test.append('UAV Aerial Combat')
    label.append('Without Opponent Distinguish')
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfb = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed, 'label':label,'test':test})



dfc1 = pd.read_csv('dp//dp1.csv')
dfc2 = pd.read_csv('dp//dp2.csv')
dfc3 = pd.read_csv('dp//dp3.csv')
dfc4 = pd.read_csv('dp//dp4.csv')
dfc = dfc1._append(dfc2._append(dfc3._append(dfc4)))
data = dfc
scalar = data['reward'].values
last = scalar[0]
smoothed = []
label = []
test = []
for point in scalar:
    test.append('UAV Aerial Combat')
    label.append('Without Policy Judgment')
    smoothed_val = last * 0.3 + (1 - 0.3) * point
    smoothed.append(smoothed_val)
    last = smoothed_val
dfc = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed, 'label':label, 'test':test})










with plt.style.context(['science', 'ieee','grid']):
    palette = sns.color_palette("deep", 6)
    data_grid = pd.read_csv('C:\\Users\\wx\\Desktop\\小论文\\ab_data\\grid_av.csv')
    data_box = pd.read_csv('C:\\Users\\wx\\Desktop\\小论文\\ab_data\\boxing_av.csv')
    data_pong = pd.read_csv('C:\\Users\\wx\\Desktop\\小论文\\ab_data\\pong_av.csv')
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
    data = dfa._append(dfb._append(dfc))
    data = data._append(data_grid._append(data_box)._append(data_pong))
    print(data['reward'].values)
    print(data['label'].values)
    # sns.barplot(data=data, y="reward", x="test", hue='label', errorbar=('ci', 0.1), palette=[palette[0],palette[1],palette[2]],capsize=.01, width= 0.8)
    sns.barplot(data=data, y="reward", x="test", hue='label',  palette=[palette[0],palette[1],palette[2]],capsize=.01, width= 0.8, order=['Grid World','Boxing Game','Pong Game','UAV Aerial Combat'])
    # sns.set(font_scale=3)

    # sns.barplot(data=dfa,  y="reward", color='black', label='Our Framework', errorbar='sd', hue="label")
    # sns.barplot(data=dfb,  y="reward", color='blue', label='Without Opponent Distinguish', errorbar='sd',hue="label")
    # sns.barplot(data=dfc, y="reward", color='red', label='Without Policy Judgment', errorbar='sd',hue="label")
    plt.title('Average Episode Reward of Our Agent', fontsize=6)
    # plt.xlabel(fontsize=3)
    plt.xlabel('Scenario', fontsize = 8)
    plt.xticks(fontsize = 6)
    # plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴

    plt.ylabel('Average Episode Reward')
    plt.legend( loc='best', prop={'size': 4})
    # plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    ax = plt.gca()
    # ax.set_xticklabels(fontsize=3)
    # ax.get_xaxis().set_visible(False)
    plt.show()