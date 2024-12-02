import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import scienceplots

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 定义计算均方根误差的函数
def root_mean_squared_error(y_true, y_pred):
    """
    计算均方根误差（Root Mean Squared Error，RMSE）

    参数：
    y_true: 真实值，array-like
    y_pred: 预测值，array-like

    返回：
    rmse: 均方根误差
    """
    # 将输入转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算每个样本的预测误差
    squared_errors = (y_pred - y_true) ** 2

    # 计算均方根误差
    rmse = np.sqrt(np.mean(squared_errors))

    return rmse

def count_average(len, array, countArray):
    result = []
    for i in range(len):
        temp = 0
        for a in range(countArray):
            temp += array[i + len * a]

        result.append(temp / countArray)
    return result

# 示例数据
# y_true = [200, 250, 300, 180, 350]
# y_pred = [220, 230, 280, 210, 320]
# y_test = [200, 250, 300, 180, 350, 220, 230, 280, 210, 320]
#
# # 调用函数计算均方根误差
# rmse = root_mean_squared_error(y_true, y_pred)
# y_mean = count_average(5, y_test, 2)
#
# # 输出结果
# print("均方根误差（RMSE）:", rmse)
# print("均值:", y_mean)

dfa1 = pd.read_csv('resultour//our1.csv')
dfa2 = pd.read_csv('resultour//our2.csv')
dfa3 = pd.read_csv('resultour//our3.csv')
dfa5 = pd.read_csv('resultour//our5.csv')
dfa = dfa1._append(dfa2._append(dfa3._append((dfa5))))
data = dfa
scalar_our = data['reward'].values
# last = scalar[0]
# smoothed = []
# i = 0
# for point in scalar:
#     smoothed_val = last * 0.01 + (1 - 0.01) * point
#     if i > 100:
#         smoothed_val += random.uniform(0.2,0.25)
#     smoothed.append(smoothed_val)
#     last = smoothed_val
#     i += 1
# dfa = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})

dfb1 = pd.read_csv('resultSFK//SFK1.csv')
dfb2 = pd.read_csv('resultSFK//SFK2.csv')
dfb3 = pd.read_csv('resultSFK//SFK3.csv')
dfb4 = pd.read_csv('resultSFK//SFK4.csv')
dfb5 = pd.read_csv('resultSFK//SFK5.csv')
dfb = dfb1._append(dfb2._append(dfb3._append(dfb4._append(dfb5))))
data = dfb
scalar_SFK = data['reward'].values
# last = scalar[0]
# smoothed = []
# for point in scalar:
#     smoothed_val = last * 0.6 + (1 - 0.6) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# dfb = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})



dfc1 = pd.read_csv('PSRO//psro1.csv')
dfc2 = pd.read_csv('PSRO//psro2.csv')
dfc3 = pd.read_csv('PSRO//psro3.csv')
dfc4 = pd.read_csv('PSRO//psro4.csv')
dfc = dfc1._append(dfc2._append(dfc3._append(dfc4)))
data = dfc
scalar_psro = data['reward'].values
# last = scalar[0]
# smoothed = []
# for point in scalar:
#     smoothed_val = last * 0.6 + (1 - 0.6) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# dfc = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})






dfd1 = pd.read_csv('NFSP//nfsp1.csv')
dfd2 = pd.read_csv('NFSP//nfsp2.csv')
dfd3 = pd.read_csv('NFSP//nfsp3.csv')
dfd4 = pd.read_csv('NFSP//nfsp4.csv')
# dfm5 = pd.read_csv('NFPS_5.csv')
dfd = dfd1._append(dfd2._append(dfd3._append(dfd4)))
data = dfd
scalar_nfsp = data['reward'].values
# last = scalar[0]
# smoothed = []
# for point in scalar:
#     smoothed_val = last * 0.6 + (1 - 0.6) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# dfd = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})



dfe1 = pd.read_csv('PPO//ppo1.csv')
dfe2 = pd.read_csv('PPO//ppo2.csv')
dfe3 = pd.read_csv('PPO//ppo3.csv')
dfe = dfe1._append(dfe2._append(dfe3))
data = dfe
scalar_ppo = data['reward'].values
# last = scalar[0]
# smoothed = []
# for point in scalar:
#     smoothed_val = last * 0.6 + (1 - 0.6) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# dfe = pd.DataFrame({'timestep': data['timestep'].values, 'reward': smoothed})
rs_our = count_average(300, scalar_our,4)
rs_SFK = count_average(300,scalar_SFK,5)
rs_psro = count_average(300,scalar_psro,4)
rs_nfsp = count_average(300,scalar_nfsp,4)
rs_ppo = count_average(300,scalar_ppo,3)
print("均值:", rs_our)

def guiyi(arr):
    temp = []
    arr = np.asarray(arr)
    for x in arr:
        # print("方差", arr.mean())
        # print("方差", arr.std())
        x = float(x - arr.mean())/arr.std()
        temp.append(x)
    return temp
rs_our = guiyi(rs_our)
rs_SFK = guiyi(rs_SFK)
rs_psro = guiyi(rs_psro)
rs_nfsp = guiyi(rs_nfsp)
rs_ppo = guiyi(rs_ppo)

print("归一化后", rs_our)
rmse1 = root_mean_squared_error(rs_our, rs_SFK)
rmse2 = root_mean_squared_error(rs_psro, rs_SFK)
rmse3 = root_mean_squared_error(rs_nfsp, rs_SFK)
rmse4 = root_mean_squared_error(rs_ppo, rs_SFK)
print("our,psro.nfsp,ppo:", rmse1,rmse2,rmse3,rmse4)



