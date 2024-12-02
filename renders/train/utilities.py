import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from matplotlib.font_manager import FontProperties
from pylab import mpl





def plot_learning_curve(reward_record, title=''):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('science')
    '''
    画出学习时的奖励曲线
    '''
    total_episodes = len(reward_record)
    x1 = range(total_episodes)
    plt.figure(figsize=(6, 4))
    plt.plot(x1, reward_record)
    plt.xlabel('Learning Steps')
    plt.ylabel('Episode Reward')
    plt.title(title)
    #plt.legend()
    plt.show()

def plot_compare_curve(reward, method, title='', isieee=True, xlabel='Episode', ylabel='', method_loc=(0.05,0.05)):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('science')
    sns.set(font='LiSu')
    '''
    画出多条曲线
    '''
    if isieee:
        plt.style.use(['science','ieee'])
    else:
        plt.style.use(['science'])
    total_episodes = len(reward[0])
    x1 = range(total_episodes)
    plt.figure(figsize=(6, 4))
    for i in reward:
        plt.plot(x1,i)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(method, loc=method_loc)
    plt.title(title)
    #plt.legend()
    plt.show()

def get_DataFrame(row_data, xlabel='episode', ylabel='y'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('science')
    '''
    将数据转为DataFrame类型
    '''
    x = np.arange(1, len(row_data[0])+1)
    row_data = np.array(row_data)
    temp_data = []
    for i in row_data:
        temp_data.append(pd.DataFrame(np.concatenate((x[:,None], i[:,None]), axis=1), columns=[xlabel,ylabel]))
    data = pd.concat([i for i in temp_data], axis=0)
    data[[xlabel, ylabel]] = data[[xlabel, ylabel]].apply(pd.to_numeric)

    return data

def plot_curve(*data, xlabel='episode', ylabel='y', label=['test'], title='', isieee=False):
    # 设置显示中文字体
    # mpl.rcParams["font.sans-serif"] = ["SimHei"]
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']

    # plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
    my_font = FontProperties(fname=r"C:\Users\wx\Desktop\SimHei.ttf", size=12)
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('science')
    '''
    画带标准差的图
    data：需要绘制的数据
    xlabel：X轴的名称
    '''
    # plt.rcParams['font.family']=['STFangsong']
    # zh_size = 8
    # eng_size = 9
    # num_size = 8
    # if isieee:
    #     plt.style.use(['science','ieee'])
    # else:
    #     plt.style.use(['science'])
    with plt.style.context(['science', 'no-latex', 'ieee']):
        palette = sns.color_palette("deep", 6)
        # 中文字体设置
        # font = FontProperties(fname="C:/Users/Fonts/simhei.ttf", size=12)
        for i in range(len(data)):
            # ax = sns.lineplot(x=xlabel,y=ylabel,label=label[i], data=data[i])
            ax = sns.lineplot(x=xlabel,y=ylabel,label=label[i], data=data[i],color=palette[i])
        ax.lines[1].set_linestyle('--')
        ax.lines[2].set_linestyle('dashdot')
        # ax.lines[0].set_color('blue')
        # ax.lines[2].set_color('green')
        # ax.lines[1].set_color('orange')
        # ax.lines[3].set_linestyle('--')
        # ax.lines[4].set_linestyle('dashdot')
        # ax.lines[4].set_color('purple')
        plt.legend(loc=0, fontsize='x-small')
        plt.xlabel('回合',fontproperties=my_font,size=5)
        plt.ylabel('每回合平均奖励',fontproperties=my_font,size=5)
        plt.title(title,fontproperties=my_font,size=5)
        plt.show()


def test():
    plt.rcParams['font.sans-serif']=['Adobe Fangsong Std']
    xpoints=np.array([1,2,3,4,5])
    ypoints=np.array([5,3,8,5,9])
    plt.plot(ypoints,'oc-')
    plt.title('折线图')
    plt.xlabel('博客写作时长')
    plt.ylabel('涨粉数')
    plt.show()





if __name__ == '__main__':
    test()
