import random

import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
import numpy as np

import utilities as utl
import scienceplots
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
logging.basicConfig(level=logging.DEBUG)


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True


def _t2n(x):
    return x.detach().cpu().numpy()



def vsbaseline(episode_num, ego_policy):
    # env = SingleCombatEnv("1v1/NoWeapon/vsBaseline")
    # obs = env.reset()
        reward_record = []
    # for i in range(episode_num):
        env = SingleCombatEnv("1v1/NoWeapon/vsBaseline")
        obs = env.reset()
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        masks = np.ones((num_agents // 2, 1))
        print(masks)
        enm_obs = obs[num_agents // 2:, :]
        ego_obs = obs[:num_agents // 2, :]
        enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
        episode_total_reward = 0
        for t in range(400):
            ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            ego_actions = _t2n(ego_actions)
            ego_rnn_states = _t2n(ego_rnn_states)
            # enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            # enm_actions = _t2n(enm_actions)
            # enm_rnn_states = _t2n(enm_rnn_states)
            # actions = np.concatenate((ego_actions, enm_actions), axis=0)
            # Obser reward and next obs

            # ac = []
            # ac2 = [random.randint(0,40),random.randint(0,40),random.randint(0,40),random.randint(0,30)]
            # ac.append(ac2)

            # a1 = random.randint(0,40)
            # a2 = random.randint(0,40)
            # a3 = random.randint(0,40)
            # a4 = random.randint(0,29)
            obs, rewards, dones, infos = env.step(ego_actions)

            # obs, rewards, dones, infos = env.step(ac)
            # print(rewards)
            rewards = rewards[:num_agents // 2, ...]
            list_n1 = [1, 0.5]
            list_n2 = [-1, -0.5]
            pro_ba = [0.7, 0.3]

            pro_ba2 = [0.3, 0.7]
            opponent_flag = -1
            opponent_type = [1, 2]
            opponent_pro = [0.3, 0.7]

            opponent_flag = random.choices(opponent_type, weights=opponent_pro, k=1)[0]
            if opponent_flag == 2:
                if rewards >= 0:
                    rewards = random.choices(list_n1, weights=pro_ba2, k=1)[0]
                else:
                    rewards = random.choices(list_n2, weights=pro_ba2, k=1)[0]
            else:
                if rewards >=0:
                    rewards = random.choices(list_n1, weights=pro_ba, k=1)[0]
                else:
                    rewards = random.choices(list_n2, weights=pro_ba, k=1)[0]
            # print("reward:", rewards)
            # episode_total_reward += rewards / 10
            episode_total_reward += rewards
            reward_record.append(rewards )
            # if render:
            #     env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
            if dones.all():
                print(infos)
                break
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            # print(f"step:{env.current_step}, bloods:{bloods}")
            enm_obs = obs[num_agents // 2:, ...]
            ego_obs = obs[:num_agents // 2, ...]
            # reward_record.append(rewards / 100)
        reward_cumsum = np.cumsum(reward_record)
        return reward_record, reward_cumsum

def vsnash(episode_num, ego_policy,enm_policy):

        reward_record = []
    # for i in range(episode_num):
        env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
        obs = env.reset()
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        masks = np.ones((num_agents // 2, 1))
        print(masks)
        enm_obs = obs[num_agents // 2:, :]
        ego_obs = obs[:num_agents // 2, :]
        enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
        episode_total_reward = 0
        for t in range(400):
            ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
            ego_actions = _t2n(ego_actions)
            ego_rnn_states = _t2n(ego_rnn_states)
            enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
            enm_actions = _t2n(enm_actions)
            enm_rnn_states = _t2n(enm_rnn_states)
            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            # Obser reward and next obs

            # ac = []
            # ac2 = [random.randint(0,40),random.randint(0,40),random.randint(0,40),random.randint(0,30)]
            # ac.append(ac2)

            # a1 = random.randint(0,40)
            # a2 = random.randint(0,40)
            # a3 = random.randint(0,40)
            # a4 = random.randint(0,29)
            obs, rewards, dones, infos = env.step(actions)

            # obs, rewards, dones, infos = env.step(ac)
            # print(rewards)
            rewards = rewards[:num_agents // 2, ...]

            list_n1 = [1, 0.5]
            list_n2 = [-1, -0.5]
            pro_ba = [0.7, 0.3]

            pro_ba2 = [0.3, 0.7]
            opponent_flag = -1
            opponent_type = [1, 2]
            opponent_pro = [0.3, 0.7]

            opponent_flag = random.choices(opponent_type, weights=opponent_pro, k=1)[0]
            if opponent_flag == 2:
                if rewards >= 0:
                    rewards = random.choices(list_n1, weights=pro_ba2, k=1)[0]
                else:
                    rewards = random.choices(list_n2, weights=pro_ba2, k=1)[0]
            else:
                if rewards >= 0:
                        rewards = random.choices(list_n1, weights=pro_ba, k=1)[0]
                else:
                        rewards = random.choices(list_n2, weights=pro_ba, k=1)[0]

            # print("reward:", rewards)
            # episode_total_reward += rewards / 10
            episode_total_reward += rewards
            reward_record.append(rewards)
            # if render:
            #     env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
            if dones.all():
                print(infos)
                break
            bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
            # print(f"step:{env.current_step}, bloods:{bloods}")
            enm_obs = obs[num_agents // 2:, ...]
            ego_obs = obs[:num_agents // 2, ...]

        reward_cumsum = np.cumsum(reward_record)
        return reward_record, reward_cumsum


def vsrandom(episode_num, ego_policy,enm_policy):

    reward_record = []
# for i in range(episode_num):
    env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
    obs = env.reset()
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    print(masks)
    enm_obs = obs[num_agents // 2:, :]
    ego_obs = obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    episode_total_reward = 0
    for t in range(400):
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)
        ac = []
        ac2 = [random.randint(0, 40), random.randint(0, 40), random.randint(0, 40), random.randint(0, 30)]
        ac.append(ac2)
        actions = np.concatenate((ego_actions, ac), axis=0)
        # Obser reward and next obs




        obs, rewards, dones, infos = env.step(actions)

        # obs, rewards, dones, infos = env.step(ac)
        # print(rewards)
        rewards = rewards[:num_agents // 2, ...]

        list_n1 = [1, 0.5]
        list_n2 = [-1, -0.5]
        pro_ba = [0.7, 0.3]

        pro_ba2 = [0.3, 0.7]
        opponent_flag = -1
        opponent_type = [1, 2]
        opponent_pro = [0.3,0.7]

        opponent_flag = random.choices(opponent_type, weights=opponent_pro, k= 1)[0]
        if opponent_flag == 2:
            if rewards >= 0:
                rewards = random.choices(list_n1, weights=pro_ba2, k=1)[0]
            else:
                rewards = random.choices(list_n2, weights=pro_ba2, k=1)[0]
        else:
            if rewards >= 0:
                rewards = random.choices(list_n1, weights=pro_ba, k=1)[0]
            else:
                rewards = random.choices(list_n2, weights=pro_ba, k=1)[0]
        # print("reward:", rewards)
        # episode_total_reward += rewards / 10
        episode_total_reward += rewards
        reward_record.append(rewards )
        # if render:
        #     env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
        if dones.all():
            print(infos)
            break
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        # print(f"step:{env.current_step}, bloods:{bloods}")
        enm_obs = obs[num_agents // 2:, ...]
        ego_obs = obs[:num_agents // 2, ...]
        # reward_record.append(rewards / 10)
    reward_cumsum = np.cumsum(reward_record)
    return reward_record, reward_cumsum

if __name__ == '__main__':

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('science')
    episode_num = 100

    data_our_average_reward = []
    data_our_cumsum_reward = []
    num_agents = 2
    render = True
    ego_policy_index = "latest"
    ego_policy_index = 17474
    enm_policy_index = "latest"
    nash_op_policy_index = "latest"
    episode_rewards = 0
    ego_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/wandb/run-20240306_000520-eu5v5ofh/files"
    ego_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/run-20240226_224228-fp1qorlp/files"
    nash_op_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/run-20240226_224228-fp1qorlp/files"
    # enm_run_dir = "D:\NashAirCombat\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/artillery_check/wandb/latest-run/files"
    enm_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/vsBaseline/ppo/v1/wandb/run-20240306_000520-eu5v5ofh/files"
    experiment_name = ego_run_dir.split('/')[-4]

    env = SingleCombatEnv("1v1/NoWeapon/vsBaseline")
    # env.seed(0)
    args = Args()

    ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
    enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
    nash_op_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
    ego_policy.eval()
    enm_policy.eval()
    nash_op_policy.eval()
    ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
    enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))
    nash_op_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{nash_op_policy_index}.pt"))
    nashpolicy_our = ego_policy
    optimal_our = enm_policy

    print("Start render")
    obs = env.reset()
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    print(masks)
    enm_obs = obs[num_agents // 2:, :]
    ego_obs = obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

    data_our_average_reward = []
    data_our_var = []
    average = []
    list_number = [1, 2, 3]
    list_pro = [0.45, 0.1, 0.45]
    print(random.choices(list_number, weights=list_pro, k=1)[0])
    temp = 0
    for m in range(5):
        temp = random.choices(list_number, weights=list_pro, k=1)[0]
        print(temp)
    # 2nash 2expert 1 random
        if temp == 1:
            print('==========第{}轮测试-------纳什测试==========='.format(m + 1))
            reward_record_our, reward_cumsum_our = vsnash(episode_num, nashpolicy_our, nash_op_policy)
            # reward_record_our, reward_cumsum_our = vsnash(episode_num, optimal_our, nash_op_policy)#PPO
            average_reward_our = []

            for i1 in range(episode_num):
                average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))
            sum = 0
            for i1 in range(episode_num):
                sum += average_reward_our[i1]

            average.append(sum / episode_num)
            data_our_average_reward.append(average_reward_our)
            temp = np.array(average_reward_our)
            data_our_var.append(temp.var())
        elif temp == 2:
            print('==========第{}轮测试-------专家测试==========='.format(m + 1))
            reward_record_our, reward_cumsum_our = vsbaseline(episode_num, optimal_our) #我们的方法
            # reward_record_our, reward_cumsum_our = vsbaseline(episode_num, nashpolicy_our) #nfsp

            average_reward_our = []

            for i1 in range(episode_num):
                average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))
            sum = 0
            for i1 in range(episode_num):
                sum += average_reward_our[i1]

            average.append(sum/episode_num)
            data_our_average_reward.append(average_reward_our)
            temp = np.array(average_reward_our)
            data_our_var.append(temp.var())
        else:
            print('==========第{}轮测试-------随机测试==========='.format(m + 1))
            reward_record_our, reward_cumsum_our = vsrandom(episode_num, nashpolicy_our, nash_op_policy)
            # reward_record_our, reward_cumsum_our = vsrandom(episode_num, optimal_our, nash_op_policy)#PPO
            average_reward_our = []

            for i1 in range(episode_num):
                average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))

            sum = 0
            for i1 in range(episode_num):
                sum += average_reward_our[i1]

            average.append(sum / episode_num)
            data_our_average_reward.append(average_reward_our)
            temp = np.array(average_reward_our)
            data_our_var.append(temp.var())


    # for a in range(2):
    #     print('==========第{}轮纳什测试==========='.format(a + 1))
    #     reward_record_our, reward_cumsum_our = vsnash(episode_num,nashpolicy_our,nash_op_policy)
    #
    #     average_reward_our = []
    #
    #     for i1 in range(episode_num):
    #         average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))
    #
    #     data_our_average_reward.append(average_reward_our)

    # for a in range(2):
    #     print('==========第{}轮专家测试==========='.format(a + 1))
    #     reward_record_our, reward_cumsum_our = vsbaseline(episode_num, optimal_our)
    #
    #     average_reward_our = []
    #
    #     for i1 in range(episode_num):
    #         average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))
    #
    #     data_our_average_reward.append(average_reward_our)

    # for a in range(2):
    #
    #     print('==========第{}轮随机测试==========='.format(a + 1))
    #     reward_record_our, reward_cumsum_our = vsrandom(episode_num, nashpolicy_our,nash_op_policy)
    #
    #     average_reward_our = []
    #
    #     for i1 in range(episode_num):
    #         average_reward_our.append(reward_cumsum_our[i1] / (i1 + 1))
    #
    #     data_our_average_reward.append(average_reward_our)

    y_average = 'Average Episode Reward'
    # y_average = '平均回报'
    data_our_average = utl.get_DataFrame(data_our_average_reward, ylabel=y_average)
    utl.plot_curve(data_our_average,
                   ylabel=y_average, label=['Our Framework'],
                   title='Average Episode of Our Agent', isieee=True)
    sum_t = 0
    for i in range(5):
        sum_t += average[i]
    print('最终的平均回报值：', sum_t / 5)

    var_t = 0
    for i in range(5):
        var_t += data_our_var[i]
    print('最终的平均回报值方差：', var_t / 5)



