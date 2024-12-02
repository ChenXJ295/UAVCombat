import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor import PPOActor
import logging
import pandas as pd #用于数据输出
import numpy as np
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


num_agents = 2
render = True
# ego_policy_index = 17474
# enm_policy_index = 17474
# ego_policy_index = 17463
# enm_policy_index = 17463
# ego_policy_index = 17410
# enm_policy_index = 17410


#SFK
#NFSP
ego_policy_index = 17359
enm_policy_index = 17359
ego_policy_index = 4366
enm_policy_index = 17376

#psro
ego_policy_index = 4366
enm_policy_index = 17359

#do
ego_policy_index = 17474
enm_policy_index = 17410
episode_rewards = 0
ego_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/run-20240226_224228-fp1qorlp/files"
# enm_run_dir = "D:\NashAirCombat\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/artillery_check/wandb/latest-run/files"
enm_run_dir = "D:\\NashAirCombat\\CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/ppo/v1/wandb/run-20240226_224228-fp1qorlp/files"
experiment_name = ego_run_dir.split('/')[-4]

env = SingleCombatEnv("1v1/NoWeapon/Selfplay")
env.seed(0)
print(env)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))
list1=[]
i = 0
print("Start render")
obs = env.reset()
# if render:
#     env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
print(masks)
enm_obs = obs[num_agents // 2:, :]
ego_obs = obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
for i in range(300):
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)

    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs

    obs, rewards, dones, infos = env.step(actions)
    print(rewards)
    list1.append([i + 1, rewards[0][0]] )
    rewards = rewards[:num_agents // 2, ...]
    # rewards += np.random.uniform(1,2)
    print("reward:", rewards)
    # list1.append([i + 1, rewards[1][0]])
    i = i + 1
    episode_rewards += rewards
    # if render:
        # env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')

    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs = obs[num_agents // 2:, ...]
    ego_obs = obs[:num_agents // 2, ...]

print(episode_rewards)
column=['timestep','reward'] #列表头名称
test=pd.DataFrame(columns=column,data=list1)#将数据放进表格
test.to_csv('train//1.csv') #数据存入csv,存储位置及文件名称