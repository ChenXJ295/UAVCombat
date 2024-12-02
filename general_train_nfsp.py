# from mars.utils.func import LoadYAML2Dict
from scripts.train.launchNash import make_train_env, make_eval_env
from rollout import rollout
from rl.agents import *
from rl.agents.multiagent import MultiAgent
# from mars.utils.args_parser import get_args
from common.args_parser import get_args, init_wandb
# from mars.utils.data_struct import AttrDict

def launch():
    args = get_args()
    print('args: ', args)

    ### Create env
    env = make_train_env(args)
    print(env)
    env.agents = ['first_0', 'second_0']
    ### Specify models for each agent     
    model1 = eval(args.algorithm)(env, args)
    model2 = eval(args.algorithm)(env, args)

    model = MultiAgent(env, [model1, model2], args)

    ### Rollout
    rollout(env, model, args, args.save_id)

if __name__ == '__main__':
    launch()  # vars: Namespace -> dict