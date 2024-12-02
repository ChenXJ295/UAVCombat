import argparse
import sys, copy
import yaml
import wandb
import collections.abc

def init_wandb(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=args.wandb_name,
        monitor_gym=True,
        save_code=True,
    )

class AttrDict(dict):
    """
    Change dictionary entries to class attributes, 
    then the property can be called with dict.attri rather than dict["attri"].
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def LoadYAML2Dict(yaml_file: str,
                  toAttr: bool = False,
                  mergeWith: str = 'D:/NashAirCombat/CloseAirCombat/confs/default.yaml',
                  confs = {}):
    """ A function loading the hyper-parameters in yaml file into a dictionary.

    :param yaml_file: the yaml file name
    :type yaml_file: str
    :param toAttr: if True, transform the configuration dictionary into a class,
        such that each hyperparameter can be called with class.attribute instead of dict['attribute']; defaults to False
    :type toAttr: bool, optional
    :param mergeWith: if not None, merge the loaded yaml (with overwritting priority) with the yaml given by this path;
    for example, merging with default yaml file will fill those missing entries with those in defaulf configurations.
    :type mergeDefault: string or None, optional
    :param confs: input a dictionary of configurations from outside the function, defaults to {}
    :type confs: dict, optional
    :return: a dictionary of configurations, including all hyper-parameters for environment, algorithm and training/testing.
    :rtype: dict
    """
    if mergeWith is not None:
        with open(mergeWith) as f:
            default = yaml.safe_load(f)
        confs = UpdateDictAwithB(confs, default, withOverwrite=False)

    with open(yaml_file + '.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    confs = UpdateDictAwithB(confs, loaded, withOverwrite=True)

    if toAttr:
        concat_dict = {
        }  # concatenate all types of arguments into one dictionary
        for k, v in confs.items():
            concat_dict.update(v)
        return AttrDict(concat_dict)
    else:
        return confs
        
def UpdateDictAwithB(
    A,
    B,
    withOverwrite: bool = True,
) -> None:
    """ Update the entries in dictionary A with dictionary B.

    :param A: a dictionary
    :type A: dict
    :param B: a dictionary
    :type B: dict
    :param withOverwrite: whether replace the same entries in A with B, defaults to False
    :type withOverwrite: bool, optional
    :return: none
    """
    # ensure original A, B is not changed
    A_ = copy.deepcopy(A)
    B_ = copy.deepcopy(B)
    if withOverwrite:
        InDepthUpdateDictAwithB(A_, B_)
    else:
        temp = copy.deepcopy(A_)
        InDepthUpdateDictAwithB(A_, B_)
        InDepthUpdateDictAwithB(A_, temp)

    return A_


def InDepthUpdateDictAwithB(
    A,
    B,
) -> None:
    """A function for update nested dictionaries A with B.

    :param A: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type A: dict
    :param B: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type B: dict
    :return: none
    """
    for k, v in B.items():
        if isinstance(v, collections.abc.Mapping):
            A[k] = InDepthUpdateDictAwithB(A.get(k, {}), v)
        else:
            A[k] = v
    return A


def get_parser_args():
    ''' deprecated '''
    parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')
    # env args
    parser.add_argument('--env', type=str, default=None, help='environment type and name')
    parser.add_argument('--num_envs', type=int, default=1, help='number of environments for parallel sampling')
    parser.add_argument('--ram', type=bool, default=False, help='use RAM observation')
    parser.add_argument('--render', type=bool, default=False, help='render the scene')
    parser.add_argument('--seed', type=str, default='random', help='random seed')
    parser.add_argument('--record_video', type=bool, default=False, help='whether recording the video')

    # agent args
    parser.add_argument('--algorithm', type=str, default=None, help='algorithm name')
    parser.add_argument('--algorithm_spec.dueling', type=bool, default=False, help='DQN: dueling trick')
    parser.add_argument('--algorithm_spec.replay_buffer_size', type=int, default=1e5, help='DQN: replay buffer size')
    parser.add_argument('--algorithm_spec.gamma', type=float, default=0.99, help='DQN: discount factor')
    parser.add_argument('--algorithm_spec.multi_step', type=int, default=1, help='DQN: multi-step return')
    parser.add_argument('--algorithm_spec.target_update_interval', type=bool, default=False, help='DQN: steps skipped for target network update')
    parser.add_argument('--algorithm_spec.eps_start', type=float, default=1, help='DQN: epsilon-greedy starting value')
    parser.add_argument('--algorithm_spec.eps_final', type=float, default=0.001, help='DQN: epsilon-greedy ending value')
    parser.add_argument('--algorithm_spec.eps_decay', type=float, default=5000000, help='DQN: epsilon-greedy decay interval')

    # train args
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for update')
    parser.add_argument('--max_episodes', type=int, default=50000, help='maximum episodes for rollout')
    parser.add_argument('--max_steps_per_episode', type=int, default=300, help='maximum steps per episode')
    parser.add_argument('--train_start_frame', type=int, default=0, help='start frame for training (not update when warmup)')
    parser.add_argument('--method', dest='marl_method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')

    # get all argument values
    parser_args = parser.parse_args()

    # get all default argument values
    defaults = vars(parser.parse_args([]))

    # get non-default argument values TODO
    print(defaults)
    print('parser args: ', parser_args)

    return parser_args

# def get_default_args(env, method):
#     # [env_type, env_name] = env.split('_', 1) # only split at the first '_'
#     yaml_file = f'confs/{env_type}_{env_name}_{method}'
#     args = LoadYAML2Dict(yaml_file, toAttr=True, mergeWith='confs/default.yaml')
#     return args
def get_default_args(env, scenario, alg):
    # [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    a = scenario.split("/")
    yaml_file = f'D:/NashAirCombat/CloseAirCombat/confs/{env}_{a[1]}_{alg}'
    args = LoadYAML2Dict(yaml_file, toAttr=True, mergeWith='D:/NashAirCombat/CloseAirCombat/confs/default.yaml')
    return args

def get_args():
    mapping_path = []
    parsed_ids = []
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--env':
            arg_env = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '--algorithm':
            arg_alg = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '--scenario':
            arg_scenario = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '--seed':
            arg_seed = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])
        if arg == '--exp':
            arg_exp = sys.argv[1:][i+1]
            parsed_ids.extend([i, i+1])

    default_args = get_default_args(arg_env, arg_scenario, arg_alg)
    print('default: ', default_args)

    # overwrite default with user input args
    for i, arg in enumerate(sys.argv[1:]):
        if i not in parsed_ids:
            if arg == '--help' or arg == '-h':
                print("help")
                exit()
            if arg.startswith('--'):
                mapping_path = arg[2:].split('.')
            else:
                ind = default_args
                for p in mapping_path[:-1]:
                    ind = ind[p]
                try:
                    ind[mapping_path[-1]] = eval(arg)
                except:
                    ind[mapping_path[-1]] = arg

    print(default_args)  # args after overwriting

    # initialize wandb if necessary
    # if default_args.wandb_activate:
    #     if len(default_args.wandb_project) == 0:
    #         default_args.wandb_project = '_'.join((default_args.env_type, default_args.env_name, default_args.marl_method))
    #     if len(default_args.wandb_group) == 0:
    #         default_args.wandb_group = ''
    #     if len(default_args.wandb_name) == 0:
    #         default_args.wandb_name = str(default_args.save_id)
    #     init_wandb(default_args)
        
    return default_args
