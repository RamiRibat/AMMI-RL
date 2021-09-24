import sys
import argparse
import importlib
import datetime
import random

import torch as T
import wandb

from rl.mfrl.sac import SAC



def main(configs, seed):
    print('\n')
    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"iii-{env_type}-{env_name}"
    # now = datetime.datetime.now()
    # exp_prefix = f"{group_name}-{seed}--[{now.year}-{now.month}-{now.day}]-->{now.hour}:{now.minute}:{now.second}"
    exp_prefix = f"{group_name}-{alg_name}-seed:{seed}"

    print('=' * 50)
    print(f'Starting a new experiment')
    print(f"\t Algorithm:   {alg_name}")
    print(f"\t Environment: {env_name}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)

    configs['seed'] = seed

    if configs['experiment']['WandB']:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='rl-ammi-2',
            config=configs
        )

    agent = SAC(exp_prefix, configs, seed)
    agent.learn()

    # T.save(agent.actor_critic.actor,
    # f'./agents/agent-{env_name}-{alg_name}-seed:{seed}.pth.tar')

    print('\n')
    print('End of the experiment')
    print('=' * 50)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=int)

    args = parser.parse_args()

    sys.path.append("./configs")
    config = importlib.import_module(args.cfg)
    seed = args.seed

    main(config.configurations, seed)
