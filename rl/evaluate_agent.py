""""
Animation code source:
https://gist.github.com/DanielTakeshi/fec9a5cd957eb05b04b6d06a16cc88ae
"""

import argparse
import time
import imageio
from PIL import Image


import numpy as np
import torch as T

import gym
import rl.environments



def evaluate(agent, env, EE, max_el, exp_name, gif=False):
    print('[ Evaluation ]')

    EZ = [] # Evaluation episodic return
    ES = [] # Evaluation episodic score
    EL = [] # Evaluation episodic
    if gif: GifObs = []

    for ee in range(1, EE+1):
        print(f' [ Episode {ee}   Agent Evaluation ]                     ')
        o, d, Z, S, el = env.reset(), False, 0, 0, 0

        while not(d or (el == max_el)):
            print(f' [ Step {el}   Agent Simulation ]                     ', end='\r')
            if gif:
                gifobs = env.render(mode='rgb_array', width=400, height=400)
                GifObs.append(gifobs)
            # Take deterministic actions at evaluation time
            pi, _ = agent(o, deterministic=True)
            a = pi.cpu().numpy()
            o, r, d, info = env.step(a)
            Z += r
            S = 0# += info['score']
            el += 1
        EZ.append(Z)
        ES.append(S/el)
        EL.append(el)

    env.close()

    print('\nlen(GifObs): ', len(GifObs))

    if gif:
        print(' [ Saving a gif for evaluation ]     ')
        exp_path = f'./gifs/{exp_name}.gif'
        with imageio.get_writer(exp_path, mode='I', duration=0.01) as writer:
            for obs_np in GifObs:
                writer.append_data(obs_np)
        # print(' [ Saving a jpg for evaluation ]     ')
        # im = Image.fromarray(GifObs[50])
        # im.save(f'./jpgs/{exp_name}.jpeg')

    return EZ, ES, EL



def main(agent, env, alg, seed=0, epoch=0, metric='return', EE=10, gif=False):
    print('\n')


    print('=' * 50)
    print(f'Starting a new evaluation')
    print(f"\t Algorithm:   {alg}")
    print(f"\t Environment: {env}")
    print(f"\t Random seed: {seed}")
    print(f"\t Epoch: {epoch}")
    print(f"\t Metric: {metric}")
    print('=' * 50)

    exp_name = f'{env}-{alg}-seed:{seed}'


    eval_env = gym.make(env)
    # eval_env.seed(seed)
    # eval_env.action_space.seed(seed)
    # eval_env.observation_space.seed(seed)

    max_el = eval_env.env.spec.max_episode_steps

    logs = dict()
    agent.eval()
    eval_start_real = time.time()
    EZ, ES, EL = evaluate(agent, eval_env, EE, max_el, exp_name, gif)
    logs['time/evaluation'] = time.time() - eval_start_real

    if metric == 'score':
        logs['evaluation/episodic_score_mean'] = np.mean(ES)
        logs['evaluation/episodic_score_std'] = np.std(ES)
    else:
        logs['evaluation/episodic_return_mean'] = np.mean(EZ)
        logs['evaluation/episodic_return_std'] = np.std(EZ)
    logs['evaluation/episodic_length_mean'] = np.mean(EL)

    for k, v in logs.items():
        print(f'{k}: {round(v, 2)}')

    print('\n')
    print('End of the evaluation')
    print('=' * 50)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-env', type=str)
    parser.add_argument('-alg', type=str)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-EE', type=int)
    parser.add_argument('-metric', type=str)
    parser.add_argument('-gif', nargs='?', const=True, type=bool)

    args = parser.parse_args()

    agent_path = f'./saved_agents/{args.env}-{args.alg}-seed:{args.seed}-epoch:{args.epoch}' + '.pth.tar'
    agent = T.load(agent_path)

    kwaergs = vars(args)

    main(agent, **kwaergs)
