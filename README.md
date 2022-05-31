# AMMI-RL: Implementation for Continuous Control


## Algorithms:
Algorithms we are re-implementing/plannning to re-implement:

🟢 Done || 🟡 Now || 🟣 Next || 🔴 No plan

| Algorithms | Model | Value | On Policy | MPC | Paper | Progress |
| --- | --- | --- | --- | --- | --- | :---: |
| VPG | False | V(GAE) | True | False | [NeurIPS](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | 🟢 |
| NPG | False | V(GAE) | True | False | [NeurIPS](http://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf) | 🔴 |
| PPO | False | V(GAE) | True | False | [Arxiv](https://arxiv.org/pdf/1707.06347.pdf?ref=https://githubhelp.com) | 🟢 |
| SAC | False | 2xQ | False | False | [Arxiv](https://arxiv.org/abs/1812.05905) | 🟢 |
| PETS | True | None | None | True | [Arxiv](https://arxiv.org/abs/1805.12114) | 🔴 |
| MB-Game | True | V | True | False | [Arxiv](https://arxiv.org/abs/2004.07804) | 🟡 |
| MBPO | True | 2xQ | False | False | [Arxiv](https://arxiv.org/abs/1812.05905) | 🟢 |
| MoPAC | True | 2xQ | False | True | [Arxiv](https://arxiv.org/abs/2103.13842) | 🟡 |
| MPC-SAC | True | V/2xQ | False | True | [IEEE](https://ieeexplore.ieee.org/document/9429677) | 🔴 |


## Experiments and Results

In thoe following we evaluate our code on the following environments. Download gifs from this Google drive [folder](https://drive.google.com/drive/folders/1l5ina4xFu-LdTMeuF0tfgmS6uooMjZqR?usp=sharing) at drive. Results are averaged across 3 random seeds, and smoothed with 0.75 Exponential Moving Average.


### Locomotion Tasks

| **Hopper-v2** | **Walker2d-v2** |
| :---: | :---: |
| <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/Hopper-v2.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> | <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/SAC-Walker2d-v2%20.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> |
| **HalfCheetah-v2** | **Ant-v2** |
| <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/SAC-HalfCheetah-v2.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> | <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/SAC-Ant-v2.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> |

### Manipulation Tasks

| **DClaw Valve Turning** | **ShadowHand Cube Re-orientation** |
| :---: | :---: |
| <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/SAC-DClawTurn.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> | <img src="https://github.com/RamiSketcher/AMMI-RL/blob/main/results/SAC-SHC.png" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" /> |



## How to use this code
### Installation
#### Ubuntu 20.04

Move into `AMMI-RL/` directory, and then run the following:

```
conda create -n ammi-rl python=3.8

pip install -e .

pip install numpy torch wandb gym
```

If you want to run MuJoCo Locomotion tasks, and ShadowHand, you should install [MuJoCo](http://www.mujoco.org/) first (it's open sourced until 31th Oct), and then install [mujoco-py](https://github.com/openai/mujoco-py):
```
sudo apt-get install ffmpeg

pip install -U 'mujoco-py<2.1,>=2.0'
```

If you are using A local GPU of Nvidia and want to record MuJoCo environments [issue link](https://github.com/openai/mujoco-py/issues/187#issuecomment-384905400), run:
```
unset LD_PRELOAD
```

#### MacOS

Move into `AMMI-RL/` directory, and then run the following:

```
conda create -n ammi-rl python=3.8

pip install -e .

pip install numpy torch wandb gym
```

If you want to run MuJoCo Locomotion tasks, and ShadowHand, you should install [MuJoCo](http://www.mujoco.org/) first (it's open sourced until 31th Oct), and then install [mujoco-py](https://github.com/openai/mujoco-py):
```
brew install ffmpeg gcc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

pip install -U 'mujoco-py<2.1,>=2.0'
```

If you are using A local GPU of Nvidia and want to record MuJoCo environments [issue link](https://github.com/openai/mujoco-py/issues/187#issuecomment-384905400), run:
```
unset LD_PRELOAD
```



### Run an experiment

Move into `AMMI-RL/` directory, and then:

```
python experiment.py -cfg <cfg_file-.py> -seed <int>
```
for example:

```
python experiment.py -cfg sac_hopper -seed 1
```

### Evaluate an Agent
To evaluate a saved policy model, run the following command:
```
python evaluate_agent.py -env <env_name> -alg <alg_name> -seed <int> -EE <int>
```
for example:

```
python evaluate_agent.py -env Walker2d-v2 -alg SAC -seed 1 -EE 5
```







## AMMI-RL Team, Group 2
(last name alphabetical order)
- [Rami Ahmed](https://github.com/RamiSketche) (VPG, PPO, SAC, MBPO)
- [Wafaa Mohammed](https://github.com/Wafaa014) (SAC)
- [Ruba Mutasim](https://github.com/ruba128) (SAC)
- [MohammedElfatih Salah](https://github.com/mohammedElfatihSalah) (SAC)



## Acknowledgement
This repo was inspired by many great repos, mostly the following ones:
- [SpinningUp](https://github.com/openai/spinningup)
- [Stabel Baselines](https://github.com/hill-a/stable-baselines)
- [RLKit](https://github.com/rail-berkeley/rlkit)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Youtube-Code-Repository](https://github.com/philtabor/Youtube-Code-Repository)
