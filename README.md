# AMMI-RL

## RL Implementation for Continuous Control

This project was initiated in the RL course of Fall 2021 at **The African Master's in Machine Intelligence** ([**AMMI**](https://aimsammi.org/)) as a course-project where we implemented the SAC algorithm ([Haarnoja et al.](https://arxiv.org/abs/1812.05905)) for continuous control tasks. It is now an open project where we care to design code bases and benchmarks for RL algorithms in order to ease the development of new algorithms. We are designing this repo based on existing repositories as well as original papers to produce better general implementations for a selected set of algorithms.


## Algorithms
Algorithms we are re-implementing/plannning to re-implement:

| Algorithms | Model | Value | On Policy | MPC | Progress | Reference |
| --- | --- | --- | --- | --- | :---: | --- |
| VPG | False | V(GAE) | True | False | 🟢 | [Sutton et al., 1999](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) |
| NPG | False | V(GAE) | True | False | 🔴 | [Kakade, 2001](http://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf) |
| PPO | False | V(GAE) | True | False | 🟢 | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf?ref=https://githubhelp.com) |
| SAC | False | 2xQ | False | False | 🟢 | [Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905) |
| PETS | True | None | None | True | 🔴 | [Chua et al., 2018](https://arxiv.org/abs/1805.12114) |
| MB-PPO | True | V(GAE) | True | False | 🟢 | Similar~[Rajeswaran et al., 2020](https://arxiv.org/abs/2004.07804) |
| MB-SAC | True | 2xQ | False | False | 🟢 | [Janner et al., 2019](https://arxiv.org/abs/1812.05905) |
| MOVOQ | True | N/A | N/A | N/A | 🟡 | N/A |
| MoPAC | True | 2xQ | False | True | 🟣 | [Morgan et al., 2021](https://arxiv.org/abs/2103.13842) |
| MPC-SAC | True | V(GAE)/2xQ | False | True | 🔴 | [Omer et al., 2021](https://ieeexplore.ieee.org/document/9429677) |

🟢 Done || 🟡 Now || 🟣 Next || 🔴 No plan

## Generalized Network Hyperparameters
We aim to finetune our implementations to work with a generalized set of hyperparametrs across different algorithms. We are working with the following hyperparameters in the mean time:

| ☑️ | Network | Arch | Act | LRate | MFOV | MFOQ | MBOV | MBOQ | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | Policy | [2x128] | Tanh | 3e-4 | 🟩 | 🟨 | 🟩 | 🟥 | |
| | Policy | [2x256] | ReLU | 3e-4 | 🟥 | 🟩 | ⬜️ | 🟩 | |
| ✅ | Policy | [2x256] | PReLU | 3e-4 | 🟩 | 🟩 | 🟩 | 🟦 | |
| | V | [2x128] | Tanh | 1e-3 | 🟩 | ⬜️ | 🟩 | ⬜️ | |
| | V | [2x128] | PReLU | 1e-3 | 🟩 | ⬜️ | 🟨 | ⬜️ | |
| ✅ | V | [2x256] | PReLU | 3e-4 | 🟩 | ⬜️ | 🟩 | ⬜️ | |
| | Q | [2x256] | ReLU | 3e-4 | ⬜️ | 🟩 | ⬜️ | 🟩 | |
| ✅ | Q | [2x256] | PReLU | 3e-4 | ⬜️ | 🟩 | ⬜️ | 🟦 | |
| ✅ | V-Model | [2x512] | ReLU | 1e-3 | ⬜️ | ⬜️ | 🟩 | 🟥 | |
| ✅ | Q-Model | [4x200] | Swish | 3e-4 | ⬜️ | ⬜️ | 🟥 | 🟩 | |

🟩 Best || 🟨 Good || 🟥 Bad || 🟦 In progress



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







## AMMI-RL Team
(last name alphabetical order) | contribution 
- [Rami Ahmed](https://github.com/RamiSketche) | VPG, PPO, SAC, MB{PPO, SAC}
- [Wafaa Mohammed](https://github.com/Wafaa014) | SAC
- [Ruba Mutasim](https://github.com/ruba128) | SAC
- [MohammedElfatih Salah](https://github.com/mohammedElfatihSalah) | SAC

## AMMI-RL Advisors
- Bilal Piot, Corentin Tallec and Florian Strub (During RL Course Fall 2021)
- Vlad Mnih, Eszter Vértes and Theophane Weber (During Rami's AMMI project)

## Acknowledgement
This repo was inspired by many great repos, mostly the following ones (not necessarily in order):
- [RLKit](https://github.com/rail-berkeley/rlkit)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [SpinningUp](https://github.com/openai/spinningup)
- [RL for MuJoCo](https://github.com/aravindr93/mjrl)
- [MBPO-PyTorch](https://github.com/Xingyu-Lin/mbpo_pytorch)
- [Stabel Baselines](https://github.com/hill-a/stable-baselines)
- [Youtube-Code-Repository](https://github.com/philtabor/Youtube-Code-Repository)
