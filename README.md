# AMMI-RL
## AMMI, Deep RL, Fall 2021: RL Implementation for Continuous Control Tasks

[Project report](https://drive.google.com/file/d/1_0ocjyF_3m9hPVpZjwmDEuT3LTi8_Vs5/view?usp=sharing)

## Course and Project details
This Deep RL course was taught at **The African Master's in Machine Intelligence** [AMMI](https://aimsammi.org/) in Fall 2021. It was instructed by researchers at [DeepMind](https://deepmind.com/): *[Bilal Piot](https://scholar.google.com/citations?user=fqxNUREAAAAJ)*, *[Corentin Tallec](https://scholar.google.com/citations?user=OPKX4GgLCxIC)* and *[Florian Strub](https://scholar.google.com/citations?user=zxO5kccAAAAJ)*. This project is the coursework of Deep RL where we **Catalyst Agents** team trying to re-implement RL algorithm(s) for continuous control tasks. We chose three types of environments: easy, medium, and hard to run the algorithm(s). The course project meant to submit only one algorithm, but we plan to continue working on this repo making it an open project by this team of student from AMMI. This is why we're trying to make a modular repo to ease the re-implementation of future algorithms.



## Algorithms:
Algorithms we are re-implementing/plannning to re-implement:
1. Soft Actor-Critic (SAC) [Paper](https://arxiv.org/abs/1812.05905) (Done)

2. Model-Based Policy Optimization (MBPO) [Paper](https://arxiv.org/abs/1812.05905) (Now; Almost Done)

3. Natural Policy Gradient (NPG) [Paper](http://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf) (Now)

4. Model-Based Natural Policy Gradient (MB-NPG) [Paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2004.07804.pdf&sa=D&sntz=1&usg=AFQjCNHD-jeaBoWiamUhN0zP8bHLWsYysQ) (Now)

5. Probabilistic Ensembles with Trajectory Sampling (PETS) [Paper](https://arxiv.org/abs/1805.12114) (Next)

6. Model Predictive Actor-Critic (MoPAC) [Paper](https://arxiv.org/abs/2103.13842) (Next)

7. Model Predictive Control-Soft Actor Critic (MPC-SAC) [Paper](https://ieeexplore.ieee.org/document/9429677) (Next; Future work)




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


## Experiments and Results

In thoe following we evaluate our code on the following environments. For some reason gifs don't run fast on google drive, so we encourage you to download them from this [folder](https://drive.google.com/drive/folders/1l5ina4xFu-LdTMeuF0tfgmS6uooMjZqR?usp=sharing) at drive.

### Locomotion
Tasks we are evaluating:
1. HalfCheetah

<img src="https://drive.google.com/uc?export=view&id=1l6YA9hhnQRTReLh98McIXVIOkXD5U9uv" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />
  
  
2. Ant

<img src="https://drive.google.com/uc?export=view&id=1mKwzMDDtv8Dw_1tfwrvmgBtL_dl_1MQ5" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />


3. Walker2d

<img src="https://drive.google.com/uc?export=view&id=1Nm8bbqRZXxLopnsepq66wqmmcPLbL36K" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />


4. Hooper

<img src="https://drive.google.com/uc?export=view&id=17zdbI5zgzi-pdmrTGXmTNMWYhgQ02-7N" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />


### Manipulation
Tasks we are evaluating:

1. DClaw Valve Turning

<img src="https://drive.google.com/uc?export=view&id=1ux1nMubQZxmmI90UTfdJeIv_KZNrVNJt" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />


2. ShadowHand Cube Re-orientation

<img src="https://drive.google.com/uc?export=view&id=1FWFViyX35-OXwEEjWywzaFu6OUSBEJSy" style="width: 300px; max-width: 100%; height: 200" title="Click to enlarge picture" />


## Catalyst Agents Team, Group 2
(first name alphabetical order)
- [MohammedElfatih Salah](https://github.com/mohammedElfatihSalah)
- Rami Ahmed
- [Ruba Mutasim](https://github.com/ruba128)
- [Wafaa Mohammed](https://github.com/Wafaa014)



## Acknowledgement
This repo was inspired by many great repos, mostly the following ones:
- [SpinningUp](https://github.com/openai/spinningup)
- [Stabel Baselines](https://github.com/hill-a/stable-baselines)
- [RLKit](https://github.com/rail-berkeley/rlkit)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Youtube-Code-Repository](https://github.com/philtabor/Youtube-Code-Repository)
