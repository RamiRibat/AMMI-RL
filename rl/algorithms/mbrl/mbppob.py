import os, subprocess, sys
import argparse
import importlib
import datetime
import random

import time
import wandb

import numpy as np
import torch as T
import torch.nn.functional as F

# T.multiprocessing.set_sharing_strategy('file_system')

from rl.algorithms.mbrl.mbrl import MBRL
from rl.algorithms.mfrl.ppo import PPO
from rl.dynamics.world_model import WorldModel
import rl.environments.mbpo.static as mbpo_static
# from rl.data.dataset import RLDataModule


class color:
	"""
	Source: https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
	"""
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'





class MBPPO(MBRL, PPO):
	def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
		super(MBPPO, self).__init__(exp_prefix, configs, seed, device)
		# print('init MBPPO Algorithm!')
		self.configs = configs
		self.seed = seed
		self._device_ = device
		self.WandB = wb
		self._build()


	## build MBPPO components: (env, D, AC, alpha)
	def _build(self):
		super(MBPPO, self)._build()
		self._set_ppo()
		self._set_ov_world_model()
		self.init_model_traj_buffer()


	## PPO
	def _set_ppo(self):
		PPO._build_ppo(self)


	def _set_ov_world_model(self):
		device = self._device_
		num_ensembles = self.configs['world_model']['num_ensembles']
		net_arch = self.configs['world_model']['network']['arch']

		self.world_model_local = [ WorldModel(self.obs_dim, self.act_dim, seed=0+m, device=device) for m in range(num_ensembles) ]
		# self.world_model_global = [ WorldModel(self.obs_dim, self.act_dim, seed=0+m, device=device) for m in range(num_ensembles) ]


	def learn(self):
		N = self.configs['algorithm']['learning']['epochs']
		NT = self.configs['algorithm']['learning']['epoch_steps']
		Ni = self.configs['algorithm']['learning']['init_epochs']
		Nx = self.configs['algorithm']['learning']['expl_epochs']

		E = self.configs['algorithm']['learning']['env_steps']
		G_WM = self.configs['algorithm']['learning']['grad_WM_steps']
		G_AC = self.configs['algorithm']['learning']['grad_AC_steps']
		G_PPO = self.configs['algorithm']['learning']['grad_PPO_steps']
		max_dev = self.configs['actor']['max_dev']

		global_step = 0
		start_time = time.time()
		logs = dict()
		lastEZ, lastES = 0, -2
		t = 0

		start_time_real = time.time()
		for n in range(1, N+1):
			if self.configs['experiment']['print_logs']:
				print('=' * 50)
				if n > Nx:
					print(f'\n[ Epoch {n}   Learning ]'+(' '*50))
					oldJs = [0, 0]
					JVList, JPiList, KLList = [], [], []
					HList, DevList = [], []
					ho_mean = 0
				elif n > Ni:
					print(f'\n[ Epoch {n}   Exploration + Learning ]'+(' '*50))
					JVList, JPiList, KLList = [], [], []
					HList, DevList = [], []
				else:
					print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
					oldJs = [0, 0]
					JVList, JPiList, KLList = [0], [0], [0]
					HList, DevList = [0], [0]
					ho_mean = 0

			nt = 0
			o, d, Z, el, = self.learn_env.reset(), 0, 0, 0
			ZList, elList = [0], [0]
			ZmeanImag, ZstdImag, ELmeanImag, ELstdImag = 0, 0, 0, 0
			AvgZ, AvgEL = 0, 0
			ppo_grads = 0

			learn_start_real = time.time()
			while nt < NT: # full epoch
				# Interaction steps
				for e in range(1, E+1):
					# o, Z, el, t = self.internact(n, o, Z, el, t)
					o, Z, el, t = self.internact_opB(n, o, Z, el, t, return_pre_pi=False)

					if el > 0:
						currZ = Z
						AvgZ = (sum(ZList)+currZ)/(len(ZList))
						currEL = el
						AvgEL = (sum(elList)+currEL)/(len(elList))
					else:
						lastZ = currZ
						ZList.append(lastZ)
						AvgZ = sum(ZList)/(len(ZList)-1)
						lastEL = currEL
						elList.append(lastEL)
						AvgEL = sum(elList)/(len(elList)-1)

					print(f'[ Epoch {n}   Interaction ] Env Steps: {e} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*10), end='\r')
				with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
				# self.buffer.traj_tail(d, v, el)
				self.buffer.finish_path(el, v)
				
				# Taking gradient steps after exploration
				if n > Ni:
					# 03. Train model pθ on Denv via maximum likelihood
					print(f'\n[ Epoch {n} | Training World Model ]'+(' '*50))

					ho_mean = self.train_world_model(n, local=True)

					for g in range(1, G_AC+1):
						ZmeanImag, ZstdImag, ELmeanImag, ELstdImag = self.rollout_world_model_trajectories_batch(g, n)
						ppo_batch_size = int(self.model_traj_buffer.total_size())
						stop_pi = False
						kl = 0
						dev = 0
						# print(f'\n\n[ Epoch {n}   Training Actor-Critic ({g}/{G}) ] Model Buffer: Size={self.model_traj_buffer.total_size()} | AvgK={self.model_traj_buffer.average_horizon()}'+(" "*25)+'\n')
						for gg in range(1, G_PPO+1): # 101
							# print(f'[ Epoch {n} ] AC: {g}/{G_AC} | ac: {gg}/{G_PPO} || stopPG={stop_pi} | KL={round(kl, 4)}'+(' '*50), end='\r')
							print(f"[ Epoch {n} | {color.RED}Training AC{color.END} ] AC: {g}/{G_AC} | ac: {gg}/{G_PPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
							batch = self.model_traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
							Jv, Jpi, kl, PiInfo = self.trainAC(g, batch, oldJs)
							oldJs = [Jv, Jpi]
							JVList.append(Jv)
							JPiList.append(Jpi)
							KLList.append(kl)
							HList.append(PiInfo['entropy'])
							DevList.append(PiInfo['deviation'])
							dev = PiInfo['deviation']
							if not self.stop_pi:
								ppo_grads += 1
							stop_pi = PiInfo['stop_pi']
						model_buffer_val = T.mean(self.model_traj_buffer.val_batch).item()
						model_buffer_ret = T.mean(self.model_traj_buffer.ret_batch).item()
						model_buffer_size = self.model_traj_buffer.total_size()
						# self.model_traj_buffer.reset()
						self.init_model_traj_buffer() # To spare some gpu-memory
						# del self.model_traj_buffer
					# PPO-P <<<<

				if n > Ni:
					LoG_PI = PiInfo['log_pi']
				else:
					LoG_PI = 0
					model_buffer_val = 0
					model_buffer_ret = 0
					model_buffer_size = 0

				nt += E

			print('\n')

			# logs['time/training                       '] = time.time() - learn_start_real

			# logs['training/wm/Jtrain_mean             '] = np.mean(JMeanTrainList)
			# logs['training/wm/Jtrain                  '] = np.mean(JTrainList)
			logs['training/wm/Jval                    '] = ho_mean
			# logs['training/wm/test_mse                '] = np.mean(LossTestList)

			logs['training/ppo/critic/Jv              '] = np.mean(JVList)
			logs['training/ppo/critic/V(s)            '] = model_buffer_val
			# logs['training/ppo/critic/V(s)            '] = T.mean(self.model_traj_buffer.val_buf).item()
			logs['training/ppo/critic/V-R             '] = model_buffer_val-model_buffer_ret
			# logs['training/ppo/critic/V-R             '] = T.mean(self.model_traj_buffer.val_buf).item()-T.mean(self.model_traj_buffer.ret_buf).item()

			logs['training/ppo/actor/Jpi              '] = np.mean(JPiList)
			# logs['training/ppo/actor/ac-grads         '] = ac_grads
			logs['training/ppo/actor/ppo-grads        '] = ppo_grads
			# logs['training/ppo/actor/H                '] = np.mean(HList)
			# logs['training/ppo/actor/KL               '] = np.mean(KLList)
			logs['training/ppo/actor/deviation        '] = np.mean(DevList)
			logs['training/ppo/actor/STD              '] = self.actor_critic.actor.std_value.clone().mean().item()
			logs['training/ppo/actor/log_pi           '] = LoG_PI


			logs['data/env_buffer_size                '] = self.buffer.total_size()
			# logs['data/env_rollout_steps              '] = self.buffer.average_horizon()
			if hasattr(self, 'model_traj_buffer'):
			    # logs['data/init_obs                       '] = 0. #len(self.buffer.init_obs)
			    # logs['data/model_buffer_size              '] = self.model_traj_buffer.total_size()
			    logs['data/model_buffer_size              '] = model_buffer_size
			    # logs['data/model_rollout_steps            '] = self.model_traj_buffer.average_horizon()
			else:
			    # logs['data/gae_lambda                '] = self.buffer.gae_lambda
			    logs['data/init_obs                       '] = 0.
			    logs['data/model_buffer_size              '] = 0.
			    logs['data/model_rollout_steps            '] = 0.
			# else:
			#     logs['data/model_buffer              '] = 0
			# logs['data/rollout_length            '] = K

			logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
			logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
			logs['learning/real/rollout_length        '] = np.mean(elList[1:])

			# logs['learning/imag/rollout_return_mean   '] = np.mean(ZListImag[1:])
			# logs['learning/imag/rollout_return_std    '] = np.std(ZListImag[1:])
			# logs['learning/imag/rollout_length        '] = np.mean(elListImag[1:])

			logs['learning/imag/rollout_return_mean   '] = ZmeanImag
			logs['learning/imag/rollout_return_std    '] = ZstdImag
			logs['learning/imag/rollout_length        '] = ELmeanImag

			eval_start_real = time.time()
			print('\n[ Evaluation ]')
			EZ, ES, EL = self.evaluate()
			# EZ, ES, EL = self.evaluateII()
			# EZ, ES, EL = self.evaluate_op()

			# logs['time/evaluation				'] = time.time() - eval_start_real

			if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
			    logs['evaluation/episodic_score_mean      '] = np.mean(ES)
			    logs['evaluation/episodic_score_std       '] = np.std(ES)
			else:
			    logs['evaluation/episodic_return_mean     '] = np.mean(EZ)
			    logs['evaluation/episodic_return_std      '] = np.std(EZ)
			logs['evaluation/episodic_length_mean     '] = np.mean(EL)
			logs['evaluation/return_to_length         '] = np.mean(EZ)/np.mean(EL)
			logs['evaluation/return_to_full_length    '] = (np.mean(EZ)/1000)

			logs['time/total                          '] = time.time() - start_time_real

			# if n > (N - 50):
			#	 if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
			#		 if np.mean(ES) > lastES:
			#			 print(f'[ Epoch {n}   Agent Saving ]					')
			#			 env_name = self.configs['environment']['name']
			#			 alg_name = self.configs['algorithm']['name']
			#			 T.save(self.actor_critic.actor,
			#			 f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pTtar')
			#			 lastES = np.mean(ES)
			#	 else:
			#		 if np.mean(EZ) > lastEZ:
			#			 print(f'[ Epoch {n}   Agent Saving ]					')
			#			 env_name = self.configs['environment']['name']
			#			 alg_name = self.configs['algorithm']['name']
			#			 T.save(self.actor_critic.actor,
			#			 f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pTtar')
			#			 lastEZ = np.mean(EZ)

			# Printing logs
			if self.configs['experiment']['print_logs']:
			    return_means = ['learning/real/rollout_return_mean   ',
			                    'learning/imag/rollout_return_mean   ',
			                    'evaluation/episodic_return_mean     ',
			                    'evaluation/return_to_length         ',
			                    'evaluation/return_to_full_length    ']
			    for k, v in logs.items():
			        if k in return_means:
			            print(color.PURPLE+f'{k}  {round(v, 4)}'+color.END+(' '*10))
			        else:
			            print(f'{k}  {round(v, 4)}'+(' '*10))

			# WandB
			if self.WandB:
				wandb.log(logs)

		self.learn_env.close()
		self.eval_env.close()


	def train_world_model(self, n, local=True):
		Ni = self.configs['algorithm']['learning']['init_epochs']
		G_WML = self.configs['algorithm']['learning']['grad_WML_steps']
		G_WMG = self.configs['algorithm']['learning']['grad_WMG_steps']
		model_fit_bs = min(self.configs['data']['local_buffer_size'], self.buffer.total_size())
		model_fit_batch = self.buffer.sample_batch(batch_size=model_fit_bs, recent=True, device=self._device_)
		s, _, a, sp, r, _, _, _, _, _ = model_fit_batch.values()

		if n == Ni+1:
			samples_to_collect = min((Ni+1)*1000, self.buffer.total_size())
		else:
			samples_to_collect = 1000

		LossGen = []

		for i, model in enumerate(self.world_model_local):
			loss_general = model.compute_loss(s[-samples_to_collect:],
											  a[-samples_to_collect:],
											  sp[-samples_to_collect:])
			dynamics_loss = model.fit_dynamics(s, a, sp, fit_mb_size=200, fit_epochs=G_WML)
			reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), fit_mb_size=200, fit_epochs=G_WML)
			LossGen.append(loss_general)

		ho_mean = np.mean(LossGen)

		return ho_mean


	def rollout_world_model_trajectories_q(self, g, n):
		# 07. Sample st uniformly from Denv
		device = self._device_
		Nτ = 250
		K = 1000

		O = O_init = self.buffer.sample_init_obs_batch(Nτ)
		O_Nτ = len(O_init)
		L = 5

		# 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
		ZList, elList = [0], [0]
		AvgZ, AvgEL = 0, 0

		for l in range(1, L+1):
			for nτ, oi in enumerate(O_init): # Generate trajectories
				o, Z, el = oi, 0, 0
				for k in range(1, K+1): # Generate rollouts
					print(f'[ Epoch {n} | AC {g} ] Model Rollout: L={l} | nτ={nτ+1}/{O_Nτ} | k={k}/{K} | Buffer={self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
					# print('\no: ', o)
					# print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
					with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_a_and_v(o)

					# o_next, r, d, _ = self.fake_world.step(o, a) # ip: Tensor, op: Tensor
					o_next, r, d, _ = self.fake_world.step(o, a, deterministic=True) # ip: Tensor, op: Tensor

					Z += float(r)
					el += 1
					self.model_traj_buffer.store(o, a, r, o_next, v, log_pi, el)
					o = o_next

					currZ = Z
					AvgZ = (sum(ZList)+currZ)/(len(ZList))
					currEL = el
					AvgEL = (sum(elList)+currEL)/(len(elList))

					if d or (el == K):
						break

				if el == K:
					with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
				else:
					v = T.Tensor([0.0])
				self.model_traj_buffer.finish_path(el, v)

				lastZ = currZ
				ZList.append(lastZ)
				AvgZ = sum(ZList)/(len(ZList)-1)
				lastEL = currEL
				elList.append(lastEL)
				AvgEL = sum(elList)/(len(elList)-1)

				if self.model_traj_buffer.total_size() >= self.configs['data']['model_buffer_size']:
					# print(f'Breaking img rollouts at L={l}/nτ={nτ+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
					break

			if self.model_traj_buffer.total_size() >= self.configs['data']['model_buffer_size']:
				print(f'Breaking img rollouts at L={l}/nτ={nτ+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
				break

		return ZList, elList


	def rollout_world_model_trajectories(self, g, n):
		# 07. Sample st uniformly from Denv
		device = self._device_
		Nτ = self.configs['data']['init_obs_size']
		K = 1000

		O = O_init = self.buffer.sample_init_obs_batch(Nτ)
		O_Nτ = len(O_init)

		# 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
		k_end_total = 0
		ZList, elList = [0], [0]
		AvgZ, AvgEL = 0, 0

		for nτ, oi in enumerate(O_init): # Generate trajectories
			for m, model in enumerate(self.world_model_local):
				o, Z, el = oi, 0, 0
				for k in range(1, K+1): # Generate rollouts
					print(f'[ Epoch {n} | {color.RED}AC {g}{color.END} ] Model Rollout: nτ = {nτ+1} | M = {m+1}/{len(self.world_model_local)} | k = {k}/{K} | Buffer = {self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
					# print('\no: ', o)
					# print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
					# with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_a_and_v(o, on_policy=True, return_pre_pi=True)
					with T.no_grad(): pre_a, a, log_pi, _, v = self.actor_critic.get_a_and_v(o, on_policy=True, return_pre_pi=True)

					o_next = model.forward(o, a).detach().cpu() # ip: Tensor, op: Tensor
					r = model.reward(o, a).detach()
					d = self._termination_fn("Hopper-v2", o, a, o_next)
					d = T.tensor(d, dtype=T.bool)

					Z += float(r)
					el += 1
					self.model_traj_buffer.store(o, pre_a, a, r, o_next, v, log_pi, el)
					o = o_next

					currZ = Z
					AvgZ = (sum(ZList)+currZ)/(len(ZList))
					currEL = el
					AvgEL = (sum(elList)+currEL)/(len(elList))

					if d or (el == K):
						break

				if el == K:
					with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
				else:
					v = T.Tensor([0.0])
				self.model_traj_buffer.finish_path(el, v)

				k_end_total += k

				lastZ = currZ
				ZList.append(lastZ)
				AvgZ = sum(ZList)/(len(ZList)-1)
				lastEL = currEL
				elList.append(lastEL)
				AvgEL = sum(elList)/(len(elList)-1)

			if self.model_traj_buffer.total_size() >= self.configs['data']['ov_model_buffer_size']:
				# print(f'[ Epoch {n} | AC {g} ] Breaking img rollouts at nτ={nτ+1}/m={m+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*40)
				break
		print(f'[ Epoch {n} | AC {g} ] RollBuffer={self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | L={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*35)

		# EZ, ES, EL = self.evaluate()
		#
		# print(color.RED+f'[ Epoch {n} | AC {g} ] Inner Evaluation | Z={round(np.mean(EZ), 2)}±{round(np.std(EZ), 2)} | L={round(np.mean(EL), 2)}±{round(np.std(EL), 2)} | x{round(np.mean(EZ)/np.mean(EL), 2)}'+color.END+(' ')*40+'\n')

		return ZList, elList


	def rollout_world_model_trajectories_batch(self, g, n, local=True):
		# 07. Sample st uniformly from Denv
		device = self._device_
		Nτ = self.configs['data']['init_obs_size']
		K = self.configs['world_model']['rollout_length']

		O_init = self.buffer.sample_init_obs_batch(Nτ)
		O_Nτ = len(O_init)
		D_init = T.zeros((O_Nτ, 1), dtype=T.bool).to(O_init.device)

		Zi, ELi = T.zeros((O_Nτ, 1), dtype=T.float32), T.zeros((O_Nτ, 1), dtype=T.float32)
		ZList, ELList = T.zeros((O_Nτ, 4), dtype=T.float32), T.zeros((O_Nτ, 4), dtype=T.float32)
		zeros, ones = T.zeros((O_Nτ, 1), dtype=T.float32), T.ones((O_Nτ, 1), dtype=T.float32)
		O_zeros, A_zeros = T.zeros((O_Nτ, self.obs_dim), dtype=T.float32), T.zeros((O_Nτ, self.act_dim), dtype=T.float32)
		R_zeros, V_zeros = T.zeros((O_Nτ, 1), dtype=T.float32), T.zeros((O_Nτ, 1), dtype=T.float32)
		log_Pi_zeros = T.zeros((O_Nτ, 1), dtype=T.float32)

		if local:
			world_model = self.world_model_local

		for m, model in enumerate(world_model):
			el = 0
			O = O_init.clone()

			O_last = O_init.clone()
			D_last = D_init.clone()

			Z, EL = Zi.clone(), ELi.clone()
			Zmean, ELmean = 0, 0

			for k in range(1, K+1): # Generate rollouts
				# print(f'\nk={k}\n')
				with T.no_grad(): pre_A, A, log_Pi, _, V = self.actor_critic.get_a_and_v(O, on_policy=True, return_pre_pi=True)

				O_next = model.forward(O, A).detach().cpu() # ip: Tensor, op: Tensor
				R = model.reward(O, A).detach().cpu()
				D = self._termination_fn("Hopper-v2", O, A, O_next)
				D = T.tensor(D, dtype=T.bool).squeeze(-1)
				D_last = D_last.squeeze(-1)
				# print(f'O=\n{O}')
				# print(f'D_last(old)=\n{D_last}')
				# print(f'D=\n{D}')

				nonD_last = ~D_last.squeeze(-1)


				O[D_last] = O_zeros[D_last] # del fake O to proceed only with non-terminated
				pre_A[D_last] = A_zeros[D_last] # del fake O to proceed only with non-terminated
				A[D_last] = A_zeros[D_last] # del fake O to proceed only with non-terminated
				O_next[D_last] = O_zeros[D_last] # del fake O to proceed only with non-terminated
				R[D_last] = R_zeros[D_last] # del fake O to proceed only with non-terminated
				V[D_last] = V_zeros[D_last] # del fake O to proceed only with non-terminated
				log_Pi[D_last] = log_Pi_zeros[D_last] # del fake O to proceed only with non-terminated


				O_last[nonD_last] = O_next[nonD_last] # update only non-terminated
				D_last[nonD_last] += D[nonD_last] # new D_last

				Z[nonD_last] += R[nonD_last]
				EL[nonD_last] += ones[nonD_last]
				# print(f'Z=\n{Z}')
				# print(f'EL=\n{EL}')
				# print(f'O_next=\n{O_next}')
				# print(f'D_last(new)=\n{D_last}')
				el += 1

				Zmean, ELmean = float(Z.mean().numpy()), float(EL.mean().numpy())
				self.model_traj_buffer.store_batch(O, pre_A, A, R, O_next, V, log_Pi, el)

				nonD = ~D.squeeze(-1)
				nonD_last = ~D_last.squeeze(-1)
				O = O_next

				# print(f'O_last=\n{O_last}')

				print(f'[ Epoch {n} | Model Rollout for {color.RED}AC {g}{color.END} ] M = {m+1}/{len(self.world_model_local)} | k = {k}/{K} | Buffer = {self.model_traj_buffer.total_size()} | AvgZ={round(Zmean, 2)} | AvgEL={round(ELmean, 2)}', end='\r')

				if nonD.sum() == 0:
					# print(f'\n[ Epoch {n} Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}'+(' ')*50)
					break

			V = T.zeros((O_Nτ, 1))
			if k == K:
				# print(f'\nCalc term Val at EL={EL}')
				x = (EL==K)
				with T.no_grad(): V_ = self.actor_critic.get_v(T.Tensor(O_last)).cpu()
				V[x] = V_[x]

			self.model_traj_buffer.finish_path_batch(EL, V)

			ZList[:,m] = Z.reshape(-1)
			ELList[:,m] = EL.reshape(-1)

		ZMEAN, ZSTD = float(ZList.mean().numpy()), float(ZList.std().numpy())
		ELMEAN, ELSTD = float(ELList.mean().numpy()), float(ELList.std().numpy())

		print(f'[ Epoch {n} | {color.RED}AC {g}{color.END} ] RollBuffer={self.model_traj_buffer.total_size()} | Z={round(ZMEAN, 2)}±{round(ZSTD, 2)} | L={round(ELMEAN, 2)}±{round(ELSTD, 2)} | x{round(ZMEAN/ELMEAN, 2)}'+(' ')*35)

		return ZMEAN, ZSTD, ELMEAN, ELSTD


	def _reward_fn(self, env_name, obs, act):
		next_obs = next_obs.numpy()
		if len(obs.shape) == 1 and len(act.shape) == 1:
			obs = obs[None]
			act = act[None]
			return_single = True
		elif len(obs.shape) == 1:
			obs = obs[None]
			return_single = True
		else:
			return_single = False

		next_obs = next_obs.cpu().numpy()

		if env_name == "Hopper-v2":
			assert len(obs.shape) == len(act.shape) == 2
			vel_x = obs[:, -6] / 0.02
			power = np.square(act).sum(axis=-1)
			height = obs[:, 0]
			ang = obs[:, 1]
			alive_bonus = 1.0 * (height > 0.7) * (np.abs(ang) <= 0.2)
			rewards = vel_x + alive_bonus - 1e-3*power

			return rewards
		elif env_name == "Walker2d-v2":
			assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
			pass
		elif 'walker_' in env_name:
			pass


	def _termination_fn(self, env_name, obs, act, next_obs):
		if len(obs.shape) == 1 and len(act.shape) == 1:
			obs = obs[None]
			act = act[None]
			next_obs = next_obs[None]
			return_single = True
		elif len(obs.shape) == 1:
			obs = obs[None]
			next_obs = next_obs[None]
			return_single = True
		else:
			return_single = False

		next_obs = next_obs.cpu().numpy()

		if env_name == "Hopper-v2":
			assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

			height = next_obs[:, 0]
			angle = next_obs[:, 1]
			not_done = np.isfinite(next_obs).all(axis=-1) \
					   * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
					   * (height > .7) \
					   * (np.abs(angle) < .2)

			done = ~not_done
			done = done[:, None]
			return done
		elif env_name == "Walker2d-v2":
			assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

			height = next_obs[:, 0]
			angle = next_obs[:, 1]
			not_done = (height > 0.8) \
					   * (height < 2.0) \
					   * (angle > -1.0) \
					   * (angle < 1.0)
			done = ~not_done
			done = done[:, None]
			return done
		elif 'walker_' in env_name:
			torso_height =  next_obs[:, -2]
			torso_ang = next_obs[:, -1]
			if 'walker_7' in env_name or 'walker_5' in env_name:
				offset = 0.
			else:
				offset = 0.26
			not_done = (torso_height > 0.8 - offset) \
					   * (torso_height < 2.0 - offset) \
					   * (torso_ang > -1.0) \
					   * (torso_ang < 1.0)
			done = ~not_done
			done = done[:, None]
			return done









def main(exp_prefix, config, seed, device, wb):

	print('Start an MBPPOB experiment...')
	print('\n')

	configs = config.configurations

	if seed:
		random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

	alg_name = configs['algorithm']['name']
	alg_mode = configs['algorithm']['mode']
	env_name = configs['environment']['name']
	env_type = configs['environment']['type']
	wm_epochs = configs['algorithm']['learning']['grad_WM_steps']
	DE = configs['world_model']['num_ensembles']

	group_name = f"{env_name}-{alg_name}-1" # Local
	# group_name = f"{env_name}-{alg_name}-GCP-0" # GCP
	exp_prefix = f"seed:{seed}"

	if wb:
		wandb.init(
			name=exp_prefix,
			group=group_name,
			# project='test',
			# project='AMMI-RL-2022',
			project=f'AMMI-RL-{env_name}',
			config=configs
		)

	agent = MBPPO(exp_prefix, configs, seed, device, wb)

	agent.learn()

	print('\n')
	print('... End the MBPPO experiment')


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('-exp_prefix', type=str)
	parser.add_argument('-cfg', type=str)
	parser.add_argument('-seed', type=str)
	parser.add_argument('-device', type=str)
	parser.add_argument('-wb', type=str)

	args = parser.parse_args()

	exp_prefix = args.exp_prefix
	sys.path.append(f"{os.getcwd()}/configs")
	config = importlib.import_module(args.cfg)
	seed = int(args.seed)
	device = args.device
	wb = eval(args.wb)

	main(exp_prefix, config, seed, device, wb)
