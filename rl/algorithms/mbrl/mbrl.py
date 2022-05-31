import gym
from gym.spaces import Box

import torch as T
# T.multiprocessing.set_sharing_strategy('file_system')

from rl.data.buffer import TrajBuffer, ReplayBuffer
# from rl.data.buffer import DataBuffer
from rl.data.dataset import RLDataModule
# from rl.world_models.world_model import WorldModel
from rl.world_models.model import EnsembleDynamicsModel
from rl.dynamics.world_model import WorldModel





class MBRL:
    """
    Model-Based Reinforcement Learning
    """
    def __init__(self, exp_prefix, configs, seed, device):
        # super(MFRL, self).__init__(configs, seed)
        # print('init MBRL!')
        self.exp_prefix = exp_prefix
        self.configs = configs
        self.seed = seed
        self._device_ = device


    def _build(self):
        self._set_env()
        self._set_env_buffer()
        # self._set_data_module()
        self._set_world_model()


    def _set_env(self):
        name = self.configs['environment']['name']
        evaluate = self.configs['algorithm']['evaluation']

        # Inintialize Learning environment
        self.learn_env = gym.make(name)
        self._seed_env(self.learn_env)
        assert isinstance (self.learn_env.action_space, Box), "Works only with continuous action space"

        if True:
            self.traj_env = gym.make(name)
            self._seed_env(self.traj_env)

        if evaluate:
            # Ininialize Evaluation environment
            self.eval_env = gym.make(name)
            self._seed_env(self.eval_env)
        else:
            self.eval_env = None

        # Spaces dimensions
        self.obs_dim = self.learn_env.observation_space.shape[0]
        self.act_dim = self.learn_env.action_space.shape[0]
        self.act_up_lim = self.learn_env.action_space.high
        self.act_low_lim = self.learn_env.action_space.low
        self.rew_dim = 1


    def _seed_env(self, env):
        seed = self.seed
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


    def _set_env_buffer(self):
        # max_size = self.configs['data']['buffer_size']
        # device = self._device_
        # self.env_buffer = ReplayBuffer(self.obs_dim, self.act_dim, max_size, self.seed, device)
        num_traj = 1000
        horizon = 1000
        max_size = self.configs['data']['buffer_size']
        device = self._device_
        if self.configs['algorithm']['on-policy']:
            self.buffer = TrajBuffer(self.obs_dim, self.act_dim, horizon, num_traj, max_size, self.seed, device)
        else:
            self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, max_size, self.seed, device)


    # def _set_data_module(self):
    #     self.data_module = RLDataModule(self.buffer, self.configs['data'])
    #     pass


    def _set_world_model(self):
        device = self._device_
        num_ensembles = self.configs['world_model']['num_ensembles']
        num_elites = self.configs['world_model']['num_elites']
        net_arch = self.configs['world_model']['network']['arch']
        # self.world_model = WorldModel(self.obs_dim, self.act_dim, self.rew_dim, self.configs, self.seed, device)
        self.world_model = EnsembleDynamicsModel(num_ensembles, num_elites,
                                                 self.obs_dim, self.act_dim, 1,
                                                 net_arch[0], use_decay=True, device=device)

        # self.world_model = WorldModel(self.obs_dim, self.act_dim, self.rew_dim, self.configs, self.seed, device)

        # self.models = [ WorldModel(self.obs_dim, self.act_dim, seed=0+m) for m in range(num_ensembles) ]


    def init_model_traj_buffer(self):
        # print('Initialize Model Buffer..')
        seed = self.seed
        device = self._device_

        if self.configs['algorithm']['on-policy']:
            num_traj = int(200*1.25)
            horizon = 500
            max_size = self.configs['data']['model_buffer_size']
            self.model_buffer = TrajBuffer(self.obs_dim, self.act_dim, horizon, num_traj, max_size, self.seed, device)


    def reallocate_model_buffer(self, batch_size_ro=None, K=None, NT=None, model_train_frequency=None):
        print('Rellocate Model Buffer..')
        seed = self.seed
        device = self._device_

        if self.configs['algorithm']['on-policy']:
            max_size = self.configs['data']['model_buffer_size']
            self.model_buffer = TrajBuffer(self.obs_dim, self.act_dim, max_size, self.seed, device)
        else:
            model_retain_epochs = self.configs['world_model']['model_retain_epochs']
            rollouts_per_epoch = batch_size_ro * (NT / model_train_frequency)
            model_steps_per_epoch = int(K * rollouts_per_epoch)
            new_buffer_size = model_retain_epochs * model_steps_per_epoch

            if not hasattr(self, 'model_buffer'):
            	print('[ MBRL ] Initializing new model buffer with size {:.2e}'.format(new_buffer_size)+(' '*50))
            	self.model_buffer = ReplayBuffer(obs_dim=self.obs_dim,
            								act_dim=self.act_dim,
            								size=new_buffer_size,
            								seed=seed,
            								device=device)

            elif self.model_buffer.max_size != new_buffer_size:
            	new_model_buffer = ReplayBuffer(obs_dim=self.obs_dim,
            								act_dim=self.act_dim,
            								size=new_buffer_size,
            								seed=seed,
            								device=device)
            	old_data = self.model_buffer.return_all_np()
            	O, A, R, O_next, D = old_data.values()
            	new_model_buffer.store_batch(O, A, R, O_next, D)
            	assert self.model_buffer.size == new_model_buffer.size
            	self.model_buffer = new_model_buffer

        print(f'[ Model Buffer ] Size: {self.model_buffer.max_size}'+(' '*80))


    def initialize_learning(self, NT, Ni):
        max_el = self.configs['environment']['horizon']

        o, Z, el, t = self.learn_env.reset(), 0, 0, 0

        if Ni < 1: return o, Z, el, t

        print(f'[ Initial exploaration ] Starting')
        for ni in range(1, Ni+1):
            print(f'[ Initial exploaration ] Epoch {ni}')
            nt = 0
            while nt < NT:
                # Random actions
                a = self.learn_env.action_space.sample()
                o_next, r, d, info = self.learn_env.step(a)
                d = True if el == max_el else d # Ignore artificial termination

                self.env_buffer.store_transition(o, a, r, o_next, d)

                o = o_next
                Z += r
                el +=1
                t +=1

                if d or (el == max_el): o, Z, el = self.learn_env.reset(), 0, 0

                nt += 1

        return o, Z, el, t


    def internact_op(self, n, o, d, Z, el, t):
        Nt = self.configs['algorithm']['learning']['epoch_steps']
        max_el = self.configs['environment']['horizon']

        # a = self.actor_critic.get_action_np(o)
        with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))

        o_next, r, d_next, _ = self.learn_env.step(a)
        Z += r
        el += 1
        t += 1

        self.buffer.store_transition(o, a, r, d, v, log_pi)

        if d_next or (el == max_el):
            # print(f'termination: {d_next}')
            with T.no_grad(): v_next = self.actor_critic.get_v(T.Tensor(o_next)).cpu()
            self.buffer.traj_tail(d_next, v_next, el)
            o_next, d_next, Z, el = self.learn_env.reset(), 0, 0, 0

        o, d = o_next, d_next

        return o, d, Z, el, t


    def internact_opB(self, n, o, Z, el, t):
        Nt = self.configs['algorithm']['learning']['epoch_steps']
        max_el = self.configs['environment']['horizon']
        # a = self.actor_critic.get_action_np(o)
        with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))
        o_next, r, d, _ = self.learn_env.step(a)
        Z += r
        el += 1
        t += 1
        self.buffer.store(o, a, r, o_next, v, log_pi, el)
        o = o_next
        if d or (el == max_el):
            if el == max_el:
                with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
            else:
                # print('v=0')
                v = T.Tensor([0.0])
            self.buffer.finish_path(el, v)
            o, Z, el = self.learn_env.reset(), 0, 0
        return o, Z, el, t


    def internact(self, n, o, Z, el, t):
        Nx = self.configs['algorithm']['learning']['expl_epochs']
        max_el = self.configs['environment']['horizon']

        if n > Nx:
            a = self.actor_critic.get_action_np(o) # Stochastic action | No reparameterization
        else:
            a = self.learn_env.action_space.sample()

        o_next, r, d, _ = self.learn_env.step(a)
        d = False if el == max_el else d # Ignore artificial termination

        self.buffer.store_transition(o, a, r, o_next, d)

        o = o_next
        Z += r
        el +=1
        t +=1

        if d or (el == max_el): o, Z, el = self.learn_env.reset(), 0, 0

        return o, Z, el, t


    def evaluate_op(self):
        evaluate = self.configs['algorithm']['evaluation']
        if evaluate:
            print('[ Evaluation ]')
            EE = self.configs['algorithm']['evaluation']['eval_episodes']
            max_el = self.configs['environment']['horizon']
            EZ = [] # Evaluation episodic return
            ES = [] # Evaluation episodic score
            EL = [] # Evaluation episodic length

            for ee in range(1, EE+1):
                print(f' [ Agent Evaluation ] Episode: {ee}   ', end='\r')
                o, d, Z, S, el = self.eval_env.reset(), False, 0, 0, 0
                while not(d or (el == max_el)):
                    # with T.no_grad(): a, _, _ = self.actor_critic.get_pi(T.Tensor(o))
                    # a = self.actor_critic.get_action_np(o)
                    a = self.actor_critic.get_action_np(o, deterministic=True) # MB-PPO
                    # a = self.actor_critic.get_action_np(o, deterministic=True)
                    o, r, d, info = self.eval_env.step(a)
                    Z += r
                    if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': S += info['score']
                    el += 1

                EZ.append(Z)
                if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': ES.append(S/el)
                EL.append(el)

            # if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #     for i in range(len(ES)):
            #         ES[i] /= EL[i]

        return EZ, ES, EL


    def evaluate(self):
        evaluate = self.configs['algorithm']['evaluation']
        if evaluate:
            print('[ Evaluation ]')
            EE = self.configs['algorithm']['evaluation']['eval_episodes']
            max_el = self.configs['environment']['horizon']
            EZ = [] # Evaluation episodic return
            ES = [] # Evaluation episodic score
            EL = [] # Evaluation episodic length

            for ee in range(1, EE+1):
                print(f' [ Agent Evaluation ] Episode: {ee}   ', end='\r')
                o, d, Z, S, el = self.eval_env.reset(), False, 0, 0, 0

                while not(d or (el == max_el)):
                    # Take deterministic actions at evaluation time
                    a = self.actor_critic.get_action_np(o, deterministic=True) # Deterministic action | No reparameterization
                    o, r, d, info = self.eval_env.step(a)
                    Z += r
                    if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': S += info['score']
                    el += 1

                EZ.append(Z)
                if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': ES.append(S/el)
                EL.append(el)

            # if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #     for i in range(len(ES)):
            #         ES[i] /= EL[i]

        return EZ, ES, EL
