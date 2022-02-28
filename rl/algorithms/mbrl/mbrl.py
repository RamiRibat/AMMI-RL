import gym
from gym.spaces import Box
from gym.wrappers import RecordVideo

import torch as T
# T.multiprocessing.set_sharing_strategy('file_system')

# from rl.data.buffer import ReplayBuffer
from rl.data.buffer import DataBuffer
from rl.data.dataset import RLDataModule
from rl.world_models.world_model import WorldModel





class MBRL:
    """
    Model-Based Reinforcement Learning
    """
    def __init__(self, exp_prefix, configs, seed):
        # super(MFRL, self).__init__(configs, seed)
        # print('init MBRL!')
        self.exp_prefix = exp_prefix
        self.configs = configs
        self.seed = seed


    def _build(self):
        self._set_env()
        self._set_env_buffer()
        self._set_data_module()
        self._set_world_model()


    def _set_env(self):
        name = self.configs['environment']['name']
        evaluate = self.configs['algorithm']['evaluation']

        # Inintialize Learning environment
        self.learn_env = gym.make(name)
        self._seed_env(self.learn_env)
        assert isinstance (self.learn_env.action_space, Box), "Works only with continuous action space"

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
        max_size = self.configs['data']['buffer_size']
        device = self.configs['experiment']['device']
        # self.env_buffer = ReplayBuffer(self.obs_dim, self.act_dim,
        #                                   max_size, self.seed, device)
        self.env_buffer = DataBuffer(self.obs_dim, self.act_dim, max_size, self.seed, device)


    def _set_data_module(self):
        self.data_module = RLDataModule(self.env_buffer, self.configs['data'])
        pass


    def _set_world_model(self):
        self.world_model = WorldModel(self.obs_dim, self.act_dim, self.rew_dim, self.configs, self.seed)


    def reallocate_model_buffer(self, batch_size_ro, K, NT, model_train_frequency):
        # print('Rellocate Model Buffer..')

        seed = self.seed
        device = self.configs['experiment']['device']
        model_retain_epochs = self.configs['world_model']['model_retain_epochs']

        rollouts_per_epoch = batch_size_ro * NT / model_train_frequency
        model_steps_per_epoch = int(K * rollouts_per_epoch)
        new_buffer_size = model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, 'model_buffer'):
        	print('[ MBRL ] Initializing new model buffer with size {:.2e}'.format(new_buffer_size))
        	self.model_buffer = DataBuffer(obs_dim=self.obs_dim,
        								act_dim=self.act_dim,
        								size=new_buffer_size,
        								seed=seed,
        								device=device)

        elif self.model_buffer.max_size != new_buffer_size:
        	new_model_buffer = DataBuffer(obs_dim=self.obs_dim,
        								act_dim=self.act_dim,
        								size=new_buffer_size,
        								seed=seed,
        								device=device)
        	old_data = self.model_buffer.return_all()
        	O, A, R, O_next, D = old_data.values()
        	new_model_buffer.store_batch(O, A, R, O_next, D)
        	assert self.model_buffer.size == new_model_buffer.size
        	# # Delete old data buffer and free GPU memory
        	# del self.model_buffer
        	# T.cuda.empty_cache()
        	self.model_buffer = new_model_buffer

        print(f'[ Model Buffer ] Size: {self.model_buffer.size}')


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


    def internact(self, n, o, Z, el, t):
        Nx = self.configs['algorithm']['learning']['expl_epochs']
        max_el = self.configs['environment']['horizon']

        if n > Nx:
            a, _ = self.actor_critic.actor.step_np(o)
        else:
            a = self.learn_env.action_space.sample()

        o_next, r, d, _ = self.learn_env.step(a)
        d = False if el == max_el else d # Ignore artificial termination

        self.env_buffer.store_transition(o, a, r, o_next, d)

        o = o_next
        Z += r
        el +=1
        t +=1

        if d or (el == max_el): o, Z, el = self.learn_env.reset(), 0, 0

        return o, Z, el, t


    def evaluate(self):
        evaluate = self.configs['algorithm']['evaluation']
        if evaluate:
            print('[ Evaluation ]'+(' '*100))
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
                    # pi, _ = self.actor_critic.actor(o, deterministic=True)
                    # a = pi.cpu().numpy()
                    a, _ = self.actor_critic.actor.step_np(o, deterministic=True)
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
