import gym

from buffer import ReplayBuffer


"""
Notes:
    1. Two vertical steps between class methods
    2. (+1) on 'def _seed_env' (DONE)
    3. __init__ should contain only configs, and seed (DONE)
    4. Reduce info in class methods, ideally do a general one for the class
    5. self.interact: where's reset when termination
    6. policy always return 2 quantities: pi, log_pi (DONE)
    7. self.evaluate: we don't use detatch; because the policy module alreay detatch it from the comp.graph from within
    8. self.evaluate: where's 'Score'?
    9. Other notes start with '# !<note>'
"""


class MFRL:
    def __init__(self, config, seed): 
        print('Initialize MFRL!')
        self.config = config

        self.env_horizon = config['environment']['env_horizon']
        self.max_size = config['data']['buffer_size'] 
        self.seed = seed
        self.num_test_eps = config['evaluation']['eval_episodes']
        self._build()

    def _build(self):
        self._set_env()
        self._set_replay_buffer()

    def _seed_env(self, env, seed):
        '''
        helper method to seed an environement created by gym
        '''
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


    def _set_env(self):
        env_name = self.config['environment']['name']
        evaluate = self.config['algorithm']['evaluation']

        # Inintialize Learning environment:
        self.learn_env = gym.make(env_name)
        # seeding learn enviroment
        self._seed_env(self.learn_env, self.seed) 

        if evaluate:        
            # Ininialize Evaluation environment:
            self.eval_env = gym.make(env_name)
            # seeding evaluation environement
            self._seed_env(self.eval_env, self.seed) 

        # initialize observation dimension, actuator dimension, actuator lower limit and actuator upper limit
        self.obs_dim = self.learn_env.observation_space.shape
        self.act_dim = self.learn_env.action_space.shape[0]
        self.act_low_lim = self.learn_env.action_space.low
        self.act_up_lim = self.learn_env.action_space.high
            

    def _set_replay_buffer(self):
        buffer_size = self.config['data']['buffer_size']
        device = self.config['experiment']['device']
        self.buffer =  ReplayBuffer(self.obs_dim, self.act_dim, buffer_size, self.seed, device)

    def initialize_learning(self, init_epochs): # !This's for future modeifications
        self.state = self.learn_env.reset()

    
    def interact(self, policy, epoch_no, number_of_epochs_before_exploitation, state, accumulated_rewards, eps_len, t):
        '''
        act on the enviroment and return next state, and reward

        Parameters
        -------------------------------------------------------
        policy: a function that accepts state and returns action
        epoch_no: int, the current epoch number
        number_of_epochs_before_exploitation: int, a hyperparameter which represents the number of epochs
                                             the model will explore the environment before applying the policy
        state: the current state
        accumulated_rewards: float, accumulated rewards
        eps_len: int, current episode length
        t: int, the current timestep
    
        Returns
        -------------------------------------------------------
        next_state: the next state,
        new_accumulated_rewards:float, the reward after applying action
        new_eps_len: int, updated episode length
        d: bool, whether it's done or not
        t: int, updated timestep
        '''

        max_eps_len = self.config['environment']['horizon'] 

        # get the action
        action = None
        if epoch_no > number_of_epochs_before_exploitation:
            # get an action from the policy
            action = policy(state)[0].detach().cpu().numpy()
        else:
            # sample a random action
            action = self.learn_env.action_space.sample

        # apply the action on the environement
        next_state, reward, d , _ = self.learn_env.step(action)
        
        # stop forcefully if the episode length equal to the max episode length
        d = True if eps_len == max_eps_len else d  # Doesn't exist

        # store in the replay buffer
        self.buffer.store_transition(self.state, action, reward, next_state, d)

        

        new_accumulated_rewards = accumulated_rewards + reward
        new_eps_length  = eps_len + 1
        t +=1 

        #reset values based on d
        if d:
            next_state, new_accumulated_rewards, new_eps_length = self.learn_env.reset(), 0, 0

        return next_state, new_accumulated_rewards, new_eps_length, d, t

    
    #TODO: should we return the std also?
    def evaluate(self, policy):
        '''
        Parameters
        -------------------------------------------------------
        policy: a function that accepts state and returns action
        
        Returns
        -------------------------------------------------------
        average_evaluation_eps_return: float, the average evaluation for all episode
        '''
        print('.....Evaluation.....')
        env = self.eval_env
        eval_episodes = self.config['algorithm']['evaluation']['eval_episodes']
        max_eps_len = self.config['environment']['horizon']
        evaluation_eps_return = 0

        for eps_no in range(eval_episodes):
            state, d, eps_return, eps_len = env.reset(), False, 0, 0
            not_finished = False
            while not_finished:
                # get an action determinstically from policy
                action = policy(state, True)[0].detach().cpu().numpy() #TODO: why not using detach
                # apply the action to the environment
                state, reward, d, _ = env.step(action)
                eps_return += reward
                eps_len += 1
                not_finished = not(d or (eps_len == max_eps_len))
            
            evaluation_eps_return += eps_return
        average_evaluation_eps_return = evaluation_eps_return / eval_episodes
        return average_evaluation_eps_return
            

