

configurations = {

    'Comments': {
        'Rami': 'This was a nightmare!'
    },

    'environment': {
            'name': 'Hopper-v2',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
            'action_repeat': 1, # New (leave this)
        },

    'algorithm': {
        'name': 'MBPPO',
        'mode': 'PAL', # P: cons (few PG steps) | M: Aggr (many model updates + small real buffer)
        # 'mode': 'MAL', # P: Aggr (many PG steps) | M: Cons (few model updates + large real buffer)
        'model-based': True,
        'on-policy': True,
        'learning': {
            'epochs': 200, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 2, # Ni-- PAL: 5 | MAL: 10
            'expl_epochs': 2, # Nx-- PAL: 5 | MAL: 10

            'env_steps' : 1000, # E: interact E times then train
            'grad_WM_steps': 25, # G-- PAL: 25 | MAL: 10
            'grad_AC_steps': 10, # ACG: ac grad, 40
            'grad_PPO_steps': 50, # ACG: ac grad, 40

            'policy_update_interval': 1,
            'alpha_update_interval': 1,
            'target_update_interval': 1,

            'n_episodes_rollout': -1,

            'use_sde': False,
            'sde_sample_freq': -1,
            'use_sde_at_warmup': False,
                    },

        'evaluation': {
            'evaluate': True,
            'eval_deterministic': True,
            'eval_freq': 1, # Evaluate every 'eval_freq' epochs --> Ef
            'eval_episodes': 5, # Test policy for 'eval_episodes' times --> EE
            'eval_render_mode': None,
        }
    },


    'world_model': {
        'type': 'PE',
        'num_ensembles': 7,
        'num_elites': 5,
        # 'num_ensembles': 4,
        # 'num_elites': 4,
        'sample_type': 'Random',
        'learn_reward': True,
        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_schedule': [20, 100, 1, 15],
        'network': {
            # 'arch': [512, 512],
            'arch': [200, 200, 200, 200],
            'init_weights': 3e-3,
            'init_biases': 0,
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam", #@#
            'lr': 1e-3, #@#
            'wd': 1e-5,
            'dropout': None,
            'batch_size': 256,
            'device': "auto",
        }
    },

    'actor': {
        'type': 'ppopolicy',
        'constrained': False,
        'action_noise': None,
        'clip_eps': 0.25,
        'kl_targ': 0.02, # 0.03
        'max_dev': 0.15,
        'entropy_coef': 0.0,
        # 'normz_step_size': 0.01,
        'network': {
            'arch': [128, 128],
            # 'arch': [256, 256],
            # 'arch': [256, 128, 64],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            # 'lr': 1e-3,
            'lr': 3e-4,
            'max_grad_norm': 0.5,
        }
    },

    'critic': {
        'type': 'V',
        'number': 1,
        # 'gamma': 0.995, # Discount factor - γ
        # 'lam': 0.99, # GAE - λ
        'gamma': 0.99, # ReLU-12
        'gae_lam': 0.95, # ReLU-12
        'network': {
            'arch': [128, 128],
            # 'arch': [256, 256],
            # 'arch': [256, 128, 64],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            # 'lr': 1e-3,
            'lr': 3e-4,
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },


    'data': {
        'buffer_type': 'simple',
        'optimize_memory_usage': False,
        # 'init_obs_size': 50,
        'init_obs_size': 250,
        'buffer_size': int(1e4), # PAL: small- 1e4 | MAL: large- 1e5
        'ov_model_buffer_size': int(2e4),
        # 'ov_model_buffer_size': int(2e5),
        'device': "auto",
    },


    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }
}
