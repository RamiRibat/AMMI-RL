

configurations = {
    'environment': {
            'name': 'Hopper-v2',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
        },

    'algorithm': {
        'name': 'PPO',
        'model-based': False,
        'on-policy': True,
        'learning': {
            'epochs': 200, # N epochs
            'epoch_steps': 10000, # NT steps/epoch
            'init_epochs': 0, # Ni epochs
            'expl_epochs': 0, # Nx epochs

            'env_steps' : 10000, # E: interact E times then train
            'train_AC_freq': 1, # F: frequency of AC training
            'grad_AC_steps': 100, # ACG: ac grad

            'policy_update_interval': 1,
                    },

        'evaluation': {
            'evaluate': True,
            'eval_deterministic': True,
            'eval_freq': 1, # Evaluate every 'eval_freq' epochs --> Ef
            'eval_episodes': 5, # Test policy for 'eval_episodes' times --> EE
            'eval_render_mode': None,
        }
    },

    'actor': { # No init
        'type': 'Gaussian',
        # 'type': 'TanhSquashedGaussian',
        'constrained': False,
        'action_noise': None,
        'clip_eps': 0.25,
        'kl_targ': 0.02,
        'max_dev': 0.5,
        'entropy_coef': 0.0,
        'network': {
            # 'std_grad': True,
            'log_std_grad': False,
            # 'init_std': 3.,
            'init_log_std': 1.,
            # 'min_std': 1e-6,
            # 'log_std_grad': False,
            # 'init_log_std': 1,
            # 'arch': [64, 64],
            # 'arch': [128, 128],
            'arch': [256, 256],
            # 'arch': [512, 512],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            # 'lr': 1e-3,
            'lr': 3e-4,
            'op_activation': 'Identity',
            'initialize_weights': True,
            'optimizer': "Adam",
            'max_grad_norm': 0.25,
        }
    },

    'critic': { # Init
        'type': 'V',
        'number': 1,
        # 'gamma': 0.995, # Stable performance
        # 'gae_lam': 0.99, # Stable performance
        'gamma': 0.99,
        'gae_lam': 0.95,
        'network': {
            # 'arch': [64, 64],
            'arch': [128, 128],
            # 'arch': [256, 256],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            'lr': 1e-3,
            # 'lr': 3e-4,
            'op_activation': 'Identity',
            'initialize_weights': True,
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(1e4),
        'batch_size': int(1e4),
    },

    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }

}
