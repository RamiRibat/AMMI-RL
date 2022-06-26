

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

    'actor': {
        'type': 'ppopolicy',
        'constrained': False,
        'action_noise': None,
        'clip_eps': 0.2,
        'kl_targ': 0.02, # 0.03
        'max_dev': 0.15,
        'entropy_coef': 0.0,
        'network': {
            # 'arch': [64, 64],
            # 'arch': [128, 64],
            # 'arch': [128, 128],
            # 'activation': 'Tanh',
            # 'lr': 1e-3,
            'arch': [256, 128, 64],
            'activation': 'ReLU',
            'lr': 3e-4,
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },

    'critic': {
        'type': 'V',
        'number': 1,
        'gamma': 0.995,
        'gae_lam': 0.99,
        # 'gamma': 0.99,
        # 'gae_lam': 0.95,
        'network': {
            # 'arch': [64, 64],
            # 'arch': [128, 64],
            # 'arch': [128, 128],
            # 'activation': 'Tanh',
            # 'lr': 1e-3,
            'arch': [256, 128, 64],
            'activation': 'ReLU',
            'lr': 3e-4,
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(10000),
        'batch_size': 10000,
    },

    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }

}
