

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
            'epochs': 250, # N epochs
            # 'epoch_steps': 2048, # NT steps/epoch
            'epoch_steps': 4000, # NT steps/epoch
            'init_epochs': 0, # Ni epochs
            'expl_epochs': 0, # Nx epochs

            # 'env_steps' : 2048, # E: interact E times then train
            'env_steps' : 4000, # E: interact E times then train
            'train_AC_freq': 1, # F: frequency of AC training
            'grad_AC_steps': 80, # ACG: ac grad

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
        'action_noise': None,
        'clip_eps': 0.2,
        'kl_targ': 0.01,
        'entropy_coef': 0.0,
        'network': {
            'arch': [64, 64],
            'activation': 'Tanh',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4,
            'max_grad_norm': 0.5,
        }
    },

    'critic': {
        'type': 'V',
        'number': 1,
        'gamma': 0.99,
        'lam': 0.97,
        'network': {
            'arch': [64, 64],
            'activation': 'Tanh',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 1e-3,
            'max_grad_norm': 0.5,
        }
    },

    'data': {
        'buffer_type': 'simple',
        # 'buffer_size': int(2048),
        # 'batch_size': 2048,
        # 'n_mini_batches': 32,
        # 'mini_batch_size': 64,
        'buffer_size': int(4000),
        'batch_size': 4000,
        'n_mini_batches': 62,
        'mini_batch_size': 64,
    },

    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }

}
