

configurations = {
    'environment': {
            'name': 'Hopper-v2',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
        },

    'algorithm': {
        'name': 'SAC',
        'model-based': False,
        'on-policy': False,
        'learning': {
            'epochs': 2000, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 1, # Ni epochs
            'expl_epochs': 10, # Nx epochs

            'env_steps' : 1, # E: interact E times then train
            'grad_AC_steps': 1, # ACG: ac grad

            'policy_update_interval': 1,
            'alpha_update_interval': 1,
            'target_update_interval': 1,
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
        'type': 'gaussianpolicy',
        'action_noise': None,
        'alpha': 0.2, # Temprature/Entropy #@#
        'automatic_entropy': False,
        'target_entropy': 'auto',
        'network': {
            'arch': [128, 128],
            # 'arch': [256, 256],
            # 'arch': [256, 128, 64],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            # 'n_parameters': 2,
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4,
        }
    },

    'critic': {
        'type': 'sofQ',
        'number': 2,
        'gamma': 0.99,
        # 'gamma': 0.995,
        'tau': 5e-3,
        'network': {
            # 'arch': [128, 128],
            # 'arch': [256, 128],
            'arch': [256, 256],
            # 'arch': [256, 128, 64],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            # 'n_parameters': 1,
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            # 'lr': 1e-3, # Conv at Ep:?
            'lr': 3e-4, # Conv at Ep:340 | ReLU-16
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(1e6),
        # 'batch_size': 128,
        'batch_size': 256,
        # 'batch_size': 512
    },

    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }

}
