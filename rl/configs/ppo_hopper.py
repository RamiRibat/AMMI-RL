

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
            'epochs': 500, # N epochs
            'epoch_steps': 4000, # NT steps/epoch
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
        'target_entropy': 'auto',
        'network': {
            'arch': [256,256],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'critic': {
        'type': 'V',
        'number': 1,
        'gamma': 0.99,
        'tau': 5e-3,
        'network': {
            'arch': [256,256],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(4e3),
        'batch_size': 256
    },

    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }

}
