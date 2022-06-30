

configurations = {
    'environment': {
            'name': 'MountainCarContinuous-v0',
            'type': 'classic-control',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 100,
        },

    'algorithm': {
        'name': 'SAC',
        'learning': {
            'epochs': 100, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 2, # Ni epochs
            'expl_epochs': 5, # Nx epochs

            'env_steps' : 50, # E: interact E times then train
            'grad_AC_steps': 50, # ACG: ac grad
            
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
        'alpha': 0.02, # Temprature/Entropy #@#
        'automatic_entropy': False,
        'target_entropy': 'auto',
        'network': {
            'arch': [64,64],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'critic': {
        'type': 'sofQ',
        'number': 2,
        'gamma': 0.999,
        'tau': 5e-3,
        'network': {
            'arch': [64,64],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(1e6),
        'batch_size': 100
    },

    'experiment': {
        'name': 'seed1',
        'seed': 1,
        'verbose': 0,
        # 'device': "cpu",
        'device': "cuda:0",
        'WandB': False,
        'print_logs': True,
        'logdir': 'tmp/sac',
        'capture_video': False,
        'video_dir': './video'
    }
    
}
