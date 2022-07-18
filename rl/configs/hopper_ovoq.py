

configurations = {

    'Comments': {
        'Rami': 'This was a real nightmare!'
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
        'name': 'OVOQ',
        'full-name': 'Model-based Oo-policy Value, Off-policy Quality actor-critic',
        # 'mode': 'PAL', # P: cons (few PG steps) | M: Aggr (many model updates + small real buffer)
        # 'mode': 'MAL', # P: Aggr (many PG steps) | M: Cons (few model updates + large real buffer)
        'model-based': False,
        'on-policy': True, # TrajBuffer for env
        'learning': {
            'epochs': 2000, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'ov_init_epochs': 0, # Random Actions + No Learning
            'oq_init_epochs': 1, # Random Actions + No Learning
            'expl_epochs': 10, # Random Actions + Learning

            'env_steps' : 1,
            'ov_N_freq': 10,
            'ov_E_freq': 1000,
            'grad_OV_steps': 5,
            'grad_PPO_steps': 100,
            'grad_OQ_SAC_steps': 1,

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

    'actor': {
        'type': 'ovoqpolicy',
        'constrained': False,
        'action_noise': None,
        'alpha': 0.2,
        'automatic_entropy': False,
        'target_entropy': 'auto',
        'clip_eps': 0.25,
        'kl_targ': 0.02,
        'max_dev': 0.1,
        'entropy_coef': 0.0,
        'network': {
            'log_std_grad': False,
            'init_log_std_v': 1,
            # 'arch': [64, 64],
            # 'arch': [128, 128],
            'arch': [256, 256],
            # 'arch': [512, 512],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            # 'lr': 1e-3,
            'lr': 3e-4,
            'output_activation': 'nn.Identity',
            'initialize_weights': True,
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },

    'critic-v': {
        'type': 'V',
        'number': 1,
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
            'output_activation': 'nn.Identity',
            'initialize_weights': True,
            'optimizer': "Adam",
            'max_grad_norm': 0.5,
        }
    },

    'critic-q': {
        'type': 'sofQ',
        'number': 2,
        'gamma': 0.99,
        'tau': 5e-3,
        'network': {
            # 'arch': [128, 128],
            # 'arch': [256, 128],
            'arch': [256, 256],
            # 'activation': 'Tanh',
            'activation': 'PReLU',
            'output_activation': 'nn.Identity',
            'initialize_weights': True,
            'optimizer': "Adam",
            # 'lr': 1e-3,
            'lr': 3e-4,
        }
    },


    'data': {
        'buffer_type': 'simple',
        'optimize_memory_usage': False,

        'oq_buffer_size': int(1e6), # Total learning
        # 'recent_buffer_size': int(1e4), # Agressive OV model
        'ov_buffer_size': int(1e4),
        # 'oq_model_buffer_size': int(1e7),
        # 'oq_real_ratio': 0.05,
        # 'oq_model_val_ratio': 0.2,
        # 'oq_rollout_batch_size': int(1e5),
        # 'oq_model_batch_size': 256,
        'oq_batch_size': 256,

        # 'device': "auto",
    },


    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }
}
