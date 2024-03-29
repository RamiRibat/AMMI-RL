

configurations = {

    'environment': {
            'name': 'Hopper-v2',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
        },

    'algorithm': {
        'name': 'MBNPG',
        'mode': 'PAL', # P: cons (few PG steps) | M: Aggr (many model updates + small real buffer)
        # 'mode': 'MAL', # P: Aggr (many PG steps) | M: Cons (few model updates + large real buffer)
        'model-based': True,
        'on-policy': True,
        'learning': {
            'epochs': 500, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 0, # Ni-- PAL: 5 | MAL: 10
            'expl_epochs': 0, # Nx-- PAL: 5 | MAL: 10

            'env_steps' : 1000, # E: interact E times then train
            'grad_WM_steps': 25, # G-- PAL: 25 | MAL: 10
            'grad_NPG_steps': 5, # ACG: ac grad, 40

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
        'num_ensembles': 4, # 7
        'num_elites': 4, # 5
        'sample_type': 'Random',
        'learn_reward': True,
        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_schedule': [20, 100, 1, 15],
        'network': {
            'arch': [512, 512],
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
        'action_noise': None,
        'clip_eps': 0.2,
        'kl_targ': 0.02,
        'normz_step_size': 0.01,
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
        'gamma': 0.995, # Discount factor - γ
        'lam': 0.95, # GAE - λ
        'network': {
            'arch': [128, 128],
            'activation': 'Tanh',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4,
            'max_grad_norm': 0.5,
        }
    },


    'data': {
        'buffer_type': 'simple',
        'optimize_memory_usage': False,
        'buffer_size': int(1e4), # PAL: small- 1e4 | MAL: large- 1e5
        'model_buffer_size': int(4e5),
        'real_ratio': 0.05,
        'model_val_ratio': 0.2,
        'rollout_trajectories': 800, # 4 Models x 200 Traj's
        'rollout_horizon': 500,
        'model_batch_size': 256,
        'batch_size': 256,
        'mini_batch_size': 64,
        'device': "auto",
    },


    'experiment': {
        'verbose': 0,
        'print_logs': True,
    }
}
