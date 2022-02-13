

configurations = {

    'environment': {
            'name': 'Hopper-v2',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
        },

    'algorithm': {
        'name': 'MBPO',
        'learning': {
            'epochs': 125, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 5, # Ni epochs = 5000 exploration steps
            'expl_epochs': 10, # Nx epochs
            'real_epochs': 0, # Nr epochs

            'env_steps' : 1, # E: interact E times then train
            'grad_WM_steps': 50, # G: ac grad
            'grad_SAC_steps': 20, # ACG: ac grad, 40

            'policy_update_interval': 1,
            'alpha_update_interval': 1,
            'target_update_interval': 1,


            'n_episodes_rollout': -1,
            # 'net_arch': [64, dict(vf=[256, 256], pi=[128])], # [shared, dict(non-shared)]

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
        'sample_type': 'Random',
        'learn_reward': True,
        'model_train_freq': 250, # Mf
        'rollout_schedule': [20, 100, 1, 15],
        'network': {
            'net_arch': [200,200,200,200], #@#
            'init_weights': 3e-3,
            'init_biases': 0,
            'activation': 'LeakyReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam", #@#
            'lr': 1e-3, #@#
            'wd': 1e-5,
            'wd': 1e-5,
            'batch_size': 250,
            'device': "auto",
        }
    },

    'control': {
        'type': 'gaussianpolicy',
        'action_noise': None, # Optional
        'alpha': 0.2, # Temprature/Entropy #@#
        'automatic_entropy': False, # trainer_kwargs
        'target_entropy': "auto",
        'network': {
            'net_arch': [256,256], #@#
            'init_weights': 3e-3,
            'init_biases': 0,
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam", #@#
            'lr': 3e-4, #@#
            'wd': 1e-5,
            'wd': 1e-5,
            'batch_size': 256,
            'device': "auto",
        }
    },

    'value-functions': {
        'type': 'sofQ',
        'number': 2,
        'gamma': 0.99,
        'tau': 5e-3,
        'network': {
            'net_arch': [256,256], #@#
            'init_weights': 3e-3,
            'init_biases': 0,
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam", #@#
            'lr': 3e-4, #@#
            'wd': 1e-5,
            'batch_size': 256,
            'device': "auto",
        }
    },


    'data': {
        'buffer_type': 'simple',
        'optimize_memory_usage': False,
        'buffer_size': int(5e5),
        'model_buffer_size': int(1e7),
        'real_ratio': 0.05,
        'model_val_ration': 0.2,
        'rollout_batch_size': 400,
        'model_batch_size': 256,
        'batch_size': 256,
        'device': "auto",
    },


    'experiment': {
        'verbose': 0,
        'device': "cpu",
        # 'device': "cuda:0",
        # 'WandB': True,
        'WandB': False,
        'print_logs': True,
    }
}
