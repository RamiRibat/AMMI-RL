# rollout_batch_size = 100000
# epoch_length = 1000
# model_train_freq = 250
# model_retain_epochs = 1
#
# rollout_length = 1
#
# rollouts_per_epoch = rollout_batch_size * epoch_length / model_train_freq
# print('rollouts_per_epoch: ', rollouts_per_epoch)
# model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
# print('model_steps_per_epoch: ', model_steps_per_epoch)
# new_pool_size = model_retain_epochs * model_steps_per_epoch
# print('new_pool_size: ', new_pool_size)



import numpy as np

# batch =  [(1,2,3), (4,5,6)]
#
# a, b, c = map(np.stack, zip(*batch))
#
# print('a', a)

obs = np.array([1, 4])

# obs_ = list(obs)

obs_stack = np.stack(obs)

print('obs stack: ', obs_stack)
