import numpy as np

batch_size = 20

mini_batch_size = 5

b_inx = np.arange(batch_size)
print('b_inx: ', b_inx)

np.random.shuffle(b_inx)
print('b_inx: ', b_inx)

for start in range(0, batch_size, mini_batch_size):
    end = start + mini_batch_size
    mb_inx = b_inx[start:end]
    print('mb_inx: ', mb_inx)

    
