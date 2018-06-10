import numpy as np

s = np.array([0, 0, 0, 0], dtype=np.uint64)

def seed(x):
    x = np.uint64(x)
    s[0] = 1000*x
    s[1] = 2000*x
    s[2] = 3000*x
    s[3] = 4000*x
    

def next_u64():
    t = s[0] ^ (s[0] << np.uint64(11))
    s[0] = s[1]
    s[1] = s[2]
    s[2] = s[3]
    s[3] = s[3] ^ (s[3] >> np.uint64(19)) ^ t ^ (t >> np.uint64(8))
    return s[3]

def next_f32():
    x = np.float32(next_u64())
    div = np.float32(0xFFFFFFFFFFFFFFFF)
    return x / div;

def np_f32(shape):

    res = np.empty(shape).astype(np.float32)
    res2 = res.reshape(res.size)
    for i in range(res2.size):
        res2[i] = next_f32()
    return res
    

import tensors_saver


if __name__ == '__main__':

    seed(234)
    
    tensors_saver.set_out_path('./out.npz')
    tensors_saver.add(np_f32((145, 18, 12, 34)))
    
