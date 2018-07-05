import os
import sys

import numpy as np
import tensorflow as tf
import tensors_saver

WEIGHTS_PATH = sys.argv[1]
tensors_saver.set_out_path(sys.argv[2])

BATCH = 64

def padd_ker(x0):

    res = np.zeros((63, 63, 64, 64), dtype=np.float32)
    for hIndex in range(0, 63, 2):
        for wIndex in range(0, 63, 2):
            for inCh in range(64):
                for oCh in range(64):
                    prevHIndex = (int)(hIndex - ((int)(hIndex / 2)))
                    prevWIndex = (int)(wIndex - ((int)(wIndex / 2)))
                    res[hIndex, wIndex, inCh, oCh] = x0[inCh, prevHIndex, prevWIndex, oCh]

    return res

def im2col(y0):
    res = np.zeros((64*63*63, 5*5*3), dtype=np.float32)
    for b in range(3):
        for ch in range(64):
            for stepH in range(5):
                for stepW in range(5):
                    for winH in range(63):
                        for winW in range(63):
                            hIndex = stepH + winH - 1
                            wIndex = stepW + winW - 1
                            if (hIndex >= 0 and wIndex >= 0 and hIndex < 64 and wIndex < 64):
                                res[ch * 63*63 + winH * 63 + winW, b * (5*5) + stepH * 5 + stepW] = y0[ch, hIndex, wIndex, b]
    return res

def kercol(x0):
    res = np.zeros((64, 64*63*63), dtype=np.float32)
    for hIndex in range(63):
        for wIndex in range(63):
            for inCh in range(64):
                for oCh in range(64):
                    res[oCh, inCh*63*63+hIndex*63+wIndex] = x0[hIndex, wIndex, inCh, oCh]
    return res

x = np.random.randn(BATCH, 32, 32, 64).astype(np.float32)
dx = np.random.randn(64, 64, 64, 3).astype(np.float32)

y = padd_ker(x)
dy = im2col(dx)
yker = kercol(y)
conv = yker @ dy

print(conv.shape)
print("Rand val[481] = {}".format(conv[6, 31]));

tensors_saver.add(y)
tensors_saver.add(dy)
tensors_saver.add(yker)
tensors_saver.add(conv)
tensors_saver.save()

data = tensors_saver.Saver(WEIGHTS_PATH)
data.add(x)
data.add(dx)
data.save()
