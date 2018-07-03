import os
import sys
from sklearn.datasets import fetch_mldata
import numpy as np

def digit_to_vec(x):
    res = np.zeros(10, dtype=np.float32)
    res[int(x)] = 1
    return res

def get_dataset_ori():
    mnist = fetch_mldata('MNIST original')
    return mnist.data, mnist.target

def get_dataset():
    X, y = get_dataset_ori()
    X = (X / 255.0).astype(np.float32)
    y = np.array([digit_to_vec(n) for n in y])
    return X, y

def save_bin_dataset(path):
    if os.path.isfile(path):
        return

    print('Generating MNIST binary file')
    X, y = get_dataset_ori()
    with open(path, 'wb') as f:
        for i in range(len(X)):
            f.write(bytes(X[i].tolist()))
            f.write(bytes([int(y[i])]))
    print('Saved to ' + path)



if __name__ == '__main__':
    save_bin_dataset(sys.argv[1])
