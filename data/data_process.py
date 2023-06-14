from datetime import time
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def gen_stamp(data_path,data_root,cycle):
    data = pd.read_csv(data_path, header=None).values.astype(float)
    T, N = data.shape
    time_stamp = np.zeros(T)
    for idx in range(T):
        time_stamp[idx] = idx % cycle
    root = data_root
    name = "time_stamp.npy"
    np.save(os.path.join(root, name), time_stamp)

if __name__ == '__main__':
    import fire
    fire.Fire()