#!/usr/bin/env bash

data_path='data/PeMS/V_228.csv' #path to the MTS data
adj_path='data/PeMS/W_228.csv'  #path to the adjacency matrix, None if not exists
data_root='data/PeMS' #Directory to the MTS data

stamp_path="${data_root}/time_stamp.npy"
#training model
python main_stamp.py train --device=7 --n_route=228 --n_his=12 --n_pred=12 --n_train=34 --n_val=5 --n_test=5 --mode=1 --name='PeMS'\
    --data_path="data/PeMS/V_228.csv" --adj_matrix_path="data/PeMS/W_228.csv" --stamp_path=$stamp_path