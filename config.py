import warnings
import os.path
import sys
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def countuse(configlist):
    count = 0
    for config in configlist:
        if config['use'] == True:
            count += 1
    return count


class DefaultConfig(object):
    seed = 666
    device = 0

    scaler = StandardScaler()
    day_slot = 288
    n_route, n_his, n_pred = 228, 12, 12
    n_train, n_val, n_test = 34, 5, 5

    mode = 1
    # 1: 3, 6, 9, 12
    # 2: 3, 6, 12, 18, 24
    n_c = 10
    model = 'STAGNN_stamp'
    TPG = 'TPGNN'  
    name = r'TPGNN_r05p02kt3outer'
    if not os.path.exists("log"):
        os.makedirs("log")
    log_path = os.path.join(
        "log", name)
    if os.path.exists(log_path):
        crash = 1
        new_name = "_".join([name, str(crash)])
        log_path = os.path.join(
            "log", new_name)
        while os.path.exists(log_path):
            crash += 1
            new_name = "_".join([name, str(crash)])
            log_path = os.path.join(
                "log", new_name)
        name = new_name
    batch_size = 50
    lr = 1e-3

    a = 0.1
    r = 0.5
    n_mask = 1

    #optimizer
    adam = {'use': True, 'weight_decay': 1e-4}
    slr = {'use': True, 'step_size': 400, 'gamma': 0.3}

    resume = False
    start_epoch = 0
    epochs = 1500

    n_layer = 1
    n_attr, n_hid = 64, 512
    reg_A = 1e-4
    circle = 12*24
    drop_prob = 0.2

    # expand attr by conv
    CE = {'use': True, 'kernel_size': 1, 'bias': False}
    # expand attr by linear
    LE = {'use': False, 'bias': False}
    # spatio encoding
    SE = {'use': True, 'separate': True, 'no': False}
    # tempo encoding
    TE = {'use': True, 'no': True}

    # MultiHeadAttention
    attn = {'head': 1, 'd_k': 32, 'd_v': 32, 'drop_prob': drop_prob}

    # TPGNN polynomial order
    STstamp = {'use': True, 'kt': 3, 'temperature': 1.0}

    # TeaforN
    T4N = {'use': True, 'step': 2, 'end_epoch': 10000,
           'change_head': True, 'change_enc': True}
    stamp_path = "data/PeMS/time_stamp.npy"
    data_path = 'data/PeMS/V_228.csv'
    adj_matrix_path = 'data/PeMS/W_228.csv'
    dis_mat = None

    prefix = 'log/' + name + '/'
    checkpoint_temp_path = prefix + '/temp.pth'
    checkpoint_best_path = prefix + '/best.pth'
    tensorboard_path = prefix
    record_path = prefix + 'record.txt'

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    eps = 0.1

    def parse(self, kwargs):
        '''
        customize configuration by input in terminal
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has no attribute %s' % k)
            setattr(self, k, v)

    def output(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class Logger(object):
    def __init__(self, file_name='Default.log'):

        self.terminal = sys.stdout
        self.log = open(file_name, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
