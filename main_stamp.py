import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import models
import sys
import os
import os.path
import time
from models import predict, predict_stamp
from data import STAGNN_Dataset, STAGNN_stamp_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import evaluate_metric, weight_matrix, weight_matrix_nl, laplacian, vendermonde
from config import DefaultConfig, Logger


opt = DefaultConfig()

sys.stdout = Logger(opt.record_path)

# random seed
seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def test(model, loss_fn, test_iter, opt):
    model.eval()
    loss_sum, n = 0.0, 0
    for x, stamp, y in test_iter:
        x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
        x = x.type(torch.cuda.FloatTensor)
        stamp = stamp.type(torch.cuda.LongTensor)
        y = y.type(torch.cuda.FloatTensor)

        y_pred = predict_stamp(model, x, stamp, y, opt)

        loss = loss_fn(y_pred, y)
        loss_sum += loss.item()
        n += 1
    return loss_sum / n


def train(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.device)
    # adj matrix
    if opt.adj_matrix_path != None:
        opt.dis_mat = weight_matrix_nl(opt.adj_matrix_path, epsilon=opt.eps)
        opt.dis_mat = torch.from_numpy(opt.dis_mat).float().cuda()
    else:
        opt.dis_mat = 0.0

    # path
    opt.prefix = 'log/' + opt.name + '/'
    if not os.path.exists(opt.prefix):
        os.makedirs(opt.prefix)
    opt.checkpoint_temp_path = opt.prefix + '/temp.pth'
    opt.checkpoint_best_path = opt.prefix + '/best.pth'
    opt.tensorboard_path = opt.prefix
    opt.record_path = opt.prefix + 'record.txt'

    opt.output()

    # load data
    batch_size = opt.batch_size
    train_dataset = STAGNN_stamp_Dataset(opt, train=True, val=False)
    val_dataset = STAGNN_stamp_Dataset(opt, train=False, val=True)
    test_dataset = STAGNN_stamp_Dataset(opt, train=False, val=False)
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size)

    # mask
    n_route = opt.n_route
    n_his = opt.n_his
    n_pred = opt.n_pred
    enc_spa_mask = torch.ones(1, 1, n_route, n_route).cuda()
    enc_tem_mask = torch.ones(1, 1, n_his, n_his).cuda()
    dec_slf_mask = torch.tril(torch.ones(
        (1, 1, n_pred + 1, n_pred + 1)), diagonal=0).cuda()
    dec_mul_mask = torch.ones(1, 1, n_pred + 1, n_his).cuda()

    # loss
    loss_fn = nn.L1Loss()

    MAEs, MAPEs, RMSEs = [], [], []
    for i in range(1):
        # model
        model = getattr(models, opt.model)(
            opt,
            enc_spa_mask, enc_tem_mask,
            dec_slf_mask, dec_mul_mask
        )
        model.cuda()

        # optimizer
        lr = opt.lr
        if opt.adam['use']:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=opt.adam['weight_decay'])

        # scheduler
        if opt.slr['use']:
            step_size, gamma = opt.slr['step_size'], opt.slr['gamma']
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
        elif opt.mslr['use']:
            milestones, gamma = opt.mslr['milestones'], opt.mslr['gamma']
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones, gamma=gamma)

        # resume
        start_epoch = opt.start_epoch
        min_val_loss = np.inf
        checkpoint_temp_path = opt.checkpoint_temp_path
        if opt.resume:
            if os.path.isfile(checkpoint_temp_path):
                checkpoint = torch.load(checkpoint_temp_path)
                start_epoch = checkpoint['epoch'] + 1
                min_val_loss = checkpoint['min_loss']
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint (epoch {})'.format(
                    checkpoint['epoch']))

        # tensorboard
        tensorboard_path = opt.tensorboard_path + str(start_epoch)
        writer = SummaryWriter(tensorboard_path)

        # train
        name = opt.name
        epochs = opt.epochs
        checkpoint = None
        checkpoint_temp_path = opt.checkpoint_temp_path
        start_time = time.perf_counter()
        best_perf = 0
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            loss_sum, n = 0.0, 0
            for x, stamp, y in train_iter:
                x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
                x = x.type(torch.cuda.FloatTensor)
                stamp = stamp.type(torch.cuda.LongTensor)
                y = y.type(torch.cuda.FloatTensor)

                x = x.repeat(2, 1, 1, 1)
                stamp = stamp.repeat(2, 1)
                y = y.repeat(2, 1, 1, 1)
                y_pred, loss = model(x, stamp, y, epoch)
                bs = y.shape[0]
                y_pred1 = y_pred[:bs//2, :, :, :]
                y_pred2 = y_pred[bs//2:, :, :, :]
                r_loss = F.l1_loss(y_pred1, y_pred2)
                r_loss = r_loss * opt.r
                loss = loss + r_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                n += 1
            scheduler.step()

            model.eval()

            val_loss = test(model, loss_fn, val_iter, opt)
            print('epoch', epoch, ' ', name, ', train loss:',
                  loss_sum / n, ', validation loss:', val_loss)
            if epoch>200 and val_loss < min_val_loss**0.999:
                if val_loss<min_val_loss:
                    min_val_loss = val_loss
                print(
                    torch.abs(model.encoder.layer_stack[0].stgc.r1.data).sum())
                checkpoint = {
                    'epoch': epoch,
                    'min_loss': min_val_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, checkpoint_temp_path)

                MAE, MAPE, RMSE = evaluate_metric(model, test_iter, opt)
                print("MAE:", MAE, ", MAPE:", MAPE, "%, RMSE:", RMSE)
                best_perf = MAE,MAPE,RMSE
            writer.add_scalar('train loss', loss_sum / n, epoch)
            writer.add_scalar('test loss', val_loss, epoch)

        test_loss = "NIL"
        if opt.mode == 1:
            MAE, MAPE, RMSE = best_perf
            MAEs.append(MAE)
            MAPEs.append(MAPE)
            RMSEs.append(RMSE)
            print("test loss:", test_loss, "\nMAE:",
                  MAE, ", MAPE:", MAPE, "%, RMSE:", RMSE)
        elif opt.mode == 2:
            RAE, RSE, COR = best_perf
            print("test loss:", test_loss, "\nRAE:",
                  RAE, ", RSE:", RSE, "%, RMSE:", COR)
        print('='*20)
    end_time = time.perf_counter()
    total_time = end_time-start_time
    print("training elapsedd with {:.2f} seconds for {} iterations, the sec/iter = {:.2f}".format(total_time, opt.epochs, total_time/opt.epochs))
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    RMSEs = np.array(RMSEs)

    MAE_mean, MAE_std = np.mean(MAEs, axis=0), np.std(MAEs, axis=0, ddof=1)
    MAPE_mean, MAPE_std = np.mean(MAPEs, axis=0), np.std(MAPEs, axis=0, ddof=1)
    RMSE_mean, RMSE_std = np.mean(RMSEs, axis=0), np.std(RMSEs, axis=0, ddof=1)
    np.savez(opt.prefix + '/result.npz', MAE=MAEs, MAPE=MAPEs, RMSE=RMSEs)
    print("\nMAE_mean:", MAE_mean, ", MAPE_mean:",
          MAPE_mean, ", RMSE_mean:", RMSE_mean)
    print("\nMAE_std:", MAE_std, ", MAPE_std:",
          MAPE_std, ", RMSE_std:", RMSE_std)


if __name__ == '__main__':
    import fire
    fire.Fire()
