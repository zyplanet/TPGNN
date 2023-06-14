import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from models import predict, predict_stamp


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs() > 1e-6)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs() > 1e-6)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels.abs() > 1e-6)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss


def evaluate_metric(model, data_iter, opt):
    if opt.mode == 1:
        model.eval()
        scaler = opt.scaler
        n_pred = opt.n_pred

        length = n_pred // 3
        with torch.no_grad():
            mae = [[] for _ in range(length)]
            mape = [[] for _ in range(length)]
            mse = [[] for _ in range(length)]
            MAE, MAPE, RMSE = [0.0] * length, [0.0] * length, [0.0] * length

            if 'pretext' in opt.model:
                for x, y, z in data_iter:
                    x, y = x.cuda(), y.cuda()
                    x = x.type(torch.cuda.FloatTensor)
                    y = y.type(torch.cuda.FloatTensor)
                    y_pred = predict(model, x, y, opt).permute(
                        0, 3, 2, 1).reshape(-1, 228)

                    y_pred = scaler.inverse_transform(
                        y_pred.cpu().numpy()).reshape(-1, 1, 12, 228)
                    y = scaler.inverse_transform(
                        y.permute(0, 3, 2, 1).reshape(-1, 228).cpu().numpy()).reshape(-1, 1, 12, 228)

                    for i in range(length):
                        y_pred_select = y_pred[:, :, 3 * i + 2, :].reshape(-1)
                        y_select = y[:, :, 3 * i + 2, :].reshape(-1)
                        d = np.abs(y_select - y_pred_select)

                        y_pred_select = torch.from_numpy(y_pred_select)
                        y_select = torch.from_numpy(y_select)
                        mae[i] += masked_mae(y_pred_select,
                                             y_select, 0.0).numpy().tolist()
                        mape[i] += masked_mape(y_pred_select,
                                               y_select, 0.0).numpy().tolist()
                        mse[i] += masked_mse(y_pred_select,
                                             y_select, 0.0).numpy().tolist()

                for j in range(length):
                    MAE[j] = np.array(mae[j]).mean()
                    MAPE[j] = np.array(mape[j]).mean()
                    RMSE[j] = np.sqrt(np.array(mse[j]).mean())

                return MAE, MAPE, RMSE
            elif 'stamp' in opt.model:
                for x, stamp, y in data_iter:
                    x, stamp, y = x.cuda(), stamp.cuda(), y.cuda()
                    x = x.type(torch.cuda.FloatTensor)
                    stamp = stamp.type(torch.cuda.LongTensor)
                    y = y.type(torch.cuda.FloatTensor)
                    y_pred = predict_stamp(model, x, stamp, y, opt).permute(
                        0, 3, 2, 1).reshape(-1, 228)

                    y_pred = scaler.inverse_transform(
                        y_pred.cpu().numpy()).reshape(-1, 1, 12, 228)
                    y = scaler.inverse_transform(
                        y.permute(0, 3, 2, 1).reshape(-1, 228).cpu().numpy()).reshape(-1, 1, 12, 228)

                    for i in range(length):
                        y_pred_select = y_pred[:, :, 3 * i + 2, :].reshape(-1)
                        y_select = y[:, :, 3 * i + 2, :].reshape(-1)
                        d = np.abs(y_select - y_pred_select)

                        y_pred_select = torch.from_numpy(y_pred_select)
                        y_select = torch.from_numpy(y_select)
                        mae[i] += masked_mae(y_pred_select,
                                             y_select, 0.0).numpy().tolist()
                        mape[i] += masked_mape(y_pred_select,
                                               y_select, 0.0).numpy().tolist()
                        mse[i] += masked_mse(y_pred_select,
                                             y_select, 0.0).numpy().tolist()

                for j in range(length):
                    MAE[j] = np.array(mae[j]).mean()
                    MAPE[j] = np.array(mape[j]).mean()
                    RMSE[j] = np.sqrt(np.array(mse[j]).mean())

                return MAE, MAPE, RMSE

            for x, y in data_iter:
                x, y = x.cuda(), y.cuda()
                x = x.type(torch.cuda.FloatTensor)
                y = y.type(torch.cuda.FloatTensor)
                y_pred = predict(model, x, y, opt).permute(
                    0, 3, 2, 1).reshape(-1, 228)

                y_pred = scaler.inverse_transform(
                    y_pred.cpu().numpy()).reshape(-1, 1, 12, 228)
                y = scaler.inverse_transform(
                    y.permute(0, 3, 2, 1).reshape(-1, 228).cpu().numpy()).reshape(-1, 1, 12, 228)

                for i in range(length):
                    y_pred_select = y_pred[:, :, 3 * i + 2, :].reshape(-1)
                    y_select = y[:, :, 3 * i + 2, :].reshape(-1)
                    d = np.abs(y_select - y_pred_select)

                    y_pred_select = torch.from_numpy(y_pred_select)
                    y_select = torch.from_numpy(y_select)
                    mae[i] += masked_mae(y_pred_select,
                                         y_select, 0.0).numpy().tolist()
                    mape[i] += masked_mape(y_pred_select,
                                           y_select, 0.0).numpy().tolist()
                    mse[i] += masked_mse(y_pred_select,
                                         y_select, 0.0).numpy().tolist()

            for j in range(length):
                MAE[j] = np.array(mae[j]).mean()
                MAPE[j] = np.array(mape[j]).mean()
                RMSE[j] = np.sqrt(np.array(mse[j]).mean())

            return MAE, MAPE, RMSE

    elif opt.mode == 2:
        model.eval()
        scaler = opt.scaler

        evaluateL1 = nn.L1Loss(size_average=False)
        evaluateL2 = nn.MSELoss(size_average=False)
        RAE = []
        RSE = []
        COR = []
        with torch.no_grad():
            for i in range(5):
                output_empty = True
                output = None
                label_empty = True
                label = None
                n_samples = 0

                l1loss = 0.0
                l2loss = 0.0

                for x, y in data_iter:
                    y_pred = predict(model, x, y, opt).permute(0, 3, 2, 1)
                    y_pred = scaler.inverse_transform(y_pred.cpu().numpy())
                    y = scaler.inverse_transform(
                        y.permute(0, 3, 2, 1).cpu().numpy())

                    y = y[:, :, i].squeeze(1)
                    y_pred = y_pred[:, :, i].squeeze(1)

                    y = torch.from_numpy(y)
                    y_pred = torch.from_numpy(y_pred)
                    if output_empty:
                        output = y_pred
                        output_empty = False
                    else:
                        output = torch.cat((output, y_pred), dim=0)

                    if label_empty:
                        label = y
                        label_empty = False
                    else:
                        label = torch.cat((label, y), dim=0)

                    l2loss += evaluateL2(y_pred, y).item()
                    l1loss += evaluateL1(y_pred, y).item()
                    n_samples += (y_pred.shape[0] * opt.n_route)

                rae = torch.mean(torch.abs(label - torch.mean(label)))
                rse = label.std() * np.sqrt((len(label) - 1.0)/len(label))

                output = output.data.numpy()
                label = label.data.numpy()

                sigma_p = (output).std(axis=0)
                sigma_g = (label).std(axis=0)
                mean_p = output.mean(axis=0)
                mean_g = label.mean(axis=0)
                idx = (sigma_g != 0)
                COR_tmp = ((output - mean_p) * (label - mean_g)
                           ).mean(axis=0) / (sigma_p * sigma_g)
                COR_tmp = (COR_tmp[idx]).mean()

                RSE_tmp = (math.sqrt(l2loss / n_samples) / rse).item()
                RAE_tmp = ((l1loss / n_samples) / rae).item()

                if i != 3:
                    RAE.append(RAE_tmp)
                    RSE.append(RSE_tmp)
                    COR.append(COR_tmp)
            return RAE, RSE, COR


def weight_matrix(file_path, sigma2=0.1, epsilon=0.1, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
        return laplacian(W)
        # return W
    else:
        return W


def weight_matrix_nl(file_path, sigma2=0.1, epsilon=0.1, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
        # return laplacian(W)
        # print((W>0).sum()/(W.shape[0])**2)
        return W
    else:
        return W
def vendermonde(S,degree):
    m = S.shape[0]
    V = np.zeros((m,degree+1))
    for k in range(degree+1):
        if k == 0:
            V[:,k] = 1.
        else:
            V[:,k] = S**(k)
    return V

def laplacian(W):
    N, N = W.shape
    W = W+np.eye(N)
    D = W.sum(axis=1)
    D = np.diag(D**(-0.5))
    out = D@W@D
    return out

def polyA(A,n):
    out = A
    for i in range(n-1):
        out = A@out
    return out

    
    
    
    