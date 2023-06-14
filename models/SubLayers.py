import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

def laplacian(W):
    N, N = W.shape
    W = W+torch.eye(N).to(W.device)
    D = W.sum(axis=1)
    D = torch.diag(D**(-0.5))
    out = D@W@D
    return out


def matrix_fnorm(W):
    # W:(h,n,n) return (h)
    h, n, n = W.shape
    W = W**2
    norm = (W.sum(dim=1).sum(dim=1))**(0.5)
    return norm/(n**0.5)


class TPGNN(nn.Module):
    # softlap+outerwave
    def __init__(self, d_attribute, d_out, n_route, n_his, dis_mat, kt=2, n_c=10, droprate=0., temperature=1.0) -> None:
        super(TPGNN, self).__init__()
        self.droprate = droprate
        print(n_route, n_c)
        self.r1 = nn.Parameter(torch.randn(n_route, n_c))
        # self.r2 = nn.Parameter(torch.randn(n_route, 10))
        self.w_stack = nn.Parameter(torch.randn(kt+1, d_attribute, d_out))
        nn.init.xavier_uniform_(self.w_stack.data)
        self.reduce_stamp = nn.Linear(n_his, 1, bias=False)
        self.temp_1 = nn.Linear(d_attribute//4, kt+1)
        # self.temp_2 = nn.Linear(d_attribute//4, kt+1)
        self.temperature = temperature
        self.d_out = d_out
        self.distant_mat = dis_mat

        self.kt = kt

    def forward(self, x, stamp):
        # x:(b,n,t,k) stamp:(b,t,k)
        residual = x
        b, n, t, k = x.size()
        h, _, _ = self.w_stack.shape
        w_stack = self.w_stack/(matrix_fnorm(self.w_stack).reshape(h, 1, 1))
        # print(stamp.shape)
        # (b,t,k)->(b,k,1)->(b,kt+1)
        period_emb = self.reduce_stamp(stamp.permute(0, 2, 1)).squeeze(2)
        temp_1 = self.temp_1(period_emb)
        # temp_2 = self.temp_2(period_emb)
        adj = self.distant_mat.clone()
        # adj_2 = self.cor_mat.clone()
        if self.training:
            nonzero_mask = self.distant_mat > 0
            adj[nonzero_mask] = F.dropout(adj[nonzero_mask], p=self.droprate)

        adj_1 = torch.softmax(torch.relu(
            laplacian(adj))/self.temperature, dim=0)
        adj_2 = torch.softmax(torch.relu(
            self.r1@self.r1.T)/self.temperature, dim=0)
        adj_1 = F.dropout(adj_1, p=self.droprate)
        adj_2 = F.dropout(adj_2, p=self.droprate)
        # (b,n,t,k)->(b,t,n,k)
        z = (x@w_stack[0])*(temp_1[:, 0].reshape(b, 1, 1, 1))
        z = z.permute(0, 2, 1, 3).reshape(b*t, n, -1)
        # for i in range(1, self.kt):
        #     z = adj@z + \
        #         (x@w_stack[i]*(temp[:, i].reshape(b, 1, 1, 1))
        #          ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        for i in range(1, self.kt):
            z = adj_1@z + \
                (x@w_stack[i]*(temp_1[:, i].reshape(b, 1, 1, 1))
                 ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        z_fix = (x@w_stack[0])*(temp_1[:, 0].reshape(b, 1, 1, 1))
        z_fix = z_fix.permute(0, 2, 1, 3).reshape(b*t, n, -1)
        for i in range(1, self.kt):
            z_fix = adj_2@z_fix + \
                (x@w_stack[i]*(temp_1[:, i].reshape(b, 1, 1, 1))
                 ).permute(0, 2, 1, 3).reshape(b*t, n, -1)
        z = z+z_fix

        z = z.reshape(b, t, n, self.d_out).permute(0, 2, 1, 3)

        return z
