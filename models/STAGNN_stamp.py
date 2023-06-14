import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Layers import ConvExpandAttr, SpatioEnc, TempoEnc, MLP, EncoderLayer_stamp, DecoderLayer

import random


class SrcProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        n_his = opt.n_his
        n_route, n_attr = opt.n_route, opt.n_attr

        # n_attr = 33

        self.CE = opt.CE['use']
        if self.CE:
            self.enc_exp = ConvExpandAttr(
                1, n_attr, opt.CE['kernel_size'], opt.CE['bias'])

        self.LE = opt.LE['use']
        if self.LE:
            self.enc_exp = nn.Linear(1, n_attr, bias=opt.LE['bias'])

        self.SE = opt.SE['use']
        if self.SE:
            self.enc_spa_enco = SpatioEnc(n_route, n_attr, opt.SE['no'])

        self.TE = opt.TE['use']
        if self.TE:
            self.enc_tem_enco = TempoEnc(n_his, n_attr, opt.TE['no'])
        # self.time_emb = nn.Embedding(opt.circle, opt.n_attr//4)
        # self.emb_conv = nn.Linear(opt.n_attr//4, opt.n_attr, bias=False)
        self.distant_mat = opt.dis_mat
        # self.re = nn.Linear(64, n_attr)

    def forward(self, src, stamp):
        src = self.enc_exp(src)
        b, n, t, k = src.shape
        if self.SE:
            src = self.enc_spa_enco(src)
        if self.TE:
            # src = src+self.emb_conv(self.time_emb(stamp)).reshape(b, 1, t, k)
            src = self.enc_tem_enco(src)

        return src


class TrgProcess(nn.Module):
    def __init__(self, opt):
        super().__init__()
        n_his, n_pred = opt.n_his, opt.n_pred
        n_route, n_attr = opt.n_route, opt.n_attr

        self.mlp = MLP(n_his, 1)

        self.CE = opt.CE['use']
        if self.CE:
            self.dec_exp = ConvExpandAttr(
                1, n_attr, opt.CE['kernel_size'], opt.CE['bias'])

        # spatio encoding
        self.SE = opt.SE['use']
        if self.SE:
            self.dec_spa_enco = SpatioEnc(n_route, n_attr, opt.SE['no'])

        # temporal encoding
        self.TE = opt.TE['use']
        if self.TE:
            self.dec_tem_enco = TempoEnc(
                n_pred + opt.T4N['step'], n_attr, opt.TE['no'])


    def forward(self, trg, enc_output, head=None):
        head = self.mlp(enc_output)
        trg = self.dec_exp(trg)
        if self.SE:
            trg = self.dec_spa_enco(trg)
        trg = torch.cat([head, trg], dim=2)
        if self.TE:
            trg = self.dec_tem_enco(trg)

        return trg


class Decoder(nn.Module):
    def __init__(
        self,
        opt,
        dec_slf_mask, dec_mul_mask
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(opt, dec_slf_mask, dec_mul_mask)
            for _ in range(opt.n_layer)
        ])

    def forward(self, x, enc_output):
        for layer in self.layer_stack:
            x = layer(x, enc_output)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        opt,
        enc_spa_mask, enc_tem_mask
    ):
        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer_stamp(opt, enc_spa_mask, enc_tem_mask)
            for _ in range(1)
        ])

    def forward(self, x, time_stamp):
        for layer in self.layer_stack:
            x = layer(x, time_stamp)
        return x


class timestamp(nn.Module):
    def __init__(
        self,
        opt
    ):
        super().__init__()
        self.time_stamp = nn.Embedding(opt.circle, opt.n_attr//4)
        # add temporal embedding and normalize
        self.tempral_enc = TempoEnc(opt.n_his, opt.n_attr//4, opt.TE['no'])

    def forward(self, stamp):
        time_emb = self.time_stamp(stamp)
        time_emb = self.tempral_enc(time_emb)
        return time_emb


class STAGNN_stamp(nn.Module):
    def __init__(
        self,
        opt,
        enc_spa_mask, enc_tem_mask,
        dec_slf_mask, dec_mul_mask
    ):
        super().__init__()
        self.src_pro = SrcProcess(opt)
        self.trg_pro = TrgProcess(opt)
        self.stamp_emb = timestamp(opt)
        self.dec_rdu = ConvExpandAttr(
            opt.n_attr, 1, opt.CE['kernel_size'], opt.CE['bias'])

        self.encoder = Encoder(opt, enc_spa_mask, enc_tem_mask)
        self.decoder = Decoder(opt, dec_slf_mask, dec_mul_mask)

        self.reg_A = opt.reg_A
        self.T4N = opt.T4N['use']
        if self.T4N:
            self.T4N_step = opt.T4N['step']
            self.change_head = opt.T4N['change_head']
            self.change_enc = opt.T4N['change_enc']
            self.T4N_end = opt.T4N['end_epoch']

        self.n_pred = opt.n_pred
        self.n_route = opt.n_route
        self.a = opt.a
        self.n_mask = opt.n_mask
        self.n_c = opt.n_c
    def forward(self, src, time_stamp, label, epoch=1e8):
        src_residual = src
        enc_input = self.src_pro(src, time_stamp)
        time_emb = self.stamp_emb(time_stamp)
        enc_output = self.encoder(enc_input, time_emb)
        enc_output_4head = enc_output

        trg = label[:, :, :self.n_pred, 0].unsqueeze(-1)
        loss = 0.0
        dec_output = None
        if self.T4N and epoch < self.T4N_end:
            for i in range(self.T4N_step):
                dec_input = self.trg_pro(trg, enc_output_4head)

                dec_output = self.decoder(dec_input, enc_output)

                if self.change_head and i < self.T4N_step - 1:
                    pre = enc_output[:, :, 1:, :]
                    post = dec_output[:, :, 0, :].unsqueeze(2)
                    enc_output_4head = torch.cat([pre, post], dim=2)

                if self.change_enc:
                    enc_output = enc_output_4head

                dec_output = self.dec_rdu(dec_output)
                trg = dec_output[:, :, 1:, :]

                loss = loss + \
                    torch.abs(label[:, :, i:i+self.n_pred, :] -
                              dec_output[:, :, :-1, :]).mean()
            A = self.encoder.layer_stack[0].stgc.r1@self.encoder.layer_stack[0].stgc.r1.T
            A_loss = (((A**2).sum())**0.5-self.n_c**0.5)**2
            loss = loss+self.reg_A*A_loss
            return dec_output[:, :, :-1, :], loss
