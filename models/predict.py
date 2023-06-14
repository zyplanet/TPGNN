import torch


def predict(model, src, trg, opt):
    '''
    auto regression
    '''
    model.eval()
    with torch.no_grad():
        src_residual = src

        enc_input = model.src_pro(src)

        enc_output = model.encoder(enc_input)

        lines = trg.shape[2]

        trg = torch.zeros(trg.shape).cuda()

        for i in range(lines):
            if opt.HP['use']:
                head_pre = src_residual[:, :, -1, :].unsqueeze(-1)
                dec_input = model.trg_pro(trg, enc_output, head=head_pre)
            else:
                dec_input = model.trg_pro(trg, enc_output)
            dec_output = model.decoder(dec_input, enc_output)
            dec_output = model.dec_rdu(dec_output)
            trg[:, :, i, :] = dec_output[:, :, i, :]

        return trg


def predict_stamp(model, src, stamp, trg, opt):
    '''
    auto regression
    '''
    model.eval()
    with torch.no_grad():
        src_residual = src

        enc_input = model.src_pro(src, stamp)

        # enc_input = model.enc_exp(src)
        # enc_input = model.enc_spa_enco(enc_input)
        # enc_input = model.enc_tem_enco(enc_input)
        stamp_emb = model.stamp_emb(stamp)
        enc_output = model.encoder(enc_input, stamp_emb)

        lines = trg.shape[2]

        trg = torch.zeros(trg.shape).cuda()

        for i in range(lines):
            dec_input = model.trg_pro(trg, enc_output)
            dec_output = model.decoder(dec_input, enc_output)
            dec_output = model.dec_rdu(dec_output)
            trg[:, :, i, :] = dec_output[:, :, i, :]

        return trg
