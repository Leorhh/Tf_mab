import torch
import torch.nn as nn


def enable_dropout(model):
    for m in model.modules():
        if type(m) == nn.Dropout:
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, hist_items, attn_mask, tgt_item, num_samples=10):
    model.eval()
    enable_dropout(model)
    preds = []
    for _ in range(num_samples):
        out = model(hist_items=hist_items, attn_mask=attn_mask, tgt_item=tgt_item)
        preds.append(out)
    preds = torch.stack(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0, unbiased=False)
    return mean_pred, std_pred, preds
