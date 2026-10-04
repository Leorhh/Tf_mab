"""MC Dropout uncertainty for the reward model.

Dropout is re-enabled at inference time and the model is run T times;
the spread of the T predictions is a cheap approximation of predictive
uncertainty. No architecture changes needed — the only cost is T extra
forward passes.
"""
import torch
import torch.nn as nn


def _enable_dropout(model):
    """Flip only the Dropout layers back to train mode."""
    for m in model.modules():
        if type(m) == nn.Dropout:
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model, hist_items, attn_mask, tgt_item, num_samples=10):
    model.eval()
    _enable_dropout(model)
    preds = []
    for _ in range(num_samples):
        preds.append(model(hist_items=hist_items, attn_mask=attn_mask, tgt_item=tgt_item))
    preds = torch.stack(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0, unbiased=False)
    return mean_pred, std_pred, preds
