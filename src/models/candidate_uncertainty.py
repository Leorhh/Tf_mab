import torch
import torch.nn as nn

# sequences per forward pass; 25k sequences x 50 positions x 128 dims
# is what killed the 16 GB Mac, keep chunks comfortably below that
CHUNK_SIZE = 4096


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def score_with_uncertainty(model, hist_items, attn_mask, candidates, num_samples=10):
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")

    B, K = candidates.shape
    L = hist_items.size(1)
    hist = hist_items.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
    mask = attn_mask.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
    flat = candidates.reshape(-1)

    model.eval()
    _enable_dropout(model)
    device = hist_items.device
    preds = []
    for _ in range(num_samples):
        out = torch.empty(B * K, device=device)
        for s in range(0, B * K, CHUNK_SIZE):
            e = s + CHUNK_SIZE
            out[s:e] = model(hist_items=hist[s:e], attn_mask=mask[s:e], tgt_item=flat[s:e])
        preds.append(out.reshape(B, K))
    preds = torch.stack(preds, dim=0)
    if device.type == "mps":
        torch.mps.empty_cache()
    mean_pred = preds.mean(dim=0)
    uncertainty = preds.std(dim=0, unbiased=False)
    return mean_pred, uncertainty, preds
