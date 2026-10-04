"""Score a batch of candidate sets against their interaction histories."""
import torch


@torch.no_grad()
def score_candidates(model, hist_items, attn_mask, candidates):
    # expand each history across its candidates, then flatten so the model
    # sees a plain (B*K, L) batch; reshape back to (B, K) afterwards
    B, K = candidates.shape
    L = hist_items.size(1)
    hist = hist_items.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
    mask = attn_mask.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
    flat = candidates.reshape(-1)

    preds = model(hist_items=hist, attn_mask=mask, tgt_item=flat)
    return preds.reshape(B, K)
