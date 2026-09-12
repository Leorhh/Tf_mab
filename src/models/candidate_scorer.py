import torch

@torch.no_grad()
def score_candidates(model, hist_items, attn_mask, candidates):
    batch_size, num_candidates = candidates.shape
    hist_items = hist_items.unsqueeze(1).expand(batch_size, num_candidates, hist_items.size(1)).reshape(batch_size * num_candidates, hist_items.size(1))
    attn_mask = attn_mask.unsqueeze(1).expand(batch_size, num_candidates, attn_mask.size(1)).reshape(batch_size * num_candidates, attn_mask.size(1))
    candidates = candidates.reshape(-1)
    predictions = model(hist_items=hist_items, attn_mask=attn_mask, tgt_item=candidates)
    predictions = predictions.reshape(batch_size, num_candidates)
    return predictions
