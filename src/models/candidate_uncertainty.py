import torch
import torch.nn as nn

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.no_grad()
def score_candidates_with_uncertainty(model, hist_items, attn_mask, candidates, num_samples=10):
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    batch_size, num_candidates = candidates.shape
    expanded_hist = hist_items.unsqueeze(1).expand(batch_size, num_candidates, hist_items.size(1)).reshape(batch_size * num_candidates, hist_items.size(1))
    expanded_mask = attn_mask.unsqueeze(1).expand(batch_size, num_candidates, attn_mask.size(1)).reshape(batch_size * num_candidates, attn_mask.size(1))
    flat_candidates = candidates.reshape(-1)

    model.eval()
    enable_dropout(model)
    predictions = []
    for _ in range(num_samples):
        pred = model(hist_items=expanded_hist, attn_mask=expanded_mask, tgt_item=flat_candidates)
        pred = pred.reshape(batch_size, num_candidates)
        predictions.append(pred)
    predictions = torch.stack(predictions, dim=0)
    mean_prediction = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0, unbiased=False)
    return mean_prediction, uncertainty, predictions
