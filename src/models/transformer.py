"""Reward model: Transformer encoder over the interaction history.

The history is encoded once (mean-pooled over valid positions) and
concatenated with the candidate item embedding; an MLP head maps the
pair to a reward in [0, 1] via a sigmoid. Mean pooling is a deliberate
simplification — it treats the history as a bag of items and trades
order sensitivity for a stable, cheap user representation.
"""
import torch
import torch.nn as nn


class RewardTransformer(nn.Module):
    def __init__(
        self,
        num_items,
        max_seq_len=20,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
        init_bias=None,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.reward_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        if init_bias is not None:
            # start predictions near the dataset's average reward.
            # with a bare sigmoid and sparse (mostly-zero) targets, the head
            # gets dragged into saturation within the first epoch and the
            # gradients die there; logit(mean_reward) avoids that dive.
            nn.init.constant_(self.reward_head[3].bias, init_bias)

    def encode_history(self, hist_items, attn_mask):
        B, L = hist_items.shape
        positions = torch.arange(L, device=hist_items.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(hist_items) + self.pos_emb(positions)

        padding_mask = hist_items.eq(self.padding_idx)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # mean-pool over non-padded positions only
        valid = (~padding_mask).unsqueeze(-1)
        x = x * valid
        lengths = valid.sum(dim=1).clamp(min=1)
        return x.sum(dim=1) / lengths

    def forward(self, hist_items, attn_mask, tgt_item):
        user = self.encode_history(hist_items, attn_mask)
        item = self.item_emb(tgt_item)
        return self.reward_head(torch.cat([user, item], dim=-1)).squeeze(-1)
