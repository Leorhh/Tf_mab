import torch
import torch.nn as nn

class TransformerRewardModel(nn.Module):
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
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.reward_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode_history(self, hist_items, attn_mask):
        batch_size, seq_len = hist_items.shape
        positions = torch.arange(seq_len, device=hist_items.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_emb(hist_items) + self.pos_emb(positions)
        padding_mask = hist_items.eq(self.padding_idx)
        x = self.transformer_enc(x, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1)
        x = x * valid_mask
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        user_repr = x.sum(dim=1) / lengths
        # user_repr = self.norm(user_repr)
        return user_repr

    def forward(self, hist_items, attn_mask, tgt_item):
        user_repr = self.encode_history(hist_items, attn_mask)
        tgt_emb = self.item_emb(tgt_item)
        combined = torch.cat([user_repr, tgt_emb], dim=-1)
        pred_rew = self.reward_head(combined).squeeze(-1)
        return pred_rew
