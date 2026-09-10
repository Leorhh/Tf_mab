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
        self.pad_idx = padding_idx

        self.item_emb = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=d_model,
            padding_idx=self.pad_idx,
        )
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
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def encode_history(self, hist_items, attn_mask):
        bsz, seq_len = hist_items.shape
        x = self.item_emb(hist_items)

        pos = torch.arange(seq_len, device=hist_items.device)
        pos_emb = self.pos_emb(pos).unsqueeze(0)
        x = x + pos_emb

        pad_mask = attn_mask == 0
        x = self.transformer_enc(x, src_key_padding_mask=pad_mask)

        mask = attn_mask.unsqueeze(-1).float()
        x = x * mask
        sum_emb = x.sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1.0)
        user_repr = sum_emb / length
        return user_repr

    def forward(self, hist_items, attn_mask, tgt_item):
        user_repr = self.encode_history(hist_items, attn_mask)
        tgt_emb = self.item_emb(tgt_item)
        combined = torch.cat([user_repr, tgt_emb], dim=-1)
        pred_rew = self.reward_head(combined).squeeze(-1)
        return pred_rew


if __name__ == "__main__":
    import json
    from torch.utils.data import DataLoader
    from src.data.sequence_dataset import AmazonSequenceDataset

    print("=" * 60)
    print("Testing Transformer Reward Model")
    print("=" * 60)

    ds = AmazonSequenceDataset(
        sequence_path="data/processed/amazon/train_sequences.csv",
        mapping_path="data/processed/amazon/item_mapping.json",
        max_seq_len=20,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    with open("data/processed/amazon/item_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    model = TransformerRewardModel(
        num_items=mapping["num_items"],
        max_seq_len=20,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    print("\n[INFO] Model created.")

    with torch.no_grad():
        reward_pred = model(
            hist_items=batch["history_items"],
            attn_mask=batch["attention_mask"],
            tgt_item=batch["target_item"],
        )

    print("\n[TEST] Input shapes:")
    print("history_items:", batch["history_items"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("target_item:", batch["target_item"].shape)

    print("\n[TEST] Output:")
    print("reward_pred:", reward_pred.shape)
    print("reward_pred[:10]:", reward_pred[:10])
    print("min:", reward_pred.min().item())
    print("max:", reward_pred.max().item())

    print("\n" + "=" * 60)
    print("Transformer forward test completed.")
    print("=" * 60)
