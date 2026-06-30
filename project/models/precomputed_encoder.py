"""
预提取时序特征编码器（MOSEI OpenFace / COVAREP 等）
BiLSTM + 注意力池化，替代单层 Linear + mean-pool。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def subsample_temporal_features(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """均匀下采样 (B, T, F) -> (B, max_seq_len, F)。"""
    if x is None or x.dim() != 3 or max_seq_len <= 0:
        return x
    t = x.shape[1]
    if t <= max_seq_len:
        return x
    idx = torch.linspace(0, t - 1, max_seq_len, device=x.device).long()
    return x[:, idx, :]


class TemporalNpyEncoder(nn.Module):
    """(B, T, input_dim) -> (B, T_out, output_dim)，T_out = min(T, max_seq_len)。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 256,
        max_seq_len: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.max_seq_len = int(max_seq_len)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_pooled: bool = False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"TemporalNpyEncoder expects (B,T,F), got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dim {x.shape[-1]} != expected {self.input_dim}"
            )
        x = subsample_temporal_features(x, self.max_seq_len)
        x = self.input_norm(x)
        x = F.relu(self.input_proj(x))
        x, _ = self.lstm(x)
        pooled = None
        if return_pooled:
            scores = self.attn(x).squeeze(-1)
            weights = F.softmax(scores, dim=1)
            pooled_h = torch.sum(weights.unsqueeze(-1) * x, dim=1)
            pooled = self.out_proj(pooled_h)
        x = self.out_proj(x)
        if not return_pooled:
            return x
        return x, pooled
