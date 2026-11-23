"""Emotion-Shift Fusion (ESF) temporal core.

本模块是论文第 3 章 ESF 的实现对应物。相对早期版本的修正：

1. **不再对时间维做池化。** 当输入为 ``(B, T, D)`` 时，状态编码、迁移建模与
   跨模态定向聚合都在完整时间轴上计算；只有输入为 ``(B, D)``（片段级）时才
   退化为单步。
2. **``previous_emotions`` 真正参与计算。** 它可以是从上一窗口继承的状态分布
   ``(B, P, K)``，与当前窗口的逐帧状态分布拼接后送入迁移编码器；训练时可用
   教师强制的前一状态，推理时用模型自己滚动的预测状态。
3. **状态编码器有独立监督。** ``forward`` 返回逐时间步的 ``state_logits``，训练
   目标应把它作为辅助分类损失（见 ``scripts/train.py`` 的 ``shift_aux``），否则
   状态编码器只能通过融合残差间接学习，容易塌缩。
4. **位移权重按状态概率序列计算**，而不是按最终分类 logits；首个时间步可以与
   继承的 ``previous_emotions`` 比较，因此窗口边界处的转折不会被漏掉。

所有形状约定：
    features:          (B, T, D) 或 (B, D)
    previous_emotions: (B, P, K) 或 (B, K) 或 None
    valid_mask:        (B, T) 或 None，1 表示该时间步有效
返回：
    enhanced:     (B, T, D) 或 (B, D)
    state_logits: (B, T, K) 或 (B, K)
    shift_weight: (B, T) 或 (B,)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionShiftAwareness(nn.Module):
    """逐时间步状态编码 + 状态迁移建模。

    Args:
        hidden_dim: 融合隐表示维数 D。
        num_emotion_classes: 情绪类别数 K。
        dropout: dropout 比例。
        num_layers: 迁移编码器层数。
        causal: True 时使用单向 LSTM（在线推理）；False 时使用双向 LSTM
            （离线判别）。双向结构读取的是观测派生的状态分布，不读取未来标签，
            因此不违反第 3 章式 (3.10) 的因果性约束。
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_emotion_classes: int = 7,
        dropout: float = 0.1,
        num_layers: int = 2,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_emotion_classes = int(num_emotion_classes)
        self.causal = bool(causal)

        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_emotion_classes),
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.shift_detector = nn.LSTM(
            input_size=self.num_emotion_classes,
            hidden_size=self.hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=not self.causal,
            dropout=lstm_dropout,
        )
        lstm_out_dim = self.hidden_dim if not self.causal else self.hidden_dim // 2
        # 将迁移特征补回 D 维，便于与原始表示做残差。
        self.shift_proj = nn.Linear(lstm_out_dim, self.hidden_dim)
        self.shift_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # 门控让模型在“情绪平稳”时自动关小迁移残差，避免伤害静态样本。
        self.shift_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    @staticmethod
    def _as_3d(features: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if features.dim() == 2:
            return features.unsqueeze(1), True
        if features.dim() != 3:
            raise ValueError(f"features must be (B,D) or (B,T,D), got {tuple(features.shape)}")
        return features, False

    def _prefix_probs(self, previous_emotions: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
        if previous_emotions is None:
            return None
        prev = previous_emotions
        if prev.dim() == 2:
            prev = prev.unsqueeze(1)
        if prev.dim() != 3 or prev.size(-1) != self.num_emotion_classes:
            raise ValueError(
                "previous_emotions must be (B,K) or (B,P,K) with K="
                f"{self.num_emotion_classes}, got {tuple(prev.shape)}"
            )
        return prev.to(device=device, dtype=torch.float32)

    def forward(
        self,
        features: torch.Tensor,
        previous_emotions: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, squeezed = self._as_3d(features)
        B, T, _ = features.shape
        device = features.device

        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
        normed = self.input_norm(features)

        state_logits = torch.clamp(self.emotion_encoder(normed), -20.0, 20.0)
        probs = F.softmax(state_logits, dim=-1)

        prev = self._prefix_probs(previous_emotions, device)
        # detach：迁移编码器不通过“当前预测”反传，避免状态塌缩为标签平滑；
        # 状态编码器由 state_logits 的辅助损失单独监督。
        current = probs.detach()
        seq = torch.cat([prev, current], dim=1) if prev is not None else current

        shift_features, _ = self.shift_detector(seq)
        shift_features = shift_features[:, -T:, :]
        shift_features = self.shift_proj(shift_features)
        shift_features = torch.nan_to_num(shift_features, nan=0.0, posinf=1e4, neginf=-1e4)

        combined = torch.cat([features, shift_features], dim=-1)
        delta = self.shift_fusion(combined)
        gate = torch.sigmoid(self.shift_gate(combined))
        enhanced = features + gate * delta
        enhanced = torch.nan_to_num(enhanced, nan=0.0, posinf=1e4, neginf=-1e4)

        # 位移强度：状态分布的一阶差分。首步与继承状态比较。
        if T > 1:
            diff = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]).sum(dim=-1)
        else:
            diff = torch.zeros(B, 0, device=device, dtype=probs.dtype)
        if prev is not None:
            first = torch.abs(probs[:, 0, :] - prev[:, -1, :]).sum(dim=-1, keepdim=True)
            shift_weight = torch.cat([first, diff], dim=1)
        else:
            shift_weight = F.pad(diff, (1, 0), value=0.0)
        shift_weight = shift_weight.to(dtype=probs.dtype)

        if valid_mask is not None:
            mask = valid_mask.to(device=device, dtype=probs.dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask[:, :T]
            enhanced = enhanced * mask.unsqueeze(-1)
            shift_weight = shift_weight * mask
            state_logits = state_logits * mask.unsqueeze(-1)

        if squeezed:
            return enhanced.squeeze(1), state_logits.squeeze(1), shift_weight.squeeze(1)
        return enhanced, state_logits, shift_weight


class EmotionShiftFusion(nn.Module):
    """带时序状态与定向聚合的多模态融合。

    与旧实现的接口保持兼容：2D 输入返回 2D 输出；3D 输入返回 3D 输出。
    ``active_mask`` 仍接受全局 dict；如传入 ``modality_mask`` (B,4)，则按样本
    逐时间步屏蔽缺失模态。
    """

    _MODAL_NAMES = ("video", "audio", "physiological", "text")

    def __init__(
        self,
        hidden_dim: int = 512,
        num_emotion_classes: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1,
        leader_modal: str = "text",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.leader_modal = leader_modal
        self.dropout = nn.Dropout(dropout)

        self.emotion_shift = EmotionShiftAwareness(
            hidden_dim, num_emotion_classes, dropout, causal=causal
        )
        self.modal_weights = nn.Parameter(torch.ones(4) / 4)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _prepare(self, feat: Optional[torch.Tensor], B: int, T: int, device):
        if feat is None:
            return torch.zeros(B, T, self.hidden_dim, device=device)
        if feat.dim() == 2:
            feat = feat.unsqueeze(1).expand(B, T, self.hidden_dim)
        if feat.dim() != 3:
            raise ValueError(f"modality feature must be (B,D) or (B,T,D), got {tuple(feat.shape)}")
        if feat.size(1) != T:
            raise ValueError(f"time dimension mismatch: {feat.size(1)} != {T}")
        return feat

    def _leader_attention(self, query, keys):
        """单 leader、多 follower 的逐时间步定向注意力。

        query: (B,T,D); keys: (B,T,M,D) -> (B,T,D)
        """
        B, T, D = query.shape
        M = keys.size(2)
        H = max(1, min(self.num_heads, D // 32))
        head_dim = D // H
        q = query.view(B * T, 1, H, head_dim).transpose(1, 2)
        k = keys.reshape(B * T, M, H, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, k)
        out = out.transpose(1, 2).reshape(B, T, D)
        return out

    def forward(
        self,
        video_feat,
        audio_feat,
        physiological_feat,
        text_feat,
        previous_emotions=None,
        active_mask=None,
        modality_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        if active_mask is None:
            active_mask = {name: True for name in self._MODAL_NAMES}

        feats = {
            "video": video_feat,
            "audio": audio_feat,
            "physiological": physiological_feat,
            "text": text_feat,
        }
        # 推断时间维：优先取第一个 3D 输入的 T，否则为 1。
        T = 1
        for f in feats.values():
            if f is not None and f.dim() == 3:
                T = f.size(1)
                break
        B = next((f.size(0) for f in feats.values() if f is not None), 1)
        device = next((f.device for f in feats.values() if f is not None), self.modal_weights.device)

        prepared = {name: self._prepare(f, B, T, device) for name, f in feats.items()}
        weights = F.softmax(self.modal_weights, dim=0)

        # (B,T,4) 的逐样本/逐时间步可用性；全局 dict 作为兜底。
        if modality_mask is not None:
            m = modality_mask.to(device=device, dtype=weights.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(1).expand(B, T, m.size(-1))
            m = m[:, :T, :]
        else:
            base = torch.tensor(
                [1.0 if active_mask.get(n, True) else 0.0 for n in self._MODAL_NAMES],
                device=device,
                dtype=weights.dtype,
            )
            m = base.view(1, 1, 4).expand(B, T, 4)

        stacked = torch.stack([prepared[n] for n in self._MODAL_NAMES], dim=2)  # B,T,4,D
        masked_w = weights.view(1, 1, 4) * m
        denom = masked_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = masked_w / denom
        weighted = (alpha.unsqueeze(-1) * stacked).sum(dim=2)  # B,T,D

        enhanced, state_logits, shift_weight = self.emotion_shift(
            weighted, previous_emotions=previous_emotions, valid_mask=valid_mask
        )

        # leader 选择：全局 leader 不可用时退化为第一个可用模态。
        leader = self.leader_modal
        if not active_mask.get(leader, True):
            leader = next((n for n in self._MODAL_NAMES if active_mask.get(n, True)), "text")
        lead_idx = self._MODAL_NAMES.index(leader)

        # 逐样本：若 leader 在该样本全部时间步都缺失，则用加权表示自身做查询。
        lead_feat = prepared[leader]
        if modality_mask is not None:
            lead_valid = m[:, :, lead_idx].sum(dim=1) > 0
            if not bool(lead_valid.all()):
                lead_feat = torch.where(lead_valid.view(B, 1, 1), lead_feat, enhanced)

        follower_idx = [i for i, n in enumerate(self._MODAL_NAMES) if n != leader]
        follower = torch.stack([prepared[self._MODAL_NAMES[i]] for i in follower_idx], dim=2)
        attended = self._leader_attention(lead_feat, follower)
        output = self.layer_norm(enhanced + self.dropout(attended))

        if T == 1 and all((f is None or f.dim() == 2) for f in feats.values()):
            return output.squeeze(1), state_logits.squeeze(1), shift_weight.squeeze(1)
        return output, state_logits, shift_weight
