"""
情感转变感知模块
借鉴自：CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition
论文链接：https://arxiv.org/abs/2307.15432
源码地址：https://github.com/jianglil/Cross-Modal-Fusion-Network

核心思想：
- 捕捉对话/序列中的情感变化
- 通过情感转移模块建模情感状态的动态演变
- 适用于驾驶员情绪的动态监测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionShiftAwareness(nn.Module):
    """
    情感转变感知模块
    用于捕捉时序中的情感变化模式
    """
    def __init__(self, hidden_dim=512, num_emotion_classes=7, dropout=0.1):
        super(EmotionShiftAwareness, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_emotion_classes = num_emotion_classes
        
        # 情感状态编码器
        self.emotion_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotion_classes)
        )
        
        # 情感转变检测器（使用LSTM捕捉时序变化）
        self.shift_detector = nn.LSTM(
            input_size=num_emotion_classes,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 转变特征融合
        self.shift_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # 原始特征 + 转变特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features, previous_emotions=None):
        """
        Args:
            features: (B, T, hidden_dim) - 时序特征
            previous_emotions: (B, T_prev, num_classes) - 之前时间步的情感预测（可选）
        Returns:
            enhanced_features: (B, T, hidden_dim) - 增强后的特征（包含情感转变信息）
            emotion_logits: (B, T, num_classes) - 每个时间步的情感预测
            shift_weights: (B, T) - 情感转变权重
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            while features.dim() > 3:
                features = features.mean(dim=1)
            if features.dim() == 2:
                features = features.unsqueeze(1)

        B, T, _ = features.shape
        features = self.input_norm(features)
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        # 预测每个时间步的情感状态
        emotion_logits = self.emotion_encoder(features)  # (B, T, num_classes)
        emotion_logits = torch.clamp(emotion_logits, -20.0, 20.0)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # 计算情感转变
        if previous_emotions is not None:
            # 如果有之前的情感状态，计算转变
            all_emotions = torch.cat([previous_emotions, emotion_probs.detach()], dim=1)
        else:
            all_emotions = emotion_probs.detach()
        
        # 使用LSTM检测情感转变模式
        shift_features, _ = self.shift_detector(all_emotions)  # (B, T_prev+T, hidden_dim)
        
        # 只取当前时间步的转变特征
        if previous_emotions is not None:
            shift_features = shift_features[:, -T:, :]  # (B, T, hidden_dim)
        elif shift_features.shape[1] > T:
            shift_features = shift_features[:, -T:, :]
        
        shift_features = torch.nan_to_num(shift_features, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 融合原始特征和转变特征
        combined = torch.cat([features, shift_features], dim=-1)  # (B, T, hidden_dim*2)
        enhanced_features = self.shift_fusion(combined)  # (B, T, hidden_dim)
        enhanced_features = torch.nan_to_num(enhanced_features, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 计算情感转变强度（通过相邻时间步的情感差异）
        if T > 1:
            emotion_diff = torch.abs(emotion_probs[:, 1:, :] - emotion_probs[:, :-1, :])
            shift_weights = torch.sum(emotion_diff, dim=-1)  # (B, T-1)
            # 填充第一个时间步
            shift_weights = F.pad(shift_weights, (1, 0), value=0.0)  # (B, T)
        else:
            shift_weights = torch.zeros(B, 1, device=features.device)
        
        return enhanced_features, emotion_logits, shift_weights


class EmotionShiftFusion(nn.Module):
    """
    结合情感转变感知的跨模态融合模块
    扩展了CFN-ESA的思想到多模态场景
    """
    def __init__(
        self,
        hidden_dim=512,
        num_emotion_classes=7,
        num_heads=8,
        dropout=0.1,
        leader_modal="text",
    ):
        super(EmotionShiftFusion, self).__init__()
        self.leader_modal = leader_modal
        self._modal_names = ("video", "audio", "physiological", "text")
        self._kv_modal_names = ("video", "audio", "physiological")

        # 情感转变感知模块
        self.emotion_shift = EmotionShiftAwareness(hidden_dim, num_emotion_classes, dropout)
        
        # 跨模态注意力（参考CFN-ESA）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 模态权重学习（文本作为主要模态，其他作为次要模态）
        self.modal_weights = nn.Parameter(torch.ones(4) / 4)  # 4个模态：视频、音频、生理、文本
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        video_feat,
        audio_feat,
        physiological_feat,
        text_feat,
        previous_emotions=None,
        active_mask=None,
    ):
        """
        Args:
            video_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            audio_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            physiological_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            text_feat: (B, hidden_dim) 或 (B, T, hidden_dim) - 主要模态
            previous_emotions: (B, T_prev, num_classes) - 之前的情感状态
            active_mask: dict[str, bool] - 激活模态（消融实验）
        Returns:
            fused_features: (B, hidden_dim) 或 (B, T, hidden_dim)
            emotion_logits: (B, num_classes) 或 (B, T, num_classes)
            shift_weights: (B,) 或 (B, T)
        """
        if active_mask is None:
            active_mask = {name: True for name in self._modal_names}

        def _pool_time(feat):
            if feat is None:
                return None
            if feat.dim() == 3:
                return feat.mean(dim=1)
            return feat

        modal_feats = {
            "video": _pool_time(video_feat),
            "audio": _pool_time(audio_feat),
            "physiological": _pool_time(physiological_feat),
            "text": _pool_time(text_feat),
        }

        # 统一维度处理（clip 级分类：(B, hidden_dim) -> (B, 1, hidden_dim)）
        is_temporal = False
        for name in self._modal_names:
            feat = modal_feats[name]
            if feat is not None:
                modal_feats[name] = feat.unsqueeze(1)

        weights = F.softmax(self.modal_weights, dim=0)
        weighted_parts = []
        weight_sum = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
        for idx, name in enumerate(self._modal_names):
            if not active_mask.get(name, True):
                continue
            feat = modal_feats[name]
            if feat is None:
                continue
            w = weights[idx]
            weighted_parts.append(w * feat)
            weight_sum = weight_sum + w

        if weighted_parts:
            weighted_feat = sum(weighted_parts) / weight_sum.clamp(min=1e-8)
        else:
            weighted_feat = None
            for name in self._modal_names:
                if modal_feats.get(name) is not None:
                    weighted_feat = modal_feats[name]
                    break
            if weighted_feat is None:
                raise ValueError("No active modalities for EmotionShiftFusion")

        # 应用情感转变感知
        enhanced_feat, emotion_logits, shift_weights = self.emotion_shift(
            weighted_feat, previous_emotions
        )

        leader = self.leader_modal
        if not active_mask.get(leader, True):
            for name in self._modal_names:
                if active_mask.get(name, True) and modal_feats[name] is not None:
                    leader = name
                    break
        query_feat = modal_feats.get(leader)
        if query_feat is None:
            query_feat = enhanced_feat
        text_query = query_feat.view(-1, 1, query_feat.shape[-1])

        kv_list = []
        for name in self._kv_modal_names:
            if active_mask.get(name, True) and modal_feats.get(name) is not None:
                kv_list.append(modal_feats[name])

        if kv_list:
            other_modals = torch.stack(kv_list, dim=2)
            other_modals = other_modals.view(-1, len(kv_list), enhanced_feat.shape[-1])
            attended_feat, _ = self.cross_attention(
                query=text_query,
                key=other_modals,
                value=other_modals,
            )
            attended_feat = attended_feat.view(enhanced_feat.shape)
            output = self.layer_norm(enhanced_feat + attended_feat)
        else:
            output = self.layer_norm(enhanced_feat)

        output = self.dropout(output)
        
        if not is_temporal:
            # 压缩时序维度
            output = output.squeeze(1)
            emotion_logits = emotion_logits.squeeze(1)
            shift_weights = shift_weights.squeeze(1)
        
        return output, emotion_logits, shift_weights

