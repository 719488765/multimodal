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
        B, T, _ = features.shape
        
        # 预测每个时间步的情感状态
        emotion_logits = self.emotion_encoder(features)  # (B, T, num_classes)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # 计算情感转变
        if previous_emotions is not None:
            # 如果有之前的情感状态，计算转变
            all_emotions = torch.cat([previous_emotions, emotion_probs], dim=1)  # (B, T_prev+T, num_classes)
        else:
            all_emotions = emotion_probs
        
        # 使用LSTM检测情感转变模式
        shift_features, _ = self.shift_detector(all_emotions)  # (B, T_prev+T, hidden_dim)
        
        # 只取当前时间步的转变特征
        if previous_emotions is not None:
            shift_features = shift_features[:, -T:, :]  # (B, T, hidden_dim)
        
        # 融合原始特征和转变特征
        combined = torch.cat([features, shift_features], dim=-1)  # (B, T, hidden_dim*2)
        enhanced_features = self.shift_fusion(combined)  # (B, T, hidden_dim)
        
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
    def __init__(self, hidden_dim=512, num_emotion_classes=7, num_heads=8, dropout=0.1):
        super(EmotionShiftFusion, self).__init__()
        
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
        
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat, 
                previous_emotions=None):
        """
        Args:
            video_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            audio_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            physiological_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            text_feat: (B, hidden_dim) 或 (B, T, hidden_dim) - 主要模态
            previous_emotions: (B, T_prev, num_classes) - 之前的情感状态
        Returns:
            fused_features: (B, hidden_dim) 或 (B, T, hidden_dim)
            emotion_logits: (B, num_classes) 或 (B, T, num_classes)
            shift_weights: (B,) 或 (B, T)
        """
        # 统一维度处理
        is_temporal = len(text_feat.shape) == 3
        
        if not is_temporal:
            # 扩展到时序维度
            video_feat = video_feat.unsqueeze(1)  # (B, 1, hidden_dim)
            audio_feat = audio_feat.unsqueeze(1)
            physiological_feat = physiological_feat.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
        
        # 加权融合多模态特征（文本权重更高）
        weighted_feat = (
            self.modal_weights[0] * video_feat +
            self.modal_weights[1] * audio_feat +
            self.modal_weights[2] * physiological_feat +
            self.modal_weights[3] * text_feat
        )
        
        # 应用情感转变感知
        enhanced_feat, emotion_logits, shift_weights = self.emotion_shift(
            weighted_feat, previous_emotions
        )
        
        # 跨模态注意力（以文本为query，其他模态为key和value）
        other_modals = torch.stack([video_feat, audio_feat, physiological_feat], dim=2)  # (B, T, 3, hidden_dim)
        other_modals = other_modals.view(-1, 3, enhanced_feat.shape[-1])  # (B*T, 3, hidden_dim)
        text_query = enhanced_feat.view(-1, 1, enhanced_feat.shape[-1])  # (B*T, 1, hidden_dim)
        
        attended_feat, _ = self.cross_attention(
            query=text_query,
            key=other_modals,
            value=other_modals
        )  # (B*T, 1, hidden_dim)
        
        attended_feat = attended_feat.view(enhanced_feat.shape)  # (B, T, hidden_dim)
        
        # 残差连接和层归一化
        output = self.layer_norm(enhanced_feat + attended_feat)
        output = self.dropout(output)
        
        if not is_temporal:
            # 压缩时序维度
            output = output.squeeze(1)
            emotion_logits = emotion_logits.squeeze(1)
            shift_weights = shift_weights.squeeze(1)
        
        return output, emotion_logits, shift_weights

