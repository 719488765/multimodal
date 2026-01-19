"""
注意力融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, hidden_dim) - 输入特征
            mask: (B, T) - 注意力掩码
        Returns:
            output: (B, T, hidden_dim) - 输出特征
            attention_weights: (B, num_heads, T, T) - 注意力权重
        """
        B, T, _ = x.shape
        
        # 投影到Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T, T)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)  # (B, num_heads, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)  # (B, T, hidden_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制 - 允许不同模态之间相互关注
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 查询来自模态A，键值来自模态B
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query_modal, key_value_modal, mask=None):
        """
        Args:
            query_modal: (B, T_q, hidden_dim) - 查询模态特征
            key_value_modal: (B, T_kv, hidden_dim) - 键值模态特征
            mask: (B, T_q, T_kv) - 注意力掩码
        Returns:
            output: (B, T_q, hidden_dim) - 输出特征
        """
        B, T_q, _ = query_modal.shape
        _, T_kv, _ = key_value_modal.shape
        
        # 投影
        Q = self.q_proj(query_modal).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value_modal).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value_modal).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T_q, T_kv)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T_q, T_kv)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax和加权求和
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, T_q, self.hidden_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class TemporalAttention(nn.Module):
    """
    时序注意力机制 - 处理时间序列中的长期依赖关系
    """
    def __init__(self, hidden_dim=512, dropout=0.1):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 时序注意力权重计算
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, hidden_dim) - 时序特征
        Returns:
            output: (B, hidden_dim) - 加权后的特征
            attention_weights: (B, T) - 时序注意力权重
        """
        # 计算每个时间步的注意力分数
        attention_scores = self.temporal_attention(x).squeeze(-1)  # (B, T)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)  # (B, hidden_dim)
        
        return output, attention_weights


class MultimodalFusion(nn.Module):
    """
    多模态特征融合模块 - 整合所有模态特征
    """
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=3, dropout=0.1):
        super(MultimodalFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 跨模态注意力层（两两模态之间）
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 前馈神经网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers * 2)
        ])
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(hidden_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat):
        """
        Args:
            video_feat: (B, T_v, hidden_dim) 或 (B, hidden_dim)
            audio_feat: (B, hidden_dim)
            physiological_feat: (B, hidden_dim)
            text_feat: (B, hidden_dim)
        Returns:
            fused_features: (B, hidden_dim) - 融合后的特征
        """
        # 确保所有特征都是2D (B, hidden_dim)
        if len(video_feat.shape) == 3:
            # 如果有时序维度，使用时序注意力
            video_feat, _ = self.temporal_attention(video_feat)
        
        # 将所有模态特征堆叠: (B, num_modalities, hidden_dim)
        modalities = torch.stack([video_feat, audio_feat, physiological_feat, text_feat], dim=1)
        B, num_modalities, hidden_dim = modalities.shape
        
        # 通过多层Transformer编码器
        for i in range(len(self.self_attention_layers)):
            # 自注意力
            residual = modalities
            modalities = modalities.view(B * num_modalities, 1, hidden_dim)  # 展平以便处理
            modalities, _ = self.self_attention_layers[i](modalities)
            modalities = modalities.view(B, num_modalities, hidden_dim)
            modalities = self.layer_norms[i * 2](modalities + residual)
            
            # 前馈网络
            residual = modalities
            modalities = self.ffn_layers[i](modalities)
            modalities = self.layer_norms[i * 2 + 1](modalities + residual)
        
        # 跨模态注意力（简化版：使用平均池化后的特征进行跨模态交互）
        # 这里简化处理，实际可以更复杂
        fused_features = torch.mean(modalities, dim=1)  # (B, hidden_dim)
        
        return fused_features

