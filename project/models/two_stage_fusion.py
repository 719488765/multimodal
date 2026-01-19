"""
两阶段多源信息融合
借鉴自：GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection
论文链接：https://arxiv.org/abs/2207.11900

核心思想：
- 第一阶段：使用图注意力网络（GAT）进行上下文建模
- 第二阶段：使用跨模态注意力进行多模态融合
- 两阶段设计可以更好地捕捉模态内和模态间的依赖关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import MultiHeadSelfAttention, CrossModalAttention


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（简化版）
    用于建模模态之间的图结构关系
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 注意力权重计算
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, adj_matrix=None):
        """
        Args:
            features: (B, num_nodes, hidden_dim) - 节点特征（每个模态作为一个节点）
            adj_matrix: (B, num_nodes, num_nodes) - 邻接矩阵（可选）
        Returns:
            output: (B, num_nodes, hidden_dim) - 更新后的节点特征
        """
        # 使用自注意力模拟图注意力
        residual = features
        output, _ = self.attention(features, features, features)
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output


class TwoStageFusion(nn.Module):
    """
    两阶段融合模块
    第一阶段：上下文建模（使用图注意力网络）
    第二阶段：跨模态融合（使用跨模态注意力）
    """
    def __init__(self, hidden_dim=512, num_heads=8, num_gat_layers=2, num_fusion_layers=2, dropout=0.1):
        super(TwoStageFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 第一阶段：图注意力网络（上下文建模）
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gat_layers)
        ])
        
        # 第二阶段：跨模态注意力融合
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # 自注意力层（用于模态内部建模）
        self.self_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_fusion_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_fusion_layers * 2)
        ])
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat):
        """
        Args:
            video_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            audio_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            physiological_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            text_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
        Returns:
            fused_features: (B, hidden_dim) 或 (B, T, hidden_dim)
        """
        is_temporal = len(text_feat.shape) == 3
        
        if not is_temporal:
            # 扩展到时序维度
            video_feat = video_feat.unsqueeze(1)  # (B, 1, hidden_dim)
            audio_feat = audio_feat.unsqueeze(1)
            physiological_feat = physiological_feat.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
        
        # 堆叠所有模态: (B, num_modalities, hidden_dim)
        modalities = torch.stack([video_feat, audio_feat, physiological_feat, text_feat], dim=1)
        B, num_modalities, T, hidden_dim = modalities.shape
        modalities = modalities.view(B, num_modalities, T * hidden_dim)  # 简化处理
        
        # 重新reshape以适应后续处理
        if is_temporal:
            # 对于时序数据，我们需要重新组织
            modalities = modalities.view(B, num_modalities, T, hidden_dim)
            # 将时间维度合并到批次维度
            modalities = modalities.view(B * num_modalities, T, hidden_dim)
        else:
            modalities = modalities.view(B, num_modalities, hidden_dim)
        
        # 第一阶段：图注意力网络（上下文建模）
        for gat_layer in self.gat_layers:
            if is_temporal:
                # 对每个时间步应用GAT
                modalities = modalities.view(B, num_modalities, T, hidden_dim)
                modalities_list = []
                for t in range(T):
                    mod_t = modalities[:, :, t, :]  # (B, num_modalities, hidden_dim)
                    mod_t = gat_layer(mod_t)
                    modalities_list.append(mod_t)
                modalities = torch.stack(modalities_list, dim=2)  # (B, num_modalities, T, hidden_dim)
                modalities = modalities.view(B * num_modalities, T, hidden_dim)
            else:
                modalities = gat_layer(modalities)
        
        # 恢复原始形状
        if is_temporal:
            modalities = modalities.view(B, num_modalities, T, hidden_dim)
            # 对每个模态分别处理
            video_feat = modalities[:, 0, :, :]  # (B, T, hidden_dim)
            audio_feat = modalities[:, 1, :, :]
            physiological_feat = modalities[:, 2, :, :]
            text_feat = modalities[:, 3, :, :]
        else:
            video_feat = modalities[:, 0, :]  # (B, hidden_dim)
            audio_feat = modalities[:, 1, :]
            physiological_feat = modalities[:, 2, :]
            text_feat = modalities[:, 3, :]
            # 扩展维度
            video_feat = video_feat.unsqueeze(1)
            audio_feat = audio_feat.unsqueeze(1)
            physiological_feat = physiological_feat.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
        
        # 第二阶段：跨模态注意力融合
        for i in range(len(self.cross_attention_layers)):
            # 自注意力（模态内部）
            video_feat, _ = self.self_attention_layers[i](video_feat)
            audio_feat, _ = self.self_attention_layers[i](audio_feat)
            physiological_feat, _ = self.self_attention_layers[i](physiological_feat)
            text_feat, _ = self.self_attention_layers[i](text_feat)
            
            # 跨模态注意力（以文本为query）
            enhanced_video = self.cross_attention_layers[i](text_feat, video_feat)
            enhanced_audio = self.cross_attention_layers[i](text_feat, audio_feat)
            enhanced_physiological = self.cross_attention_layers[i](text_feat, physiological_feat)
            
            # 残差连接和层归一化
            video_feat = self.layer_norms[i * 2](video_feat + enhanced_video)
            audio_feat = self.layer_norms[i * 2](audio_feat + enhanced_audio)
            physiological_feat = self.layer_norms[i * 2](physiological_feat + enhanced_physiological)
            
            # 前馈网络
            video_feat = self.layer_norms[i * 2 + 1](video_feat + self.ffn_layers[i](video_feat))
            audio_feat = self.layer_norms[i * 2 + 1](audio_feat + self.ffn_layers[i](audio_feat))
            physiological_feat = self.layer_norms[i * 2 + 1](physiological_feat + self.ffn_layers[i](physiological_feat))
            text_feat = self.layer_norms[i * 2 + 1](text_feat + self.ffn_layers[i](text_feat))
        
        # 最终融合
        combined = torch.cat([video_feat, audio_feat, physiological_feat, text_feat], dim=-1)
        fused = self.final_fusion(combined)
        
        if not is_temporal:
            fused = fused.squeeze(1)
        
        return fused

