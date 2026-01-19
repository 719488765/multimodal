"""
领导-跟随注意力机制
借鉴自：Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion
论文链接：https://arxiv.org/abs/2107.01175
会议：ICCV 2021（CCF-A类）
源码地址：https://github.com/sucv/ABAW2

核心思想：
- 一个模态作为"领导者"，引导另一个模态的特征学习
- 允许模态之间相互引导，提高融合效果
- 适用于连续情感识别任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeaderFollowerAttention(nn.Module):
    """
    领导-跟随注意力机制
    一个模态（领导者）引导另一个模态（跟随者）的特征学习
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(LeaderFollowerAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 领导模态的查询投影
        self.leader_q = nn.Linear(hidden_dim, hidden_dim)
        
        # 跟随模态的键值投影
        self.follower_k = nn.Linear(hidden_dim, hidden_dim)
        self.follower_v = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, leader_feat, follower_feat, mask=None):
        """
        Args:
            leader_feat: (B, T_l, hidden_dim) - 领导模态特征
            follower_feat: (B, T_f, hidden_dim) - 跟随模态特征
            mask: (B, T_l, T_f) - 注意力掩码
        Returns:
            enhanced_follower: (B, T_f, hidden_dim) - 增强后的跟随模态特征
            attention_weights: (B, num_heads, T_f, T_l) - 注意力权重
        """
        B, T_l, _ = leader_feat.shape
        T_f = follower_feat.shape[1]
        
        # 投影
        Q = self.leader_q(leader_feat).view(B, T_l, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_l, head_dim)
        K = self.follower_k(follower_feat).view(B, T_f, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_f, head_dim)
        V = self.follower_v(follower_feat).view(B, T_f, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T_f, head_dim)
        
        # 计算注意力分数（领导模态引导跟随模态）
        # Q来自领导模态，K和V来自跟随模态
        scores = torch.matmul(K, Q.transpose(-2, -1)) / self.scale  # (B, num_heads, T_f, T_l)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T_f, T_l)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax和加权求和
        attention_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T_f, T_l)
        attention_weights = self.dropout(attention_weights)
        
        # 使用注意力权重加权领导模态的信息来增强跟随模态
        # 这里我们使用领导模态的信息来增强跟随模态
        enhanced = torch.matmul(attention_weights, Q)  # (B, num_heads, T_f, head_dim)
        enhanced = enhanced.transpose(1, 2).contiguous().view(B, T_f, self.hidden_dim)  # (B, T_f, hidden_dim)
        
        # 与原始跟随模态特征融合
        enhanced_follower = follower_feat + enhanced
        enhanced_follower = self.out_proj(enhanced_follower)
        
        return enhanced_follower, attention_weights


class BidirectionalLeaderFollower(nn.Module):
    """
    双向领导-跟随注意力
    允许两个模态相互引导
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(BidirectionalLeaderFollower, self).__init__()
        
        # 模态1引导模态2
        self.lf_12 = LeaderFollowerAttention(hidden_dim, num_heads, dropout)
        
        # 模态2引导模态1
        self.lf_21 = LeaderFollowerAttention(hidden_dim, num_heads, dropout)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, feat1, feat2, mask=None):
        """
        Args:
            feat1: (B, T1, hidden_dim) - 模态1特征
            feat2: (B, T2, hidden_dim) - 模态2特征
            mask: (B, T1, T2) - 注意力掩码
        Returns:
            fused_feat1: (B, T1, hidden_dim) - 增强后的模态1特征
            fused_feat2: (B, T2, hidden_dim) - 增强后的模态2特征
            combined: (B, max(T1, T2), hidden_dim) - 融合后的特征
        """
        # 模态1引导模态2
        enhanced_feat2, attn_12 = self.lf_12(feat1, feat2, mask)
        
        # 模态2引导模态1
        if mask is not None:
            mask_reverse = mask.transpose(-2, -1)
        else:
            mask_reverse = None
        enhanced_feat1, attn_21 = self.lf_21(feat2, feat1, mask_reverse)
        
        # 对齐时间维度并融合
        T1, T2 = feat1.shape[1], feat2.shape[1]
        if T1 == T2:
            combined = torch.cat([enhanced_feat1, enhanced_feat2], dim=-1)
            combined = self.fusion(combined)
            combined = self.layer_norm(combined)
        else:
            # 如果时间维度不同，使用平均池化或插值
            if T1 > T2:
                # 对feat2进行插值
                enhanced_feat2 = F.interpolate(
                    enhanced_feat2.transpose(1, 2),
                    size=T1,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                # 对feat1进行插值
                enhanced_feat1 = F.interpolate(
                    enhanced_feat1.transpose(1, 2),
                    size=T2,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            combined = torch.cat([enhanced_feat1, enhanced_feat2], dim=-1)
            combined = self.fusion(combined)
            combined = self.layer_norm(combined)
        
        return enhanced_feat1, enhanced_feat2, combined


class MultimodalLeaderFollowerFusion(nn.Module):
    """
    多模态领导-跟随融合
    扩展领导-跟随注意力到多个模态
    在智能驾驶场景中，可以设置文本或生理信号为领导者
    """
    def __init__(self, hidden_dim=512, num_heads=8, leader_modal='text', dropout=0.1):
        super(MultimodalLeaderFollowerFusion, self).__init__()
        
        self.leader_modal = leader_modal
        self.hidden_dim = hidden_dim
        
        # 为每个跟随模态创建领导-跟随注意力
        self.lf_modules = nn.ModuleDict({
            'video': LeaderFollowerAttention(hidden_dim, num_heads, dropout),
            'audio': LeaderFollowerAttention(hidden_dim, num_heads, dropout),
            'physiological': LeaderFollowerAttention(hidden_dim, num_heads, dropout)
        })
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 4个模态
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat):
        """
        Args:
            video_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            audio_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            physiological_feat: (B, hidden_dim) 或 (B, T, hidden_dim)
            text_feat: (B, hidden_dim) 或 (B, T, hidden_dim) - 默认作为领导者
        Returns:
            fused_features: (B, hidden_dim) 或 (B, T, hidden_dim)
        """
        is_temporal = len(text_feat.shape) == 3
        
        if not is_temporal:
            # 扩展到时序维度
            text_feat = text_feat.unsqueeze(1)  # (B, 1, hidden_dim)
            video_feat = video_feat.unsqueeze(1)
            audio_feat = audio_feat.unsqueeze(1)
            physiological_feat = physiological_feat.unsqueeze(1)
        
        # 使用文本作为领导者，引导其他模态
        enhanced_video, _ = self.lf_modules['video'](text_feat, video_feat)
        enhanced_audio, _ = self.lf_modules['audio'](text_feat, audio_feat)
        enhanced_physiological, _ = self.lf_modules['physiological'](text_feat, physiological_feat)
        
        # 融合所有模态
        combined = torch.cat([text_feat, enhanced_video, enhanced_audio, enhanced_physiological], dim=-1)
        fused = self.final_fusion(combined)
        fused = self.layer_norm(fused)
        
        if not is_temporal:
            fused = fused.squeeze(1)
        
        return fused

