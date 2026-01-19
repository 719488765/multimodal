"""
多模态功能最大相关模块
借鉴自：Multimodal Functional Maximum Correlation for Emotion Recognition
论文链接：https://arxiv.org/abs/2512.23076
源码地址：https://github.com/DY9910/MFMC

核心思想：
- 通过功能最大相关（FMC）方法捕捉多模态之间的高阶相关性
- 最大化多模态依赖性，提高情感识别准确性
- 可用于自监督预训练阶段
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FunctionalMaximumCorrelation(nn.Module):
    """
    功能最大相关模块
    通过最大化不同模态之间的相关性来学习更好的多模态表示
    """
    def __init__(self, hidden_dim=512, num_projections=64, dropout=0.1):
        super(FunctionalMaximumCorrelation, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_projections = num_projections
        
        # 为每个模态创建投影网络（用于计算相关性）
        self.projections = nn.ModuleDict({
            'video': nn.Sequential(
                nn.Linear(hidden_dim, num_projections),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'audio': nn.Sequential(
                nn.Linear(hidden_dim, num_projections),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'physiological': nn.Sequential(
                nn.Linear(hidden_dim, num_projections),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'text': nn.Sequential(
                nn.Linear(hidden_dim, num_projections),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        
    def compute_correlation(self, feat1, feat2):
        """
        计算两个特征之间的相关性
        Args:
            feat1: (B, num_projections)
            feat2: (B, num_projections)
        Returns:
            correlation: 标量，两个特征的相关性
        """
        # 中心化
        feat1_centered = feat1 - feat1.mean(dim=0, keepdim=True)
        feat2_centered = feat2 - feat2.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵
        cov = torch.matmul(feat1_centered.t(), feat2_centered) / (feat1.shape[0] - 1)
        
        # 计算相关性（使用Frobenius范数）
        correlation = torch.norm(cov, p='fro')
        
        return correlation
    
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat):
        """
        计算多模态之间的功能最大相关
        Args:
            video_feat: (B, hidden_dim)
            audio_feat: (B, hidden_dim)
            physiological_feat: (B, hidden_dim)
            text_feat: (B, hidden_dim)
        Returns:
            correlation_loss: 标量，用于最大化相关性的损失（负相关性）
            correlations: dict，各模态对之间的相关性
        """
        # 投影到低维空间
        proj_video = self.projections['video'](video_feat)
        proj_audio = self.projections['audio'](audio_feat)
        proj_physiological = self.projections['physiological'](physiological_feat)
        proj_text = self.projections['text'](text_feat)
        
        # 计算所有模态对之间的相关性
        correlations = {}
        
        # 视频-音频
        corr_va = self.compute_correlation(proj_video, proj_audio)
        correlations['video_audio'] = corr_va
        
        # 视频-文本
        corr_vt = self.compute_correlation(proj_video, proj_text)
        correlations['video_text'] = corr_vt
        
        # 音频-文本
        corr_at = self.compute_correlation(proj_audio, proj_text)
        correlations['audio_text'] = corr_at
        
        # 生理-视频
        corr_pv = self.compute_correlation(proj_physiological, proj_video)
        correlations['physiological_video'] = corr_pv
        
        # 生理-音频
        corr_pa = self.compute_correlation(proj_physiological, proj_audio)
        correlations['physiological_audio'] = corr_pa
        
        # 生理-文本
        corr_pt = self.compute_correlation(proj_physiological, proj_text)
        correlations['physiological_text'] = corr_pt
        
        # 总相关性（用于损失函数，需要最大化，所以返回负值）
        total_correlation = corr_va + corr_vt + corr_at + corr_pv + corr_pa + corr_pt
        correlation_loss = -total_correlation  # 负相关性，用于最小化损失
        
        return correlation_loss, correlations


class MultimodalCorrelationLoss(nn.Module):
    """
    多模态相关性损失函数
    用于预训练阶段，最大化多模态之间的相关性
    """
    def __init__(self, hidden_dim=512, num_projections=64, weight=0.1):
        super(MultimodalCorrelationLoss, self).__init__()
        
        self.fmc_module = FunctionalMaximumCorrelation(hidden_dim, num_projections)
        self.weight = weight
        
    def forward(self, video_feat, audio_feat, physiological_feat, text_feat):
        """
        计算相关性损失
        Args:
            video_feat: (B, hidden_dim)
            audio_feat: (B, hidden_dim)
            physiological_feat: (B, hidden_dim)
            text_feat: (B, hidden_dim)
        Returns:
            loss: 标量，相关性损失
            correlations: dict，各模态对之间的相关性
        """
        correlation_loss, correlations = self.fmc_module(
            video_feat, audio_feat, physiological_feat, text_feat
        )
        
        return self.weight * correlation_loss, correlations

