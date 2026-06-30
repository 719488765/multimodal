#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域适应模块
用于处理混合数据集训练中的域偏移问题
实现域分类器和域对抗训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层（Gradient Reversal Layer）
    
    在前向传播时保持输入不变，在反向传播时反转梯度
    用于域对抗训练
    """
    
    @staticmethod
    def forward(ctx, x, lambda_param=1.0):
        """
        前向传播：直接返回输入
        
        Args:
            x: 输入tensor
            lambda_param: 梯度反转的强度参数
        """
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：反转梯度
        
        Args:
            grad_output: 上游梯度
        
        Returns:
            反转后的梯度
        """
        return grad_output.neg() * ctx.lambda_param, None


class DomainClassifier(nn.Module):
    """
    域分类器
    
    用于识别样本来自哪个数据集（域）
    与主任务分类器对抗训练，迫使特征提取器学习域不变特征
    """
    
    def __init__(self, input_dim, num_domains=3, hidden_dim=256, dropout=0.1):
        """
        初始化域分类器
        
        Args:
            input_dim: 输入特征维度
            num_domains: 域数量（数据集数量）
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(DomainClassifier, self).__init__()
        
        self.num_domains = num_domains
        
        # 域分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, features, lambda_param=1.0):
        """
        前向传播
        
        Args:
            features: 输入特征 (B, input_dim)
            lambda_param: 梯度反转强度参数
        
        Returns:
            domain_logits: 域分类logits (B, num_domains)
        """
        # 应用梯度反转层
        reversed_features = GradientReversalLayer.apply(features, lambda_param)
        
        # 域分类
        domain_logits = self.classifier(reversed_features)
        
        return domain_logits


class DatasetSpecificNormalization(nn.Module):
    """
    数据集特定归一化层
    
    为每个数据集维护独立的归一化统计（均值和方差）
    在特征提取后应用数据集特定的归一化，减少域间的特征分布差异
    """
    
    def __init__(self, feature_dim, num_datasets=3, momentum=0.1):
        """
        初始化数据集特定归一化
        
        Args:
            feature_dim: 特征维度
            num_datasets: 数据集数量
            momentum: 移动平均的动量参数
        """
        super(DatasetSpecificNormalization, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_datasets = num_datasets
        self.momentum = momentum
        
        # 为每个数据集维护独立的均值和方差
        # 使用register_buffer确保这些参数会被移动到正确的设备
        for i in range(num_datasets):
            self.register_buffer(f'mean_{i}', torch.zeros(feature_dim))
            self.register_buffer(f'var_{i}', torch.ones(feature_dim))
            self.register_buffer(f'count_{i}', torch.tensor(0))
        
        # 未知数据集的默认统计
        self.register_buffer('mean_default', torch.zeros(feature_dim))
        self.register_buffer('var_default', torch.ones(feature_dim))
        self.register_buffer('count_default', torch.tensor(0))
    
    def update_statistics(self, features, dataset_ids):
        """
        更新数据集特定的统计信息（用于训练阶段）
        
        Args:
            features: 特征tensor (B, feature_dim)
            dataset_ids: 数据集ID (B,)
        """
        if not self.training:
            return
        
        # 按数据集分组
        for dataset_id in range(self.num_datasets):
            mask = (dataset_ids == dataset_id)
            if mask.sum() > 0:
                dataset_features = features[mask]
                
                # 计算当前batch的统计信息
                batch_mean = dataset_features.mean(dim=0)
                batch_var = dataset_features.var(dim=0, unbiased=False)
                batch_count = mask.sum().float()
                
                # 获取当前统计信息
                current_mean = getattr(self, f'mean_{dataset_id}')
                current_var = getattr(self, f'var_{dataset_id}')
                current_count = getattr(self, f'count_{dataset_id}')
                
                # 更新统计信息（移动平均）
                if current_count > 0:
                    total_count = current_count + batch_count
                    new_mean = (current_mean * current_count + batch_mean * batch_count) / total_count
                    new_var = (current_var * current_count + batch_var * batch_count) / total_count
                else:
                    new_mean = batch_mean
                    new_var = batch_var
                    total_count = batch_count
                
                # 更新buffer
                setattr(self, f'mean_{dataset_id}', new_mean)
                setattr(self, f'var_{dataset_id}', new_var)
                setattr(self, f'count_{dataset_id}', total_count)
        
        # 处理未知数据集
        unknown_mask = (dataset_ids < 0) | (dataset_ids >= self.num_datasets)
        if unknown_mask.sum() > 0:
            unknown_features = features[unknown_mask]
            batch_mean = unknown_features.mean(dim=0)
            batch_var = unknown_features.var(dim=0, unbiased=False)
            batch_count = unknown_mask.sum().float()
            
            if self.count_default > 0:
                total_count = self.count_default + batch_count
                new_mean = (self.mean_default * self.count_default + batch_mean * batch_count) / total_count
                new_var = (self.var_default * self.count_default + batch_var * batch_count) / total_count
            else:
                new_mean = batch_mean
                new_var = batch_var
                total_count = batch_count
            
            self.mean_default = new_mean
            self.var_default = new_var
            self.count_default = total_count
    
    def forward(self, features, dataset_ids=None):
        """
        应用数据集特定归一化
        
        Args:
            features: 输入特征 (B, feature_dim)
            dataset_ids: 数据集ID (B,)，如果为None则不进行归一化
        
        Returns:
            normalized_features: 归一化后的特征 (B, feature_dim)
        """
        # 基本数值防护，避免 NaN/Inf 进入归一化计算
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        if dataset_ids is None:
            # 没有数据集信息，使用标准归一化（基于当前 batch）
            mean = features.mean(dim=0, keepdim=True)
            var = features.var(dim=0, keepdim=True, unbiased=False)
            # 防止方差为 0 或出现 NaN/Inf
            var = torch.nan_to_num(var, nan=1.0, posinf=1e4, neginf=1e-4)
            var = var.clamp_min(1e-6)
            normalized = (features - mean) / (var.sqrt() + 1e-8)
            return normalized
        
        # 为每个样本应用对应的归一化
        normalized_features = torch.zeros_like(features)
        
        for dataset_id in range(self.num_datasets):
            mask = (dataset_ids == dataset_id)
            if mask.sum() > 0:
                dataset_features = features[mask]
                mean = getattr(self, f'mean_{dataset_id}')
                var = getattr(self, f'var_{dataset_id}')
                # 防止统计量出现 NaN/Inf 或 0 方差
                mean = torch.nan_to_num(mean, nan=0.0, posinf=1e4, neginf=-1e4)
                var = torch.nan_to_num(var, nan=1.0, posinf=1e4, neginf=1e-4)
                var = var.clamp_min(1e-6)

                normalized = (dataset_features - mean) / (var.sqrt() + 1e-8)
                normalized_features[mask] = normalized
        
        # 处理未知数据集
        unknown_mask = (dataset_ids < 0) | (dataset_ids >= self.num_datasets)
        if unknown_mask.sum() > 0:
            unknown_features = features[unknown_mask]
            mean = torch.nan_to_num(self.mean_default, nan=0.0, posinf=1e4, neginf=-1e4)
            var = torch.nan_to_num(self.var_default, nan=1.0, posinf=1e4, neginf=1e-4)
            var = var.clamp_min(1e-6)

            normalized = (unknown_features - mean) / (var.sqrt() + 1e-8)
            normalized_features[unknown_mask] = normalized
        
        return normalized_features


class DomainAdversarialModule(nn.Module):
    """
    域对抗训练模块
    
    整合域分类器和梯度反转层，实现域对抗训练
    """
    
    def __init__(self, feature_dim, num_domains=3, hidden_dim=256, 
                 lambda_param=1.0, adaptive_lambda=False):
        """
        初始化域对抗模块
        
        Args:
            feature_dim: 特征维度
            num_domains: 域数量
            hidden_dim: 隐藏层维度
            lambda_param: 梯度反转强度参数（初始值）
            adaptive_lambda: 是否使用自适应lambda（随训练进度调整）
        """
        super(DomainAdversarialModule, self).__init__()
        
        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim,
            num_domains=num_domains,
            hidden_dim=hidden_dim
        )
        
        self.lambda_param = lambda_param
        self.adaptive_lambda = adaptive_lambda
        self.current_epoch = 0
        self.total_epochs = 100  # 默认值，可以在训练时更新
    
    def set_epoch(self, epoch, total_epochs=None):
        """
        设置当前epoch（用于自适应lambda）
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def get_lambda(self):
        """
        获取当前的lambda参数
        
        Returns:
            lambda_param: 梯度反转强度参数
        """
        if self.adaptive_lambda:
            # 自适应lambda：从0逐渐增加到lambda_param
            # 使用sigmoid函数平滑过渡
            progress = float(self.current_epoch) / float(self.total_epochs if self.total_epochs > 0 else 1)
            # 这里使用math.exp避免将标量强制转换为Tensor
            lambda_val = self.lambda_param * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            return float(lambda_val)
        else:
            return self.lambda_param
    
    def forward(self, features, dataset_ids):
        """
        前向传播
        
        Args:
            features: 输入特征 (B, feature_dim)
            dataset_ids: 数据集ID (B,)
        
        Returns:
            domain_logits: 域分类logits (B, num_domains)
        """
        lambda_param = self.get_lambda()
        domain_logits = self.domain_classifier(features, lambda_param)
        return domain_logits
    
    def compute_domain_loss(self, domain_logits, dataset_ids):
        """
        计算域分类损失
        
        Args:
            domain_logits: 域分类logits (B, num_domains)
            dataset_ids: 真实数据集ID (B,)
        
        Returns:
            domain_loss: 域分类损失
        """
        # 将dataset_ids转换为long类型
        if isinstance(dataset_ids, torch.Tensor):
            dataset_ids = dataset_ids.long()
        else:
            dataset_ids = torch.tensor(dataset_ids, dtype=torch.long)
        
        # 确保dataset_ids在有效范围内
        dataset_ids = torch.clamp(dataset_ids, 0, self.domain_classifier.num_domains - 1)
        
        # 计算交叉熵损失
        domain_loss = F.cross_entropy(domain_logits, dataset_ids)
        
        return domain_loss

