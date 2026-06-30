#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
类别平衡损失函数
用于处理混合数据集训练中的类别不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失
    
    根据类别频率计算权重，对少数类别给予更高权重
    支持每个数据集独立的类别权重计算
    
    参考论文：Class-Balanced Loss Based on Effective Number of Samples
    """
    
    def __init__(self, num_classes=7, beta=0.9999, reduction='mean', label_smoothing=0.0):
        """
        初始化类别平衡损失
        
        Args:
            num_classes: 情感类别数
            beta: 有效样本数的超参数（0 < beta < 1），beta越大，少数类别权重越高
            reduction: 损失归约方式（'mean'或'sum'）
            label_smoothing: 标签平滑（与 F.cross_entropy 一致，再乘类别权重）
        """
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.class_weights = None
        # 由训练脚本根据全训练集标签预计算后注入，避免按 batch 重算导致验证尺度抖动
        self.fixed_class_weights = None
    
    def compute_class_weights(self, labels):
        """
        根据标签计算类别权重
        
        Args:
            labels: 标签tensor (N,)
        
        Returns:
            weights: 类别权重tensor (num_classes,)
        """
        # 统计每个类别的样本数，并忽略超出 [0, num_classes-1] 范围的标签
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        # 仅保留有效类别索引，避免预训练阶段的额外类别（例如 label=6）在
        # 微调阶段（num_classes=6）导致越界访问。
        valid_mask = (labels_np >= 0) & (labels_np < self.num_classes)
        labels_np = labels_np[valid_mask]
        
        class_counts = Counter(labels_np)
        total_samples = len(labels_np)
        
        # 计算有效样本数：E_n = (1 - beta^n) / (1 - beta)
        # 其中n是类别样本数
        weights = torch.ones(self.num_classes, dtype=torch.float32)
        
        for class_id, count in class_counts.items():
            if count > 0:
                # 有效样本数
                effective_num = (1.0 - self.beta ** count) / (1.0 - self.beta)
                # 权重 = 总样本数 / 有效样本数
                weights[class_id] = total_samples / effective_num
        
        # 归一化权重
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def forward(self, logits, labels, class_weights=None):
        """
        计算类别平衡损失
        
        Args:
            logits: 模型输出logits (N, num_classes)
            labels: 真实标签 (N,)
            class_weights: 预计算的类别权重 (num_classes,)，如果为None则自动计算
        
        Returns:
            loss: 类别平衡损失
        """
        # 保证标签在有效类别范围内：0 <= label < num_classes
        num_classes = logits.size(1)
        if isinstance(labels, torch.Tensor):
            valid_mask = (labels >= 0) & (labels < num_classes)
        else:
            labels = torch.as_tensor(labels, device=logits.device)
            valid_mask = (labels >= 0) & (labels < num_classes)

        if valid_mask.sum() == 0:
            # 本 batch 没有有效标签，返回 0 损失（不影响梯度累计）
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        labels = labels[valid_mask]

        if class_weights is None:
            if self.fixed_class_weights is not None:
                class_weights = self.fixed_class_weights
            else:
                # 自动计算类别权重（内部已再次过滤越界标签）
                class_weights = self.compute_class_weights(labels)
        
        # 将权重移到与logits相同的设备
        if isinstance(class_weights, torch.Tensor):
            class_weights = class_weights.to(logits.device)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(logits.device)
        
        # 计算交叉熵损失（可选 label_smoothing）
        ce_loss = F.cross_entropy(
            logits, labels, reduction='none', label_smoothing=self.label_smoothing
        )
        
        # 应用类别权重
        weights = class_weights[labels]
        weighted_loss = ce_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    关注难分类样本，自动调整难易样本的权重
    参考论文：Focal Loss for Dense Object Detection
    
    公式：FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    其中p_t是预测概率，alpha和gamma是超参数
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重（可以是标量或tensor）
            gamma: 聚焦参数，gamma越大，对难样本的关注度越高
            reduction: 损失归约方式
            label_smoothing: 传入底层 cross_entropy
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
    
    def forward(self, logits, labels):
        """
        计算Focal Loss
        
        Args:
            logits: 模型输出logits (N, num_classes)
            labels: 真实标签 (N,)
        
        Returns:
            loss: Focal Loss
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            logits, labels, reduction='none', label_smoothing=self.label_smoothing
        )
        
        ce_loss = torch.clamp(ce_loss, max=50.0)
        # 计算预测概率（clamp 防止 MOSEI 长序列下 exp 溢出 → NaN）
        pt = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0 - 1e-8)
        
        # 计算focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用alpha权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            # alpha是tensor，根据标签选择对应的alpha
            alpha_t = self.alpha[labels]
            if isinstance(alpha_t, torch.Tensor):
                alpha_t = alpha_t.to(logits.device)
        
        # 计算focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DatasetWeightedLoss(nn.Module):
    """
    数据集加权损失
    
    为不同数据集分配不同的损失权重
    可以根据数据集质量、标注可靠性等因素调整权重
    """
    
    def __init__(self, base_loss_fn, dataset_weights=None):
        """
        初始化数据集加权损失
        
        Args:
            base_loss_fn: 基础损失函数（如CrossEntropyLoss）
            dataset_weights: 数据集权重字典 {dataset_id: weight}
                dataset_id: 0=CREMA-D, 1=MELD, 2=CMU-MOSEI
        """
        super(DatasetWeightedLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.dataset_weights = dataset_weights or {0: 1.0, 1: 1.0, 2: 1.0, -1: 1.0}
    
    def forward(self, logits, labels, dataset_ids=None):
        """
        计算数据集加权损失
        
        Args:
            logits: 模型输出logits (N, num_classes)
            labels: 真实标签 (N,)
            dataset_ids: 数据集ID (N,)，如果为None则使用统一权重
        
        Returns:
            loss: 数据集加权损失
        """
        # 计算基础损失
        if dataset_ids is None:
            # 没有数据集信息，使用基础损失
            return self.base_loss_fn(logits, labels)
        
        # 计算每个样本的损失
        if hasattr(self.base_loss_fn, 'forward'):
            # 如果损失函数支持reduction='none'
            try:
                per_sample_loss = self.base_loss_fn(logits, labels, reduction='none')
            except TypeError:
                # 如果不支持reduction参数，手动计算
                ce_loss = F.cross_entropy(logits, labels, reduction='none')
                per_sample_loss = ce_loss
        else:
            # 使用交叉熵
            per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 应用数据集权重
        if isinstance(dataset_ids, torch.Tensor):
            dataset_ids_np = dataset_ids.cpu().numpy()
        else:
            dataset_ids_np = np.array(dataset_ids)
        
        weights = torch.ones_like(per_sample_loss)
        for i, dataset_id in enumerate(dataset_ids_np):
            weight = self.dataset_weights.get(int(dataset_id), 1.0)
            weights[i] = weight
        
        # 将权重移到正确的设备
        weights = weights.to(per_sample_loss.device)
        
        # 加权损失
        weighted_loss = per_sample_loss * weights
        
        return weighted_loss.mean()


class MultiDatasetBalancedLoss(nn.Module):
    """
    多数据集平衡损失
    
    结合类别平衡损失和数据集加权损失
    同时处理类别不平衡和数据集差异问题
    """
    
    def __init__(self, num_classes=7, class_balance_beta=0.9999, 
                 use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0,
                 dataset_weights=None):
        """
        初始化多数据集平衡损失
        
        Args:
            num_classes: 情感类别数
            class_balance_beta: 类别平衡的beta参数
            use_focal_loss: 是否使用Focal Loss
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            dataset_weights: 数据集权重字典
        """
        super(MultiDatasetBalancedLoss, self).__init__()
        
        # 选择基础损失函数
        if use_focal_loss:
            self.base_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.base_loss = ClassBalancedLoss(
                num_classes=num_classes, 
                beta=class_balance_beta
            )
        
        # 数据集加权
        self.dataset_weighted_loss = DatasetWeightedLoss(
            base_loss_fn=self.base_loss,
            dataset_weights=dataset_weights
        )
        
        self.num_classes = num_classes
    
    def forward(self, logits, labels, dataset_ids=None, class_weights=None):
        """
        计算多数据集平衡损失
        
        Args:
            logits: 模型输出logits (N, num_classes)
            labels: 真实标签 (N,)
            dataset_ids: 数据集ID (N,)
            class_weights: 预计算的类别权重
        
        Returns:
            loss: 多数据集平衡损失
        """
        if dataset_ids is not None:
            # 使用数据集加权损失
            return self.dataset_weighted_loss(logits, labels, dataset_ids)
        else:
            # 只使用类别平衡损失
            if isinstance(self.base_loss, ClassBalancedLoss):
                return self.base_loss(logits, labels, class_weights)
            else:
                return self.base_loss(logits, labels)

