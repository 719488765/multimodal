#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合训练策略
用于优化混合数据集训练流程，减少域混淆
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict
import random


class AlternatingTrainingStrategy:
    """
    交替训练策略
    
    每个epoch交替使用不同数据集，确保模型充分学习每个数据集的特征
    """
    
    def __init__(self, datasets_dict, batch_size, num_workers=4):
        """
        初始化交替训练策略
        
        Args:
            datasets_dict: 数据集字典 {dataset_name: dataset}
            batch_size: batch大小
            num_workers: 数据加载器的工作进程数
        """
        self.datasets_dict = datasets_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_names = list(datasets_dict.keys())
        self.current_dataset_idx = 0
    
    def get_dataloader_for_epoch(self, epoch):
        """
        获取当前epoch使用的数据加载器
        
        Args:
            epoch: 当前epoch
        
        Returns:
            dataloader: 当前epoch的数据加载器
        """
        # 交替选择数据集
        dataset_name = self.dataset_names[epoch % len(self.dataset_names)]
        dataset = self.datasets_dict[dataset_name]
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        return dataloader, dataset_name


class ProgressiveTrainingStrategy:
    """
    渐进式训练策略
    
    先单数据集训练，再混合训练
    逐步增加数据集的多样性
    """
    
    def __init__(self, datasets_dict, batch_size, num_workers=4, 
                 single_epochs=5, mixed_epochs=45):
        """
        初始化渐进式训练策略
        
        Args:
            datasets_dict: 数据集字典 {dataset_name: dataset}
            batch_size: batch大小
            num_workers: 数据加载器的工作进程数
            single_epochs: 单数据集训练的epoch数
            mixed_epochs: 混合训练的epoch数
        """
        self.datasets_dict = datasets_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.single_epochs = single_epochs
        self.mixed_epochs = mixed_epochs
        self.dataset_names = list(datasets_dict.keys())
        
        # 创建混合数据集
        self.mixed_dataset = ConcatDataset(list(datasets_dict.values()))
        self.mixed_dataloader = DataLoader(
            self.mixed_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    
    def get_dataloader_for_epoch(self, epoch):
        """
        获取当前epoch使用的数据加载器
        
        Args:
            epoch: 当前epoch
        
        Returns:
            dataloader: 当前epoch的数据加载器
            dataset_name: 数据集名称（单数据集训练时）或"mixed"（混合训练时）
        """
        if epoch < self.single_epochs:
            # 单数据集训练阶段：每个数据集训练几个epoch
            epochs_per_dataset = self.single_epochs // len(self.dataset_names)
            dataset_idx = epoch // epochs_per_dataset
            if dataset_idx >= len(self.dataset_names):
                dataset_idx = len(self.dataset_names) - 1
            
            dataset_name = self.dataset_names[dataset_idx]
            dataset = self.datasets_dict[dataset_name]
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            
            return dataloader, dataset_name
        else:
            # 混合训练阶段
            return self.mixed_dataloader, "mixed"


class CurriculumLearningStrategy:
    """
    课程学习策略
    
    从简单数据集到复杂数据集，逐步增加训练难度
    """
    
    def __init__(self, datasets_dict, batch_size, num_workers=4,
                 difficulty_order=None):
        """
        初始化课程学习策略
        
        Args:
            datasets_dict: 数据集字典 {dataset_name: dataset}
            batch_size: batch大小
            num_workers: 数据加载器的工作进程数
            difficulty_order: 数据集难度顺序列表，None表示自动排序
        """
        self.datasets_dict = datasets_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 确定数据集难度顺序
        if difficulty_order is None:
            # 默认顺序：CREMA-D（简单）-> MELD（中等）-> CMU-MOSEI（复杂）
            self.difficulty_order = ['crema', 'meld', 'mosei']
        else:
            self.difficulty_order = difficulty_order
        
        # 创建渐进式数据集（逐步添加数据集）
        self.progressive_datasets = []
        for i in range(len(self.difficulty_order)):
            dataset_names = self.difficulty_order[:i+1]
            datasets = [datasets_dict[name] for name in dataset_names if name in datasets_dict]
            if datasets:
                self.progressive_datasets.append(ConcatDataset(datasets))
    
    def get_dataloader_for_epoch(self, epoch, total_epochs):
        """
        获取当前epoch使用的数据加载器
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        
        Returns:
            dataloader: 当前epoch的数据加载器
            dataset_name: 数据集名称
        """
        # 根据训练进度选择数据集组合
        progress = epoch / total_epochs
        dataset_idx = int(progress * len(self.progressive_datasets))
        if dataset_idx >= len(self.progressive_datasets):
            dataset_idx = len(self.progressive_datasets) - 1
        
        dataset = self.progressive_datasets[dataset_idx]
        dataset_names = self.difficulty_order[:dataset_idx+1]
        dataset_name = "+".join(dataset_names)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        return dataloader, dataset_name


def create_training_strategy(strategy_type, datasets_dict, batch_size, 
                            num_workers=4, **kwargs):
    """
    创建训练策略
    
    Args:
        strategy_type: 策略类型 ('alternating', 'progressive', 'curriculum', 'standard')
        datasets_dict: 数据集字典
        batch_size: batch大小
        num_workers: 数据加载器的工作进程数
        **kwargs: 策略特定的参数
    
    Returns:
        strategy: 训练策略对象
    """
    if strategy_type == 'alternating':
        return AlternatingTrainingStrategy(datasets_dict, batch_size, num_workers)
    elif strategy_type == 'progressive':
        return ProgressiveTrainingStrategy(
            datasets_dict, batch_size, num_workers,
            single_epochs=kwargs.get('single_epochs', 5),
            mixed_epochs=kwargs.get('mixed_epochs', 45)
        )
    elif strategy_type == 'curriculum':
        return CurriculumLearningStrategy(
            datasets_dict, batch_size, num_workers,
            difficulty_order=kwargs.get('difficulty_order', None)
        )
    elif strategy_type == 'standard':
        # 标准策略：直接使用混合数据集
        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(list(datasets_dict.values()))
        return None  # 返回None表示使用标准DataLoader
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

