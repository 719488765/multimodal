#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集平衡采样器
用于混合数据集训练，确保每个batch包含来自不同数据集的样本
"""

import torch
from torch.utils.data import Sampler
from collections import defaultdict
import random


class BalancedDatasetSampler(Sampler):
    """
    平衡数据集采样器
    
    功能：
    1. 确保每个batch包含来自不同数据集的样本
    2. 根据数据集大小调整采样权重
    3. 支持按比例采样和均匀采样两种模式
    
    使用示例：
        >>> sampler = BalancedDatasetSampler(dataset, batch_size=16, mode='proportional')
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(self, dataset, batch_size, mode='proportional', shuffle=True, seed=None):
        """
        初始化平衡采样器
        
        Args:
            dataset: 数据集对象，必须返回包含'dataset_id'的样本
            batch_size: batch大小
            mode: 采样模式
                - 'proportional': 按数据集大小比例采样
                - 'uniform': 均匀采样，每个数据集样本数相同
            shuffle: 是否打乱顺序
            seed: 随机种子
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # 按数据集ID分组样本索引
        self.dataset_indices = defaultdict(list)
        # 优先使用数据集元信息，避免初始化时触发 __getitem__ 导致音视频重载
        if hasattr(dataset, "data_list"):
            for idx, sample in enumerate(dataset.data_list):
                dataset_id = sample.get("dataset_id", -1)
                self.dataset_indices[dataset_id].append(idx)
        elif hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "data_list"):
            # 兼容 torch.utils.data.Subset：将 subset 索引映射回 base dataset 元信息
            for subset_idx, base_idx in enumerate(dataset.indices):
                sample = dataset.dataset.data_list[base_idx]
                dataset_id = sample.get("dataset_id", -1)
                self.dataset_indices[dataset_id].append(subset_idx)
        else:
            # 回退到逐条访问 __getitem__ 的方式（可能会较慢）
            for idx in range(len(dataset)):
                sample = dataset[idx]
                dataset_id = sample.get("dataset_id", -1)
                self.dataset_indices[dataset_id].append(idx)
        
        # 计算每个数据集的样本数
        self.dataset_sizes = {did: len(indices) for did, indices in self.dataset_indices.items()}
        total_samples = sum(self.dataset_sizes.values())
        
        # 计算采样权重
        if mode == 'proportional':
            # 按数据集大小比例分配权重
            self.dataset_weights = {
                did: size / total_samples 
                for did, size in self.dataset_sizes.items()
            }
        elif mode == 'uniform':
            # 均匀分配权重
            num_datasets = len(self.dataset_sizes)
            self.dataset_weights = {
                did: 1.0 / num_datasets 
                for did in self.dataset_sizes.keys()
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'proportional' or 'uniform'")
        
        # 计算每个batch中每个数据集的样本数
        self.batch_distribution = self._compute_batch_distribution()
        
        # 生成batch索引
        self.batches = self._generate_batches()
    
    def _compute_batch_distribution(self):
        """
        计算每个batch中每个数据集的样本数
        
        Returns:
            dict: {dataset_id: samples_per_batch}
        """
        distribution = {}
        for dataset_id, weight in self.dataset_weights.items():
            samples_per_batch = max(1, int(self.batch_size * weight))
            distribution[dataset_id] = samples_per_batch
        
        # 确保总和不超过batch_size
        total = sum(distribution.values())
        if total > self.batch_size:
            # 按比例缩减
            scale = self.batch_size / total
            distribution = {
                did: max(1, int(count * scale))
                for did, count in distribution.items()
            }
        
        # 如果总和仍小于batch_size，分配给最大的数据集
        total = sum(distribution.values())
        if total < self.batch_size:
            largest_dataset = max(distribution.items(), key=lambda x: x[1])[0]
            distribution[largest_dataset] += (self.batch_size - total)
        
        return distribution
    
    def _generate_batches(self):
        """
        生成batch索引列表
        
        Returns:
            list: 每个元素是一个batch的索引列表
        """
        batches = []
        
        # 为每个数据集创建索引迭代器
        dataset_iterators = {}
        for dataset_id, indices in self.dataset_indices.items():
            if self.shuffle:
                shuffled = indices.copy()
                random.shuffle(shuffled)
            else:
                shuffled = indices
            dataset_iterators[dataset_id] = iter(shuffled)
        
        # 生成batch
        max_batches = sum(self.dataset_sizes.values()) // self.batch_size + 1
        
        for _ in range(max_batches):
            batch_indices = []
            exhausted_datasets = set()
            
            # 从每个数据集采样指定数量的样本
            for dataset_id, count in self.batch_distribution.items():
                if dataset_id in exhausted_datasets:
                    continue
                
                iterator = dataset_iterators[dataset_id]
                for _ in range(count):
                    try:
                        idx = next(iterator)
                        batch_indices.append(idx)
                    except StopIteration:
                        # 数据集用完了，重新创建迭代器
                        indices = self.dataset_indices[dataset_id]
                        if self.shuffle:
                            shuffled = indices.copy()
                            random.shuffle(shuffled)
                        else:
                            shuffled = indices
                        dataset_iterators[dataset_id] = iter(shuffled)
                        
                        try:
                            idx = next(dataset_iterators[dataset_id])
                            batch_indices.append(idx)
                        except StopIteration:
                            exhausted_datasets.add(dataset_id)
                            break
            
            if len(batch_indices) == 0:
                break
            
            # 如果batch不满，从其他数据集补充
            if len(batch_indices) < self.batch_size:
                remaining = self.batch_size - len(batch_indices)
                available_datasets = [
                    did for did in self.dataset_indices.keys() 
                    if did not in exhausted_datasets
                ]
                
                for _ in range(remaining):
                    if not available_datasets:
                        break
                    dataset_id = random.choice(available_datasets)
                    iterator = dataset_iterators[dataset_id]
                    try:
                        idx = next(iterator)
                        batch_indices.append(idx)
                    except StopIteration:
                        available_datasets.remove(dataset_id)
                        if available_datasets:
                            indices = self.dataset_indices[dataset_id]
                            if self.shuffle:
                                shuffled = indices.copy()
                                random.shuffle(shuffled)
                            else:
                                shuffled = indices
                            dataset_iterators[dataset_id] = iter(shuffled)
                            try:
                                idx = next(dataset_iterators[dataset_id])
                                batch_indices.append(idx)
                            except StopIteration:
                                available_datasets.remove(dataset_id)
            
            if len(batch_indices) > 0:
                if self.shuffle:
                    random.shuffle(batch_indices)
                batches.append(batch_indices)
        
        return batches
    
    def __iter__(self):
        """返回batch迭代器"""
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self):
        """返回batch数量"""
        return len(self.batches)
    
    def get_dataset_statistics(self):
        """
        获取数据集统计信息
        
        Returns:
            dict: 包含数据集大小、权重等统计信息
        """
        return {
            'dataset_sizes': self.dataset_sizes,
            'dataset_weights': self.dataset_weights,
            'batch_distribution': self.batch_distribution,
            'total_batches': len(self.batches)
        }


class WeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    加权随机采样器（扩展版本）
    支持数据集级别的加权采样
    """
    
    def __init__(self, dataset, dataset_weights=None, num_samples=None, replacement=True, generator=None):
        """
        初始化加权采样器
        
        Args:
            dataset: 数据集对象
            dataset_weights: 数据集权重字典 {dataset_id: weight}
            num_samples: 采样数量（如果为None，使用数据集大小）
            replacement: 是否允许重复采样
            generator: 随机数生成器
        """
        # 计算每个样本的权重
        if dataset_weights is None:
            # 默认均匀权重
            weights = torch.ones(len(dataset))
        else:
            weights = torch.zeros(len(dataset))
            for idx in range(len(dataset)):
                sample = dataset[idx]
                dataset_id = sample.get('dataset_id', -1)
                weight = dataset_weights.get(dataset_id, 1.0)
                weights[idx] = weight
        
        if num_samples is None:
            num_samples = len(dataset)
        
        super().__init__(weights, num_samples, replacement, generator)

