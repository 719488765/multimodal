"""
辅助函数模块

本模块提供项目中使用的一些通用工具函数，包括：
- 配置文件加载
- 设备设置（CPU/GPU）
- 模型检查点保存和加载
- 评估指标计算

作者：项目开发团队
日期：2024年
"""

import os
import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error


def load_config(config_path):
    """
    加载YAML配置文件
    
    功能说明：
    - 从YAML文件中读取配置信息
    - 配置文件包含模型、训练、数据等所有超参数
    
    参数：
        config_path (str): 配置文件路径，通常是'config/config.yaml'
    
    返回：
        config (dict): 配置字典，包含所有配置项
    
    示例：
        >>> config = load_config('config/config.yaml')
        >>> print(config['model']['attention']['fusion_strategy'])
        'emotion_shift'
    
    注意：
        - 使用yaml.safe_load()安全加载，避免执行恶意代码
        - 文件编码为UTF-8，支持中文配置
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)  # 安全加载YAML文件
    return config


def setup_device(config):
    """
    设置计算设备（CPU或GPU）
    
    功能说明：
    - 根据配置和系统环境自动选择计算设备
    - 优先使用GPU（如果可用），否则使用CPU
    - 打印设备信息，方便调试
    
    参数：
        config (dict): 配置字典，应包含'device'键
            - 'device': 'cuda'或'cpu'，默认为'cuda'
    
    返回：
        device (torch.device): PyTorch设备对象
    
    示例：
        >>> config = {'device': 'cuda'}
        >>> device = setup_device(config)
        Using GPU: NVIDIA GeForce RTX 3090
        >>> print(device)
        device(type='cuda')
    
    注意：
        - 如果配置为'cuda'但系统没有GPU，会自动降级为CPU
        - 使用torch.cuda.is_available()检查GPU是否可用
    """
    device_config = config.get('device', 'cuda')  # 从配置中获取设备设置，默认为'cuda'
    
    # 检查GPU是否可用
    if device_config == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        # 打印GPU信息，方便确认使用的设备
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    保存模型检查点
    
    功能说明：
    - 保存模型、优化器、学习率调度器的状态
    - 保存训练进度信息（epoch、loss等）
    - 用于训练中断后恢复训练，或保存最佳模型
    
    参数：
        model (nn.Module): PyTorch模型对象
        optimizer (torch.optim.Optimizer): 优化器对象
        scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，可以为None
        epoch (int): 当前训练轮数
        loss (float): 当前损失值（通常是验证损失）
        filepath (str): 保存路径，例如'checkpoints/best_model.pth'
    
    保存内容：
        - epoch: 训练轮数
        - model_state_dict: 模型参数
        - optimizer_state_dict: 优化器状态（学习率、动量等）
        - scheduler_state_dict: 学习率调度器状态
        - loss: 损失值
    
    示例：
        >>> save_checkpoint(model, optimizer, scheduler, epoch=10, loss=0.5, 
        ...                 filepath='checkpoints/checkpoint_epoch_10.pth')
        Checkpoint saved to checkpoints/checkpoint_epoch_10.pth
    
    注意：
        - 建议定期保存检查点，避免训练中断导致的数据丢失
        - 文件路径应包含目录，确保目录存在
    """
    checkpoint = {
        'epoch': epoch,                                    # 训练轮数
        'model_state_dict': model.state_dict(),            # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),   # 优化器状态
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # 调度器状态（可能为None）
        'loss': loss                                       # 损失值
    }
    torch.save(checkpoint, filepath)  # 保存到文件
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    功能说明：
    - 从文件中加载之前保存的模型检查点
    - 恢复模型、优化器、学习率调度器的状态
    - 返回训练进度信息
    
    参数：
        filepath (str): 检查点文件路径
        model (nn.Module): PyTorch模型对象，将加载参数到此模型
        optimizer (torch.optim.Optimizer, optional): 优化器对象，如果提供则恢复其状态
        scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，如果提供则恢复其状态
    
    返回：
        epoch (int): 保存时的训练轮数
        loss (float): 保存时的损失值
    
    示例：
        >>> epoch, loss = load_checkpoint('checkpoints/best_model.pth', model, optimizer, scheduler)
        Checkpoint loaded from checkpoints/best_model.pth
        >>> print(f"Resume from epoch {epoch}, loss {loss}")
        Resume from epoch 10, loss 0.5
    
    注意：
        - 使用map_location='cpu'确保在不同设备间兼容
        - 如果检查点中没有某些状态（如scheduler），会跳过加载
        - 确保模型结构匹配，否则会报错
    """
    # 加载检查点文件
    # map_location='cpu'确保在不同设备间兼容（例如在CPU上加载GPU训练的模型）
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器状态（如果提供且存在）
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 提取训练进度信息
    epoch = checkpoint.get('epoch', 0)  # 默认为0
    loss = checkpoint.get('loss', float('inf'))  # 默认为无穷大
    
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss


def calculate_metrics(predictions, targets, task='classification'):
    """
    计算评估指标
    
    功能说明：
    - 根据任务类型（分类或回归）计算相应的评估指标
    - 分类任务：准确率、精确率、召回率、F1分数
    - 回归任务：平均绝对误差（MAE）、均方误差（MSE）、均方根误差（RMSE）
    
    参数：
        predictions (np.ndarray): 模型预测结果
            - 分类任务：类别索引数组，形状为(N,)
            - 回归任务：连续值数组，形状为(N,)或(N, 2)（如果是二维回归）
        targets (np.ndarray): 真实标签
            - 形状应与predictions相同
        task (str): 任务类型，'classification'或'regression'，默认为'classification'
    
    返回：
        metrics (dict): 评估指标字典
            - 分类任务：{'accuracy', 'precision', 'recall', 'f1'}
            - 回归任务：{'mae', 'mse', 'rmse'}
    
    示例：
        >>> # 分类任务
        >>> pred = np.array([0, 1, 2, 0, 1])
        >>> target = np.array([0, 1, 2, 1, 1])
        >>> metrics = calculate_metrics(pred, target, task='classification')
        >>> print(metrics)
        {'accuracy': 0.8, 'precision': 0.8333, 'recall': 0.8, 'f1': 0.8}
        
        >>> # 回归任务
        >>> pred = np.array([0.5, 0.7, 0.3])
        >>> target = np.array([0.6, 0.8, 0.4])
        >>> metrics = calculate_metrics(pred, target, task='regression')
        >>> print(metrics)
        {'mae': 0.1, 'mse': 0.01, 'rmse': 0.1}
    
    注意：
        - 分类任务使用加权平均（weighted average），考虑类别不平衡
        - zero_division=0处理除零情况（当某个类别没有预测时）
    """
    if task == 'classification':
        # 分类任务指标
        accuracy = accuracy_score(targets, predictions)  # 准确率
        # 精确率、召回率、F1分数（加权平均，考虑类别不平衡）
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        return {
            'accuracy': accuracy,    # 准确率：正确预测的比例
            'precision': precision,  # 精确率：预测为正例中真正为正例的比例
            'recall': recall,        # 召回率：真正例中被正确预测的比例
            'f1': f1                 # F1分数：精确率和召回率的调和平均
        }
    elif task == 'regression':
        # 回归任务指标
        mae = mean_absolute_error(targets, predictions)      # 平均绝对误差
        mse = mean_squared_error(targets, predictions)      # 均方误差
        rmse = np.sqrt(mse)                                  # 均方根误差
        return {
            'mae': mae,   # 平均绝对误差：预测值与真实值的平均绝对差
            'mse': mse,   # 均方误差：预测值与真实值的平均平方差
            'rmse': rmse  # 均方根误差：MSE的平方根，与目标值同单位
        }
    else:
        raise ValueError(f"Unknown task: {task}")

