# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
# Embedded file name: /mnt/sda1/lizhichun_24/code/multimodal/project/utils/helpers.py
# Compiled at: 2026-06-11 21:16:58
# Size of source mod 2**32: 16881 bytes
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
import os, json, yaml, torch, numpy as np
from datetime import datetime
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
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
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
    device_config = config.get("device", "cuda")
    if device_config == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def dataloader_worker_init_fn(worker_id):
    """
    DataLoader worker 初始化：限制 OpenCV/BLAS 线程数，降低 fork 后段错误概率。
    """
    import cv2
    cv2.setNumThreads(0)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def get_dataloader_kwargs(config, shuffle=True):
    """
    构建 DataLoader 通用参数（含 worker 安全设置）。
    """
    num_workers = int(config.get("num_workers", 4))
    kwargs = {'num_workers':num_workers, 
     'pin_memory':(torch.cuda.is_available)()}
    if num_workers > 0:
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
        kwargs["persistent_workers"] = bool(config.get("dataloader_persistent_workers", False))
        kwargs["prefetch_factor"] = int(config.get("dataloader_prefetch_factor", 2))
        kwargs["timeout"] = int(config.get("dataloader_timeout", 120))
    if shuffle:
        kwargs["shuffle"] = True
    return kwargs


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, extra=None):
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
    checkpoint = {'epoch':epoch, 
     'model_state_dict':(model.state_dict)(), 
     'optimizer_state_dict':(optimizer.state_dict)(), 
     'scheduler_state_dict':scheduler.state_dict() if scheduler else None, 
     'loss':loss}
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def _write_run_meta(log_dir: str, meta: dict) -> None:
    import json

    path = os.path.join(log_dir, ".run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_run_meta(log_dir: str):
    import json

    path = os.path.join(log_dir, ".run_meta.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_experiment_logging(config, config_path: str = ""):
    """
    初始化实验日志目录与度量保存文件。

    功能：
    - 根据配置文件中的paths和experiment字段，创建本次实验的日志目录
    - experiment.log_run_dir + replace_log_dir: 固定目录并覆盖旧 TensorBoard/指标
    - 固定槽位若已有 metrics 且 job_id 不一致则拒绝启动（防 F_C_ES/AVT 互相覆盖）
    - 返回：
      - log_dir: 本次实验的日志根目录（用于TensorBoard、JSON/CSV等）
      - metrics_json_path: 训练/验证度量JSON日志路径
      - metrics_csv_path: 训练/验证度量CSV日志路径
    """
    import hashlib
    import shutil
    from pathlib import Path

    paths_cfg = config.get("paths", {})
    exp_cfg = config.get("experiment", {})
    base_log_dir = paths_cfg.get("log_dir", "logs/")
    exp_name = exp_cfg.get("name", "experiment")
    job_id = str(exp_cfg.get("job_id", "") or exp_name)
    reuse_name = os.environ.get("MULTIMODAL_LOG_RUN_DIR_NAME", "").strip()
    config_fixed = str(exp_cfg.get("log_run_dir", "")).strip()
    fixed_run = config_fixed or reuse_name
    replace = bool(exp_cfg.get("replace_log_dir", False))

    if fixed_run:
        log_dir = os.path.join(base_log_dir, fixed_run)
        metrics_csv = os.path.join(log_dir, "metrics.csv")
        if not replace and os.path.isfile(metrics_csv) and os.path.getsize(metrics_csv) > 0:
            prev = _read_run_meta(log_dir)
            prev_job = (prev or {}).get("job_id")
            if prev_job and prev_job != job_id:
                raise RuntimeError(
                    f"Log slot conflict: {log_dir} already has metrics for job_id={prev_job!r}, "
                    f"but current config requests job_id={job_id!r}. "
                    f"Set experiment.replace_log_dir: true or archive the old run first."
                )
        if replace and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        if replace:
            ckpt_base = paths_cfg.get("checkpoint_dir", "checkpoints/")
            ckpt_run = os.path.join(ckpt_base, fixed_run)
            if os.path.isdir(ckpt_run):
                shutil.rmtree(ckpt_run)
            for old in Path(base_log_dir).glob(f"{exp_name}_*"):
                if old.is_dir() and old.name != fixed_run:
                    shutil.rmtree(old, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, f"{exp_name}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    metrics_json_path = os.path.join(log_dir, "metrics.jsonl")
    metrics_csv_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", encoding="utf-8") as f:
            header = ['epoch',  'phase',  'loss', 
             'cls_loss',  'reg_loss',  'domain_loss',  'trend_loss', 
             'cls_ce_unweighted', 
             'accuracy',  'precision',  'recall',  'f1']
            f.write(",".join(header) + "\n")
    cfg_hash = ""
    if config_path and os.path.isfile(config_path):
        with open(config_path, "rb") as cf:
            cfg_hash = hashlib.md5(cf.read()).hexdigest()
    _write_run_meta(
        log_dir,
        {
            "job_id": job_id,
            "experiment_name": exp_name,
            "log_run_dir": fixed_run or os.path.basename(log_dir),
            "config_path": config_path,
            "config_md5": cfg_hash,
            "replace_log_dir": replace,
        },
    )
    return (
     log_dir, metrics_json_path, metrics_csv_path)


def append_metrics_json(metrics_json_path, record):
    """
    将单条度量记录追加到JSONL文件中（每行一个JSON）。
    """
    with open(metrics_json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_metrics_csv(metrics_csv_path, record):
    """
    将单条度量记录追加到CSV文件中。
    仅写入常用字段，不强制要求所有字段都有值。
    若文件已存在，按首行表头列顺序写入，兼容未含 cls_ce_unweighted 的旧 CSV。
    """
    default_cols = [
     'epoch',  'phase',  'loss', 
     'cls_loss',  'reg_loss',  'domain_loss',  'trend_loss', 
     'cls_ce_unweighted', 
     'accuracy',  'precision',  'recall',  'f1']
    cols = default_cols
    if os.path.exists(metrics_csv_path):
        if os.path.getsize(metrics_csv_path) > 0:
            with open(metrics_csv_path, "r", encoding="utf-8") as rf:
                first = rf.readline().strip()
            if first:
                cols = first.split(",")
    values = []
    for c in cols:
        v = record.get(c, "")
        if isinstance(v, float):
            values.append(f"{v:.6f}")
        else:
            values.append(str(v))
    else:
        line = ",".join(values)
        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def remap_legacy_checkpoint_state_dict(
    state_dict: dict,
    model_state_dict: dict,
) -> tuple[dict, list[str]]:
    """
    将旧版 checkpoint 键名/层级映射到当前模型结构。
    典型场景：video feature_projection 由 Linear@0 改为 LayerNorm@0 + Linear@1。
    返回 (remapped_state_dict, applied_rules)。
    """
    out = dict(state_dict)
    rules: list[str] = []

    old_w = out.get("video_extractor.feature_projection.0.weight")
    model_ln_w = model_state_dict.get("video_extractor.feature_projection.0.weight")
    model_lin_w = model_state_dict.get("video_extractor.feature_projection.1.weight")
    if (
        old_w is not None
        and model_ln_w is not None
        and model_lin_w is not None
        and old_w.dim() == 2
        and model_ln_w.dim() == 1
        and tuple(old_w.shape) == tuple(model_lin_w.shape)
    ):
        for suffix in ("weight", "bias"):
            old_key = f"video_extractor.feature_projection.0.{suffix}"
            new_key = f"video_extractor.feature_projection.1.{suffix}"
            if old_key in out and new_key in model_state_dict:
                out[new_key] = out.pop(old_key)
        rules.append("video_extractor.feature_projection: Linear@0 -> Linear@1")

    return out, rules


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, strict=True):
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
    checkpoint = torch.load(filepath, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict, remap_rules = remap_legacy_checkpoint_state_dict(state_dict, model.state_dict())
    if remap_rules:
        print(f"Legacy checkpoint remap: {', '.join(remap_rules)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if remap_rules and (missing or unexpected):
        print(
            f"  after remap: missing={len(missing)} unexpected={len(unexpected)}"
        )
    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            if checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    print(f"Checkpoint loaded from {filepath}")
    return (epoch, loss)


def load_checkpoint_partial(filepath, model, skip_prefixes=None, skip_keys=None, strict=False):
    """
    部分加载 checkpoint（例如更换 text backbone 时跳过 text encoder 权重）。
    返回 (epoch, loss, loaded_count, skipped_keys)。
    """
    skip_prefixes = skip_prefixes or []
    skip_keys = set(skip_keys or [])
    checkpoint = torch.load(filepath, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", {})
    state_dict, remap_rules = remap_legacy_checkpoint_state_dict(state_dict, model.state_dict())
    if remap_rules:
        print(f"Legacy checkpoint remap: {', '.join(remap_rules)}")
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in skip_keys:
            skipped.append(key)
        elif any((key.startswith(prefix) for prefix in skip_prefixes)):
            skipped.append(key)
        else:
            filtered[key] = value
    else:
        missing, unexpected = model.load_state_dict(filtered, strict=strict)
        print(f"Partial checkpoint loaded from {filepath}: loaded={len(filtered)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
        return (
         checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf")), len(filtered), skipped)


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
    if task == "classification":
        predictions = np.asarray(predictions).reshape(-1)
        targets = np.asarray(targets).reshape(-1)
        if predictions.size == 0 or targets.size == 0:
            return {
             'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(f"calculate_metrics classification: predictions/targets length mismatch: {predictions.shape[0]} vs {targets.shape[0]}")
        try:
            predictions_int = predictions.astype(int)
            targets_int = targets.astype(int)
        except Exception as e:
            try:
                raise ValueError(f"calculate_metrics classification: failed to cast to int: {e}")
            finally:
                e = None
                del e

        else:
            num_classes = int(max(predictions_int.max(), targets_int.max())) + 1
            num_classes = max(num_classes, 1)
            labels = np.arange(num_classes, dtype=int)
            accuracy = accuracy_score(targets_int, predictions_int)
            precision, recall, f1, _ = precision_recall_fscore_support(targets_int,
              predictions_int,
              average="weighted",
              labels=labels,
              zero_division=0)
            return {'accuracy':float(accuracy), 
             'precision':float(precision), 
             'recall':float(recall), 
             'f1':float(f1)}
    if task == "regression":
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        return {'mae':mae, 
         'mse':mse, 
         'rmse':rmse}
    raise ValueError(f"Unknown task: {task}")

# okay decompiling /home/lizhichun_24/sda1/code/multimodal/project/utils/__pycache__/helpers.cpython-38.pyc
