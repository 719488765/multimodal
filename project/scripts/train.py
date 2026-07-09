#!/usr/bin/env python3
"""SDAVT v3 R4 multimodal emotion training (pretrain / finetune)."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.collate import multimodal_collate_fn
from data.dataset import MultimodalDataset
from models.balanced_loss import ClassBalancedLoss, FocalLoss
from models.multimodal_model import MultimodalEmotionModel
from utils.helpers import (
    append_metrics_csv,
    append_metrics_json,
    calculate_metrics,
    get_dataloader_kwargs,
    init_experiment_logging,
    load_checkpoint_partial,
    load_config,
    save_checkpoint,
    setup_device,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

DS_NAME_TO_ID = {"crema": 0, "meld": 1, "mosei": 2}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _apply_single_domain_emotion_classes(config: dict, mode: str) -> None:
    """When a single dataset is listed, align model.output.emotion_classes with datasets.*."""
    ds_list = config.get("training", {}).get(mode, {}).get("datasets") or []
    if len(ds_list) != 1:
        return
    ds_name = str(ds_list[0]).lower()
    ds_cfg = config.get("datasets", {}).get(ds_name, {})
    n_cls = ds_cfg.get("emotion_classes")
    if n_cls is not None:
        config.setdefault("model", {}).setdefault("output", {})["emotion_classes"] = int(n_cls)


def _resolve_base_dataset(dataset) -> MultimodalDataset:
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    return base


def _subset_by_datasets(dataset, config: dict, mode: str):
    ds_list = config.get("training", {}).get(mode, {}).get("datasets") or []
    if not ds_list:
        return dataset
    target_ids = {
        DS_NAME_TO_ID[str(x).lower()]
        for x in ds_list
        if str(x).lower() in DS_NAME_TO_ID
    }
    if not target_ids:
        return dataset
    base = _resolve_base_dataset(dataset)
    keep = [
        i
        for i, s in enumerate(base.data_list)
        if s.get("dataset_id", -1) in target_ids
    ]
    if not keep:
        print(f"WARNING: no samples for datasets={ds_list}; using full split")
        return dataset
    print(f"Subset {mode}: {len(keep)} samples for {ds_list}")
    return Subset(dataset, keep)


def _build_balanced_sampler(
    dataset,
    mode: str = "proportional",
    generator: Optional[torch.Generator] = None,
) -> Optional[WeightedRandomSampler]:
    """
    Build a WeightedRandomSampler over dataset positions (0..len-1).
    uniform: equal weight per domain; proportional: natural domain mix.
    """
    base = _resolve_base_dataset(dataset)
    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
    else:
        indices = list(range(len(base)))

    groups: Dict[int, List[int]] = defaultdict(list)
    for pos, idx in enumerate(indices):
        ds_id = int(base.data_list[idx].get("dataset_id", -1))
        groups[ds_id].append(pos)

    if len(groups) <= 1:
        return None

    weights = torch.ones(len(indices), dtype=torch.double)
    mode = str(mode or "proportional").lower()
    if mode == "uniform":
        for pos_list in groups.values():
            w_val = 1.0 / max(len(pos_list), 1)
            for pos in pos_list:
                weights[pos] = w_val
    else:
        total = len(indices)
        for pos_list in groups.values():
            w_val = len(pos_list) / total / max(len(pos_list), 1)
            for pos in pos_list:
                weights[pos] = w_val

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Backbone freeze / optimizer
# ---------------------------------------------------------------------------


def _freeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def _unfreeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _resolve_submodule(model: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    obj: Any = model
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, nn.Module) else None


def _apply_text_last_n_layers(text_extractor: nn.Module, n_layers: int) -> None:
    """Re-freeze entire text backbone, then unfreeze only the last N encoder layers."""
    backbone = getattr(text_extractor, "backbone", None)
    if backbone is None:
        return
    _freeze_module(backbone)
    n = max(0, int(n_layers or 0))
    if n > 0 and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        for layer in backbone.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        backbone.train()


def apply_backbone_freeze_policy(
    model: nn.Module,
    config: dict,
    *,
    epoch_frozen: bool,
) -> None:
    """
    Apply backbone freeze policy.

    - audio_extractor.backbone (wav2vec) is ALWAYS kept frozen.
    - video_extractor.backbone (resnet) is ALWAYS kept frozen (prevents post-unfreeze overfit).
    - When text.unfreeze_encoder_layers > 0, text backbone is NEVER fully unfrozen;
      only the last N encoder layers remain trainable (including at epoch boundaries).
    """
    training = config.get("training", {})
    model_cfg = config.get("model", {})
    freeze_mode = str(training.get("backbone_freeze_mode", "full")).lower()
    text_n = int(model_cfg.get("text", {}).get("unfreeze_encoder_layers", 0) or 0)
    selective_modules = list(training.get("trainable_backbone_modules") or [])

    audio_ext = getattr(model, "audio_extractor", None)
    if audio_ext is not None:
        _freeze_module(getattr(audio_ext, "backbone", None))

    text_ext = getattr(model, "text_extractor", None)
    if text_ext is not None and getattr(text_ext, "backbone", None) is not None:
        if text_n > 0:
            _apply_text_last_n_layers(text_ext, text_n)
        elif epoch_frozen or freeze_mode == "selective":
            _freeze_module(text_ext.backbone)
        else:
            _unfreeze_module(text_ext.backbone)

    video_ext = getattr(model, "video_extractor", None)
    video_bb = getattr(video_ext, "backbone", None) if video_ext else None
    if video_bb is not None:
        _freeze_module(video_bb)

    if freeze_mode == "selective":
        for mod_path in selective_modules:
            mod = _resolve_submodule(model, mod_path)
            if mod is not None:
                _unfreeze_module(mod)


def _is_backbone_param(name: str) -> bool:
    prefixes = (
        "video_extractor.backbone",
        "audio_extractor.backbone",
        "text_extractor.backbone",
    )
    return any(name.startswith(p) for p in prefixes)


def build_training_optimizer(
    model: nn.Module,
    config: dict,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_mult = float(config.get("training", {}).get("backbone_lr_multiplier", 1.0))
    backbone_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_backbone_param(name):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups: List[dict] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_mult})

    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    opt_name = str(config.get("training", {}).get("optimizer", "adamw")).lower()
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_training_steps: int,
    last_epoch: int = -1,
):
    training = config.get("training", {})
    sched_name = str(training.get("scheduler", "cosine")).lower()
    warmup_steps = int(training.get("warmup_steps", 0) or 0)

    if sched_name != "cosine" or num_training_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0, last_epoch=last_epoch)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))
        progress = (current_step - warmup_steps) / float(max(num_training_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class MultimodalLoss(nn.Module):
    """Combined classification / regression / trend / domain-adaptation loss."""

    def __init__(self, loss_weights: dict, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.weights = loss_weights or {}
        loss_cfg = self.config.get("training", {}).get("loss", {})

        self.w_cls = float(self.weights.get("classification", 1.0))
        self.w_reg = float(self.weights.get("regression", 0.0))
        self.w_trend = float(self.weights.get("trend", 0.0))
        self.use_domain = bool(loss_cfg.get("use_domain_adaptation", False))
        self.domain_weight = float(loss_cfg.get("domain_loss_weight", 0.1))
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

        num_classes = int(
            self.config.get("model", {}).get("output", {}).get("emotion_classes", 7)
        )
        self.num_classes = num_classes

        self.use_focal = bool(loss_cfg.get("use_focal_loss", False))
        self.use_class_balanced = bool(loss_cfg.get("use_class_balanced", False))

        if self.use_focal:
            self.cls_loss_fn = FocalLoss(
                alpha=float(loss_cfg.get("focal_alpha", 1.0)),
                gamma=float(loss_cfg.get("focal_gamma", 2.0)),
                label_smoothing=self.label_smoothing,
            )
        elif self.use_class_balanced:
            self.cls_loss_fn = ClassBalancedLoss(
                num_classes=num_classes,
                beta=float(loss_cfg.get("class_balance_beta", 0.9999)),
                label_smoothing=self.label_smoothing,
            )
        else:
            self.cls_loss_fn = None

        self.mse = nn.MSELoss()

    def set_fixed_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        if isinstance(self.cls_loss_fn, ClassBalancedLoss) and weights is not None:
            self.cls_loss_fn.fixed_class_weights = weights

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(self.cls_loss_fn, ClassBalancedLoss):
            return self.cls_loss_fn(logits, labels)
        if isinstance(self.cls_loss_fn, FocalLoss):
            return self.cls_loss_fn(logits, labels)
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        logits = outputs.get("emotion_logits")
        if logits is None:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return zero, {"cls_loss": 0.0, "reg_loss": 0.0, "domain_loss": 0.0, "trend_loss": 0.0}

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        labels = targets["emotion_label"]

        cls_loss = self._classification_loss(logits, labels)
        with torch.no_grad():
            cls_ce_unweighted = F.cross_entropy(logits, labels, reduction="mean").item()

        reg_loss = torch.tensor(0.0, device=logits.device)
        if self.w_reg > 0 and "emotion_dimensions" in outputs and "emotion_dimensions" in targets:
            pred = outputs["emotion_dimensions"]
            tgt = targets["emotion_dimensions"]
            if pred is not None and tgt is not None:
                reg_loss = self.mse(pred, tgt)

        trend_loss = torch.tensor(0.0, device=logits.device)
        if self.w_trend > 0 and "trend_prediction" in outputs and "trend_target" in targets:
            pred = outputs["trend_prediction"]
            tgt = targets["trend_target"]
            if pred is not None and tgt is not None:
                trend_loss = self.mse(pred, tgt)

        domain_loss = torch.tensor(0.0, device=logits.device)
        if self.use_domain and self.domain_weight > 0:
            domain_logits = outputs.get("domain_logits")
            dataset_ids = targets.get("dataset_id")
            if domain_logits is not None and dataset_ids is not None:
                ds_ids = dataset_ids.long().clamp(min=0)
                domain_loss = F.cross_entropy(domain_logits, ds_ids)

        total = (
            self.w_cls * cls_loss
            + self.w_reg * reg_loss
            + self.w_trend * trend_loss
            + self.domain_weight * domain_loss
        )

        breakdown = {
            "cls_loss": float(cls_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "trend_loss": float(trend_loss.detach().item()),
            "domain_loss": float(domain_loss.detach().item()),
            "cls_ce_unweighted": float(cls_ce_unweighted),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Batch / training loops
# ---------------------------------------------------------------------------


def _batch_to_model_inputs(batch: dict, device: torch.device) -> dict:
    inputs: dict = {}
    for key in (
        "video",
        "audio",
        "audio_precomputed",
        "physiological",
        "text_input_ids",
        "text_attention_mask",
    ):
        if batch.get(key) is not None:
            inputs[key] = batch[key].to(device)

    if batch.get("context_text_input_ids") is not None:
        inputs["context_text_input_ids"] = batch["context_text_input_ids"].to(device)
        inputs["context_text_attention_mask"] = batch["context_text_attention_mask"].to(device)

    if batch.get("dataset_id") is not None:
        inputs["dataset_ids"] = batch["dataset_id"].to(device)

    return inputs


def _batch_targets(batch: dict, device: torch.device) -> dict:
    targets = {
        "emotion_label": batch["emotion_label"].to(device),
        "emotion_dimensions": batch["emotion_dimensions"].to(device),
    }
    if batch.get("dataset_id") is not None:
        targets["dataset_id"] = batch["dataset_id"].to(device)
    return targets


def _apply_modality_dropout(inputs: dict, config: dict) -> dict:
    """Training-only: randomly drop whole modalities to reduce overfit (M3_M7 combo)."""
    p = float(config.get("model", {}).get("attention", {}).get("modality_dropout", 0) or 0)
    if p <= 0:
        return inputs
    out = dict(inputs)
    slots: List[str] = []
    if out.get("video") is not None:
        slots.append("video")
    if out.get("audio") is not None or out.get("audio_precomputed") is not None:
        slots.append("audio")
    if out.get("text_input_ids") is not None:
        slots.append("text")
    for slot in slots:
        if random.random() >= p:
            continue
        if slot == "video" and out.get("video") is not None:
            out["video"] = torch.zeros_like(out["video"])
        elif slot == "audio":
            if out.get("audio") is not None:
                out["audio"] = torch.zeros_like(out["audio"])
            if out.get("audio_precomputed") is not None:
                out["audio_precomputed"] = torch.zeros_like(out["audio_precomputed"])
        elif slot == "text" and out.get("text_input_ids") is not None:
            out["text_input_ids"] = torch.zeros_like(out["text_input_ids"])
            if out.get("text_attention_mask") is not None:
                out["text_attention_mask"] = torch.zeros_like(out["text_attention_mask"])
    return out


def _collect_labels_for_balanced_loss(dataset, num_classes: int) -> List[int]:
    base = _resolve_base_dataset(dataset)
    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
    else:
        indices = list(range(len(base)))

    labels: List[int] = []
    for idx in indices:
        sample = base.data_list[idx]
        label = sample.get("emotion_label")
        if label is None:
            try:
                label = int(base[idx]["emotion_label"])
            except Exception:
                continue
        label = int(label)
        if 0 <= label < num_classes:
            labels.append(label)
    return labels


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultimodalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    scheduler=None,
    epoch: int = 0,
) -> dict:
    model.train()
    training = config.get("training", {})
    accum = max(1, int(training.get("gradient_accumulation_steps", 1)))
    grad_clip = float(training.get("gradient_clip", 1.0))
    use_domain = bool(training.get("loss", {}).get("use_domain_adaptation", False))

    running = defaultdict(float)
    count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        inputs = _batch_to_model_inputs(batch, device)
        inputs = _apply_modality_dropout(inputs, config)
        targets = _batch_targets(batch, device)

        outputs = model(**inputs, return_domain_logits=use_domain)
        loss, breakdown = criterion(outputs, targets)
        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss at epoch {epoch} step {step}; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / accum).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running["loss"] += loss.item()
        for k, v in breakdown.items():
            running[k] += v
        count += 1

    if count == 0:
        return {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0, "domain_loss": 0.0, "trend_loss": 0.0}
    return {k: v / count for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultimodalLoss,
    device: torch.device,
    config: dict,
) -> dict:
    model.eval()
    use_domain = bool(config.get("training", {}).get("loss", {}).get("use_domain_adaptation", False))

    running = defaultdict(float)
    count = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for batch in loader:
        inputs = _batch_to_model_inputs(batch, device)
        targets = _batch_targets(batch, device)
        outputs = model(**inputs, return_domain_logits=use_domain)
        loss, breakdown = criterion(outputs, targets)

        running["loss"] += loss.item()
        for k, v in breakdown.items():
            running[k] += v
        count += 1

        preds = outputs["emotion_logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets["emotion_label"].cpu().numpy().tolist())

    metrics = {k: v / max(count, 1) for k, v in running.items()}
    cls_metrics = calculate_metrics(predictions=all_preds, targets=all_targets, task="classification")
    metrics.update(cls_metrics)
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------


def _try_load_optimizer_state(
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
) -> None:
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as exc:
            print(f"WARNING: optimizer param groups mismatch; skipping optimizer load: {exc}")
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, RuntimeError) as exc:
            print(f"WARNING: scheduler state mismatch; skipping scheduler load: {exc}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _monitor_is_better(monitor: str, current: float, best: float, min_delta: float) -> bool:
    if monitor in ("val_loss", "loss"):
        return current < best - min_delta
    return current > best + min_delta


def _get_monitor_value(metrics: dict, monitor: str) -> float:
    key = monitor.replace("val_", "") if monitor.startswith("val_") else monitor
    if key in metrics:
        return float(metrics[key])
    if monitor in metrics:
        return float(metrics[monitor])
    return float(metrics.get("f1", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal emotion training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume")
    parser.add_argument("--dataset", type=str, default="", help="Override single dataset name")
    parser.add_argument(
        "--skip_text_encoder",
        action="store_true",
        help="Partial load: skip text_extractor backbone weights",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = copy.deepcopy(config)
    mode = args.mode

    if args.dataset:
        ds = str(args.dataset).lower()
        config.setdefault("training", {}).setdefault(mode, {})["datasets"] = [ds]

    _apply_single_domain_emotion_classes(config, mode)

    training_cfg = config.get("training", {})
    mode_cfg = training_cfg.get(mode, {})
    if not mode_cfg.get("enabled", True):
        print(f"Mode {mode} disabled in config")
        return 0

    num_epochs = int(mode_cfg.get("epochs") or training_cfg.get("num_epochs", 50))
    batch_size = int(training_cfg.get("batch_size", 1))
    lr = float(
        training_cfg.get("finetune_learning_rate")
        if mode == "finetune" and training_cfg.get("finetune_learning_rate") is not None
        else training_cfg.get("learning_rate", 1e-4)
    )
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    freeze_epochs = int(training_cfg.get("freeze_backbone_epochs", 0) or 0)
    freeze_mode = str(training_cfg.get("backbone_freeze_mode", "full"))
    accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))

    seed = int(training_cfg.get("seed", 42))
    _set_seed(seed)

    log_dir, metrics_json_path, metrics_csv_path = init_experiment_logging(config)
    checkpoint_base = config.get("paths", {}).get("checkpoint_dir", "checkpoints/")
    checkpoint_run_dir = os.path.join(checkpoint_base, os.path.basename(log_dir.rstrip("/")))
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    print(f"Experiment log directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_run_dir}")

    device = setup_device(config)

    data_root = config["data"]["root_dir"]
    train_base = MultimodalDataset(data_root, split="train", config=config)
    val_base = MultimodalDataset(data_root, split="val", config=config)
    train_ds = _subset_by_datasets(train_base, config, mode)
    val_ds = _subset_by_datasets(val_base, config, mode)

    dl_kwargs = get_dataloader_kwargs(config, shuffle=False)
    dl_kwargs["collate_fn"] = multimodal_collate_fn

    sampling_cfg = training_cfg.get("sampling", {})
    if sampling_cfg.get("enabled", False) and len(train_ds) > 0:
        gen = torch.Generator()
        gen.manual_seed(seed)
        train_sampler = _build_balanced_sampler(
            train_ds,
            mode=str(sampling_cfg.get("mode", "proportional")),
            generator=gen,
        )
        if train_sampler is not None:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler, **dl_kwargs
            )
        else:
            train_dl_kwargs = get_dataloader_kwargs(config, shuffle=True)
            train_dl_kwargs["collate_fn"] = multimodal_collate_fn
            train_loader = DataLoader(train_ds, batch_size=batch_size, **train_dl_kwargs)
    else:
        train_dl_kwargs = get_dataloader_kwargs(config, shuffle=True)
        train_dl_kwargs["collate_fn"] = multimodal_collate_fn
        train_loader = DataLoader(train_ds, batch_size=batch_size, **train_dl_kwargs)

    val_loader = DataLoader(val_ds, batch_size=batch_size, **dl_kwargs)

    model = MultimodalEmotionModel(config).to(device)
    criterion = MultimodalLoss(training_cfg.get("loss_weights", {}), config=config).to(device)

    loss_cfg = training_cfg.get("loss", {})
    if loss_cfg.get("use_fixed_class_balanced_weights", False):
        labels = _collect_labels_for_balanced_loss(train_ds, criterion.num_classes)
        if labels and isinstance(criterion.cls_loss_fn, ClassBalancedLoss):
            weights = criterion.cls_loss_fn.compute_class_weights(torch.tensor(labels))
            criterion.set_fixed_class_weights(weights)
            print(f"Fixed class-balanced weights from {len(labels)} training labels")

    permanently_freeze = mode == "finetune" and bool(mode_cfg.get("freeze_backbone", False))
    if permanently_freeze or freeze_epochs > 0:
        apply_backbone_freeze_policy(model, config, epoch_frozen=True)
    elif freeze_mode == "selective":
        apply_backbone_freeze_policy(model, config, epoch_frozen=False)

    optimizer = build_training_optimizer(model, config, lr, weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)

    writer = None
    if config.get("experiment", {}).get("use_tensorboard", False) and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    best_loss = float("inf")
    best_f1 = -float("inf")
    es_cfg = training_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_monitor = str(es_cfg.get("monitor", "val_f1"))
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_counter = 0
    es_best = float("inf") if es_monitor in ("val_loss", "loss") else -float("inf")
    save_every = int(training_cfg.get("save_every_n_epochs", 0) or 0)
    ckpt_prefix = f"checkpoint_{mode}"

    if args.resume:
        resume_path = args.resume
        if args.skip_text_encoder:
            start_epoch, resume_loss, _, _ = load_checkpoint_partial(
                resume_path,
                model,
                skip_prefixes=["text_extractor.backbone"],
                strict=False,
            )
            if mode == "finetune":
                start_epoch = 0
                best_loss = float("inf")
                best_f1 = -float("inf")
            print(f"Partial resume from epoch {start_epoch}, loss={resume_loss}")
        else:
            ckpt = torch.load(resume_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_loss = float(ckpt.get("loss", best_loss))
            best_f1 = float(ckpt.get("best_f1", best_f1))
            _try_load_optimizer_state(ckpt, optimizer, scheduler)
            print(f"Resumed from {resume_path} at epoch {start_epoch}")

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, num_epochs):
        if epoch == freeze_epochs and freeze_epochs > 0 and not permanently_freeze:
            print(f"Epoch {epoch}: backbone unfreeze boundary — rebuilding optimizer & scheduler")
            apply_backbone_freeze_policy(model, config, epoch_frozen=False)
            optimizer = build_training_optimizer(model, config, lr, weight_decay)
            remaining_epochs = num_epochs - epoch
            remaining_steps = steps_per_epoch * remaining_epochs
            scheduler = build_scheduler(optimizer, config, remaining_steps)

        elif permanently_freeze:
            apply_backbone_freeze_policy(model, config, epoch_frozen=True)
        elif freeze_epochs > 0 and epoch < freeze_epochs:
            apply_backbone_freeze_policy(model, config, epoch_frozen=True)
        elif freeze_mode == "selective":
            apply_backbone_freeze_policy(model, config, epoch_frozen=False)

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config,
            scheduler=scheduler,
            epoch=epoch,
        )
        val_metrics = validate(model, val_loader, criterion, device, config)

        global_step += steps_per_epoch

        train_record = {"epoch": epoch, "phase": "train", **train_metrics}
        val_record = {"epoch": epoch, "phase": "val", **val_metrics}
        append_metrics_json(metrics_json_path, train_record)
        append_metrics_json(metrics_json_path, val_record)
        append_metrics_csv(metrics_csv_path, train_record)
        append_metrics_csv(metrics_csv_path, val_record)

        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        val_loss = float(val_metrics.get("loss", float("inf")))
        val_f1 = float(val_metrics.get("f1", 0.0))
        print(
            f"Epoch {epoch}/{num_epochs - 1} "
            f"train_loss={train_metrics.get('loss', 0):.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_acc={val_metrics.get('accuracy', 0):.4f}"
        )

        extra = {"best_f1": best_f1, "mode": mode}

        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(checkpoint_run_dir, f"{ckpt_prefix}_best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path, extra=extra)

        if val_f1 > best_f1:
            best_f1 = val_f1
            extra["best_f1"] = best_f1
            best_f1_path = os.path.join(checkpoint_run_dir, f"{ckpt_prefix}_best_f1.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_f1_path, extra=extra)

        if save_every > 0 and (epoch + 1) % save_every == 0:
            epoch_path = os.path.join(checkpoint_run_dir, f"{ckpt_prefix}_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, epoch_path, extra=extra)

        if es_enabled:
            current = _get_monitor_value(val_metrics, es_monitor)
            improved = _monitor_is_better(es_monitor, current, es_best, es_min_delta)
            if improved:
                es_best = current
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    print(
                        f"Early stopping: {es_monitor} did not improve for {es_patience} epochs "
                        f"(best={es_best:.4f}, current={current:.4f})"
                    )
                    break

    best_f1_path = os.path.join(checkpoint_run_dir, f"{ckpt_prefix}_best_f1.pth")
    if os.path.isfile(best_f1_path):
        ckpt = torch.load(best_f1_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        best_f1 = float(ckpt.get("best_f1", best_f1))
        print(f"Restored best-F1 checkpoint from epoch {ckpt.get('epoch', '?')} (val_f1={best_f1:.4f})")

    if writer is not None:
        writer.close()

    print(f"Training finished. Best val loss={best_loss:.4f}, best val F1={best_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
