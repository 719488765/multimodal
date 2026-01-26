import argparse
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_config, setup_device, save_checkpoint, calculate_metrics
from data.dataset import MultimodalDataset
from models.multimodal_model import MultimodalEmotionModel
# 假设有一个MultimodalLoss定义在models/losses.py或直接在scanript中定义
# from models.losses import MultimodalLoss 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalLoss(nn.Module):
    """
    多模态任务的综合损失函数
    包含分类损失、回归损失和趋势预测损失（可选）
    """
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        # 趋势预测损失视具体任务定义，这里假设也是MSE
        self.trend_criterion = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        计算总损失
        outputs: 模型输出字典
        targets: 真实标签字典
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. 分类损失 (Emotion Classification)
        if 'emotion_logits' in outputs and 'emotion_label' in targets:
            cls_loss = self.cls_criterion(outputs['emotion_logits'], targets['emotion_label'])
            loss_dict['classification'] = cls_loss.item()
            total_loss += self.weights['classification'] * cls_loss

        # 2. 回归损失 (Valence/Arousal)
        if 'emotion_intensity' in outputs and 'emotion_dimensions' in targets:
            reg_loss = self.reg_criterion(outputs['emotion_intensity'], targets['emotion_dimensions'])
            loss_dict['regression'] = reg_loss.item()
            total_loss += self.weights['regression'] * reg_loss

        # 3. 趋势预测损失 (可选)
        if 'trend_prediction' in outputs and 'trend_label' in targets:
            trend_loss = self.trend_criterion(outputs['trend_prediction'], targets['trend_label'])
            loss_dict['trend'] = trend_loss.item()
            total_loss += self.weights['trend'] * trend_loss

        return total_loss, loss_dict

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个Epoch"""
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        # 将数据移至设备
        inputs = {
            'video': batch['video'].to(device) if batch.get('video') is not None else None,
            'audio': batch['audio'].to(device) if batch.get('audio') is not None else None,
            'physiological': batch['physiological'].to(device) if batch.get('physiological') is not None else None,
            'text_input_ids': batch['text_input_ids'].to(device) if batch.get('text_input_ids') is not None else None,
            'text_attention_mask': batch['text_attention_mask'].to(device) if batch.get('text_attention_mask') is not None else None
        }
        
        targets = {
            'emotion_label': batch['emotion_label'].to(device) if batch.get('emotion_label') is not None else None,
            'emotion_dimensions': batch['emotion_dimensions'].to(device) if batch.get('emotion_dimensions') is not None else None,
            # 'trend_label': ...
        }

        # 前向传播
        outputs = model(**inputs)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in progress_bar:
            inputs = {
                'video': batch['video'].to(device) if batch.get('video') is not None else None,
                'audio': batch['audio'].to(device) if batch.get('audio') is not None else None,
                'physiological': batch['physiological'].to(device) if batch.get('physiological') is not None else None,
                'text_input_ids': batch['text_input_ids'].to(device) if batch.get('text_input_ids') is not None else None,
                'text_attention_mask': batch['text_attention_mask'].to(device) if batch.get('text_attention_mask') is not None else None
            }
            
            targets = {
                'emotion_label': batch['emotion_label'].to(device) if batch.get('emotion_label') is not None else None,
                'emotion_dimensions': batch['emotion_dimensions'].to(device) if batch.get('emotion_dimensions') is not None else None,
            }

            outputs = model(**inputs)
            loss, _ = criterion(outputs, targets)
            running_loss += loss.item()
            
            # 收集预测结果用于计算指标 (仅示例分类准确率)
            if 'emotion_logits' in outputs:
                preds = torch.argmax(outputs['emotion_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets['emotion_label'].cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_preds) # 假设metrics返回字典
    logger.info(f"Validation Loss: {avg_loss:.4f}, Metrics: {metrics}")
    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="多模态情绪识别模型训练脚本")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径 (config.yaml)')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='训练模式')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--dataset', type=str, default=None, help='指定数据集名称 (crema, meld, mosei)，如果不指定则自动检测')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    
    # 2. 设置设备
    device = setup_device(config)
    logger.info(f"Using device: {device}")

    # 3. 数据集选择和配置调整
    datasets_config = config.get('datasets', {})
    
    # 如果指定了数据集，调整模型输出类别数
    if args.dataset and args.dataset.lower() in datasets_config:
        dataset_name = args.dataset.lower()
        dataset_config = datasets_config[dataset_name]
        emotion_classes = dataset_config.get('emotion_classes', 7)
        logger.info(f"Using dataset: {dataset_name.upper()}, emotion classes: {emotion_classes}")
        
        # 更新模型配置中的情感类别数
        config['model']['output']['emotion_classes'] = emotion_classes
    else:
        logger.info("No specific dataset specified, using default configuration")
        logger.info("Dataset will be auto-detected from file naming patterns")

    # 4. 数据加载
    data_dir = config['data']['root_dir']
    batch_size = config['training']['batch_size']
    
    # 根据模式选择数据集路径或split
    if args.mode == 'finetune':
        logger.info("Starting Fine-tuning mode...")
        # 微调模式可以使用不同的数据目录
        finetune_datasets = config['training']['finetune'].get('datasets', [])
        if finetune_datasets:
            logger.info(f"Fine-tuning datasets: {finetune_datasets}")
        train_dataset = MultimodalDataset(data_dir, split='train', config=config)
    else:
        logger.info("Starting Pre-training mode...")
        # 预训练模式支持多个数据集
        pretrain_datasets = config['training']['pretrain'].get('datasets', [])
        if pretrain_datasets:
            logger.info(f"Pre-training datasets: {pretrain_datasets}")
        train_dataset = MultimodalDataset(data_dir, split='train', config=config)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = MultimodalDataset(data_dir, split='val', config=config)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 5. 模型初始化（使用更新后的配置）
    model = MultimodalEmotionModel(config).to(device)

    # 5. 为了微调，冻结部分层（如果是finetune模式）
    if args.mode == 'finetune' and config['training']['finetune'].get('freeze_backbone', False):
        logger.info("Freezing backbone layers for fine-tuning...")
        for param in model.video_extractor.parameters():
            param.requires_grad = False
        for param in model.audio_extractor.parameters():
            param.requires_grad = False
        # ... 对其他提取器冻结
        
    # 6. 优化器与损失函数
    loss_weights = config['training']['loss_weights']
    criterion = MultimodalLoss(loss_weights).to(device)
    
    learning_rate = config['training']['learning_rate']
    if args.mode == 'finetune':
        learning_rate = config['training'].get('finetune_learning_rate', learning_rate * 0.1)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])

    # 7. 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint and args.mode == 'pretrain': # 微调通常重置优化器
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    # 8. 训练循环
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, is_best=True, filename=f"checkpoint_{args.mode}_best.pth")
            
        # 定期保存
        if (epoch + 1) % 5 == 0:
             save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, is_best=False, filename=f"checkpoint_{args.mode}_epoch_{epoch}.pth")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()
