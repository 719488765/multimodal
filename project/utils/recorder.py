import os
import csv
import json
import logging
import wandb
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentRecorder:
    """
    实验记录器
    负责记录实验配置、训练过程中的指标以及最终结果
    支持CSV文件和WandB
    """
    def __init__(self, log_dir, config, exp_name=None):
        self.log_dir = log_dir
        self.config = config
        os.makedirs(log_dir, exist_ok=True)
        
        # 实验名称
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = exp_name if exp_name else f"exp_{timestamp}"
        
        # 保存配置
        self.save_config()
        
        # 初始化CSV记录器
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self.init_csv()
        
        # 初始化WandB
        if config.get('experiment', {}).get('use_wandb', False):
            wandb.init(
                project=config['experiment'].get('project_name', 'driver_emotion_analysis'),
                name=self.exp_name,
                config=config
            )

    def save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Config saved to {config_path}")

    def init_csv(self):
        """初始化CSV文件头"""
        headers = ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'lr']
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_epoch(self, epoch_data):
        """
        记录一个epoch的数据
        Args:
            epoch_data (dict): {'epoch': 1, 'train_loss': 0.5, ...}
        """
        # CSV记录
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # 确保顺序对应
            row = [
                epoch_data.get('epoch'),
                epoch_data.get('train_loss'),
                epoch_data.get('val_loss'),
                epoch_data.get('val_accuracy'),
                epoch_data.get('val_f1'),
                epoch_data.get('lr')
            ]
            writer.writerow(row)
            
        # WandB记录
        if wandb.run:
            wandb.log(epoch_data)
            
        logger.info(f"Logged epoch {epoch_data.get('epoch')} data.")

    def log_final_results(self, metrics):
        """记录最终测试结果"""
        final_path = os.path.join(self.log_dir, 'final_results.json')
        with open(final_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        if wandb.run:
            wandb.summary.update(metrics)
            
        logger.info(f"Final results saved to {final_path}")

    def close(self):
        if wandb.run:
            wandb.finish()
