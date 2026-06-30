import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, emotion_names, save_path):
    """
    绘制混淆矩阵并保存
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        emotion_names: 情绪类别名称列表
        save_path: 图片保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(log_csv_path, save_dir):
    """
    从CSV日志绘制训练曲线
    
    Args:
        log_csv_path: 训练日志CSV路径
        save_dir: 图像保存目录
    """
    import pandas as pd
    try:
        df = pd.read_csv(log_csv_path)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    epochs = df['epoch']
    
    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # Accuracy Curve
    if 'val_accuracy' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, df['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
        plt.close()

def plot_modal_contribution(results_dict, save_path):
    """
    绘制各模态贡献度对比图
    
    Args:
        results_dict: {
            'video': 0.65,
            'audio': 0.58,
            'text': 0.72,
            ...
        }
    """
    modals = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modals, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#C7F464'][:len(modals)])
    plt.ylabel('Score (Accuracy/F1)')
    plt.title('Modal Contribution Comparison')
    plt.ylim([0, 1.0])
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
