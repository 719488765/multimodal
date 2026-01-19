"""
推理脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2  # type: ignore
import librosa  # type: ignore
from transformers import BertTokenizer  # type: ignore

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MultimodalEmotionModel
from utils import load_config, setup_device, load_checkpoint


# 情绪类别名称
EMOTION_NAMES = ['happy', 'sad', 'angry', 'fear', 'neutral', 'anxious', 'other']


def preprocess_video(video_path, config):
    """
    预处理视频
    """
    video_config = config['data']['video']
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = video_config.get('fps', 30)
    frame_size = video_config.get('frame_size', 224)
    num_frames = video_config.get('num_frames', 16)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # 采样固定数量的帧
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    
    frames = np.array(frames)
    frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
    frames = frames.astype(np.float32) / 255.0
    
    return torch.from_numpy(frames).unsqueeze(0)  # (1, T, C, H, W)


def preprocess_audio(audio_path, config):
    """
    预处理音频
    """
    audio_config = config['data']['audio']
    sample_rate = audio_config.get('sample_rate', 16000)
    duration = audio_config.get('duration', 3.0)
    
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        target_length = int(sample_rate * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        
        return torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    except:
        return None


def preprocess_physiological(physiological_path):
    """
    预处理生理信号
    """
    try:
        data = np.load(physiological_path)
        return torch.from_numpy(data).float().unsqueeze(0)  # (1, T, input_dim)
    except:
        return None


def preprocess_text(text, config):
    """
    预处理文本
    """
    text_backbone = config['model']['text']['backbone']
    tokenizer = BertTokenizer.from_pretrained(text_backbone)
    max_length = 128
    
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return encoded['input_ids'], encoded['attention_mask']


def inference(model, video_path=None, audio_path=None, physiological_path=None, 
              text=None, device='cuda'):
    """
    执行推理
    """
    model.eval()
    
    # 预处理输入
    video = None
    audio = None
    physiological = None
    text_input_ids = None
    text_attention_mask = None
    
    if video_path:
        video = preprocess_video(video_path, config).to(device)
    
    if audio_path:
        audio = preprocess_audio(audio_path, config).to(device)
    
    if physiological_path:
        physiological = preprocess_physiological(physiological_path).to(device)
    
    if text:
        text_input_ids, text_attention_mask = preprocess_text(text, config)
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(
            video=video,
            audio=audio,
            physiological=physiological,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )
    
    # 解析结果
    emotion_probs = outputs['emotion_probs'].cpu().numpy()[0]
    emotion_id = np.argmax(emotion_probs)
    emotion_name = EMOTION_NAMES[emotion_id]
    emotion_confidence = emotion_probs[emotion_id]
    
    emotion_dimensions = outputs['emotion_dimensions'].cpu().numpy()[0]
    valence = emotion_dimensions[0]
    arousal = emotion_dimensions[1]
    
    results = {
        'emotion': emotion_name,
        'emotion_id': int(emotion_id),
        'confidence': float(emotion_confidence),
        'valence': float(valence),
        'arousal': float(arousal),
        'all_probs': emotion_probs.tolist()
    }
    
    if 'trend_prediction' in outputs:
        trend_probs = outputs['trend_prediction'].cpu().numpy()[0]
        trend_id = np.argmax(trend_probs)
        results['trend'] = EMOTION_NAMES[trend_id]
        results['trend_confidence'] = float(trend_probs[trend_id])
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference with Multimodal Emotion Recognition Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file')
    parser.add_argument('--physiological', type=str, default=None, help='Path to physiological signal file')
    parser.add_argument('--text', type=str, default=None, help='Text input')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing input files')
    args = parser.parse_args()
    
    # 加载配置
    global config
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(config)
    
    # 创建模型
    model = MultimodalEmotionModel(config).to(device)
    
    # 加载检查点
    load_checkpoint(args.model_path, model)
    print(f"Model loaded from {args.model_path}")
    
    # 执行推理
    if args.input_dir:
        # 批量推理
        print(f"Processing directory: {args.input_dir}")
        # 这里可以实现批量处理逻辑
    else:
        # 单样本推理
        results = inference(
            model,
            video_path=args.video,
            audio_path=args.audio,
            physiological_path=args.physiological,
            text=args.text,
            device=device
        )
        
        # 打印结果
        print("\n=== Inference Results ===")
        print(f"Emotion: {results['emotion']} (ID: {results['emotion_id']})")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"Valence: {results['valence']:.4f}")
        print(f"Arousal: {results['arousal']:.4f}")
        print(f"\nAll Emotion Probabilities:")
        for i, (name, prob) in enumerate(zip(EMOTION_NAMES, results['all_probs'])):
            print(f"  {name}: {prob:.4f}")
        
        if 'trend' in results:
            print(f"\nTrend Prediction: {results['trend']} (Confidence: {results['trend_confidence']:.4f})")


if __name__ == '__main__':
    main()

