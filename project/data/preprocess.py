import os
import cv2
import librosa
import numpy as np
import torch
import re
from scipy import signal
from transformers import BertTokenizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    多模态数据预处理类
    负责处理视频、音频、生理信号和文本数据，使其符合模型输入要求。
    """
    def __init__(self, config):
        """
        初始化预处理器
        
        Args:
            config (dict): 包含数据配置的字典，例如帧率、采样率等
        """
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['model']['text']['backbone'])
        logger.info("DataPreprocessor initialized.")

    def extract_frames(self, video_path):
        """
        从视频中提取固定数量的帧
        支持多种视频格式，包括.flv（需要ffmpeg支持）
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            np.ndarray: 提取的帧数组，形状为 (T, H, W, C)
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            fps = self.config['data']['video']['fps']
            num_frames = self.config['data']['video']['num_frames']
            frame_size = self.config['data']['video']['frame_size']
            return np.zeros((num_frames, frame_size, frame_size, 3))
        
        # 检查视频格式
        video_ext = os.path.splitext(video_path)[1].lower()
        supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
        
        if video_ext not in supported_formats:
            logger.warning(f"Unsupported video format: {video_ext}. Supported formats: {supported_formats}")
        
        # 对于.flv格式，检查OpenCV是否支持（需要ffmpeg）
        if video_ext == '.flv':
            logger.info(f"Processing FLV format video: {video_path}")
            logger.info("Note: FLV format requires ffmpeg support. If extraction fails, please install ffmpeg.")
        
        fps = self.config['data']['video']['fps']
        num_frames = self.config['data']['video']['num_frames']
        frame_size = self.config['data']['video']['frame_size']
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 检查视频是否成功打开
            if not cap.isOpened():
                error_msg = f"Failed to open video file: {video_path}"
                if video_ext == '.flv':
                    error_msg += "\nFLV format may require ffmpeg. Please ensure ffmpeg is installed and OpenCV is compiled with ffmpeg support."
                logger.error(error_msg)
                cap.release()
                return np.zeros((num_frames, frame_size, frame_size, 3))
            
            frames = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整大小
                frame = cv2.resize(frame, (frame_size, frame_size))
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                if video_ext == '.flv':
                    logger.warning("FLV format extraction failed. Please check if ffmpeg is properly installed.")
                return np.zeros((num_frames, frame_size, frame_size, 3))

            # 采样固定数量的帧
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                frames = [frames[i] for i in indices]
            elif len(frames) < num_frames:
                # 重复最后一帧进行填充
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            
            # 归一化并转换形状 (T, H, W, C) -> (T, C, H, W)
            frames = np.array(frames, dtype=np.float32) / 255.0
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            if video_ext == '.flv':
                logger.error("FLV format error. Please ensure ffmpeg is installed and OpenCV supports FLV.")
            return np.zeros((num_frames, frame_size, frame_size, 3))

    def preprocess_audio(self, audio_path):
        """
        预处理音频文件：重采样、裁剪或填充
        支持多种音频格式（.wav, .mp3, .flac, .m4a等）
        
        Args:
            audio_path (str): 音频文件路径
            
        Returns:
            np.ndarray: 预处理后的音频波形
        """
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            target_sr = self.config['data']['audio']['sample_rate']
            duration = self.config['data']['audio']['duration']
            return np.zeros(int(target_sr * duration))
        
        target_sr = self.config['data']['audio']['sample_rate']
        duration = self.config['data']['audio']['duration']
        
        try:
            # 检查音频格式
            audio_ext = os.path.splitext(audio_path)[1].lower()
            supported_formats = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
            
            if audio_ext not in supported_formats:
                logger.warning(f"Unsupported audio format: {audio_ext}. Supported formats: {supported_formats}")
                logger.info("librosa will attempt to load the file anyway.")
            
            # 加载音频，重采样到目标采样率
            # librosa支持多种格式，会自动处理
            audio, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
            
            # 填充或裁剪
            target_length = int(target_sr * duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                audio = audio[:target_length]
                
            return audio
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            logger.error(f"Audio format: {os.path.splitext(audio_path)[1]}")
            logger.error("Please ensure the audio file is valid and librosa supports its format.")
            return np.zeros(int(target_sr * duration))

    def load_physiological_data(self, file_path):
        """
        加载并预处理生理信号数据
        包括加载、带通滤波和Z-score归一化
        
        Args:
            file_path (str): 生理信号文件路径 (.npy)
            
        Returns:
            np.ndarray: 处理后的生理信号数据 (T, channels)
        """
        try:
            data = np.load(file_path)
            # 确保形状 (T, channels)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # 滤波
            sr = self.config['data']['physiological']['sampling_rate']
            data = self._filter_physiological_signal(data, sampling_rate=sr)
            
            # 归一化
            data = self._normalize_physiological_signal(data)
            
            # 简单的截断或填充逻辑（假设需要固定长度，可根据需求修改）
            window_size = self.config['data']['physiological']['window_size']
            target_len = int(window_size * sr)
            if len(data) < target_len:
                pad_width = ((0, target_len - len(data)), (0, 0))
                data = np.pad(data, pad_width, mode='constant')
            elif len(data) > target_len:
                data = data[:target_len, :]
                
            return data
        except Exception as e:
            logger.error(f"Error processing physiological data {file_path}: {e}")
            return np.zeros((int(self.config['data']['physiological']['window_size'] * self.config['data']['physiological']['sampling_rate']), 1))

    def _filter_physiological_signal(self, data, sampling_rate=128, lowcut=0.5, highcut=40):
        """内部方法：带通滤波"""
        nyquist = sampling_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            # 避免数据过短导致滤波报错
            if len(data[:, i]) > 10: 
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
            else:
                filtered_data[:, i] = data[:, i]
        return filtered_data

    def _normalize_physiological_signal(self, data):
        """内部方法：Z-score归一化"""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        return (data - mean) / (std + 1e-8)

    def clean_text(self, text):
        """
        清洗文本数据
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        text = text.lower()
        # 移除特殊字符（保留字母、数字、空格和基本标点）
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        """
        对文本进行Tokenization
        
        Args:
            text (str): 清洗后的文本
            
        Returns:
            tuple: (input_ids, attention_mask)
        """
        encoded = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    def preprocess_sample(self, sample_id, data_dir):
        """
        处理单个样本的所有模态数据
        支持动态文件扩展名识别（支持.flv, .mp4等多种格式）
        
        Args:
            sample_id (str): 样本ID
            data_dir (str): 数据根目录
            
        Returns:
            dict: 包含所有模态预处理数据的字典
        """
        processed_data = {}
        
        # 支持的视频和音频格式
        video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
        audio_formats = ('.wav', '.mp3', '.flac', '.m4a')
        
        # 1. Video - 支持动态扩展名
        video_path = None
        for ext in video_formats:
            candidate_path = os.path.join(data_dir, 'video', f'{sample_id}{ext}')
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        
        if video_path:
            try:
                frames = self.extract_frames(video_path)
                # (T, H, W, C) -> (T, C, H, W) for PyTorch
                frames = frames.transpose(0, 3, 1, 2)
                processed_data['video'] = torch.from_numpy(frames).float()
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
        else:
            logger.debug(f"Video file not found for sample {sample_id}")
            
        # 2. Audio - 支持动态扩展名
        audio_path = None
        for ext in audio_formats:
            candidate_path = os.path.join(data_dir, 'audio', f'{sample_id}{ext}')
            if os.path.exists(candidate_path):
                audio_path = candidate_path
                break
        
        if audio_path:
            try:
                audio = self.preprocess_audio(audio_path)
                processed_data['audio'] = torch.from_numpy(audio).float()
            except Exception as e:
                logger.error(f"Error processing audio {audio_path}: {e}")
        else:
            logger.debug(f"Audio file not found for sample {sample_id}")
            
        # 3. Physiological
        physio_path = os.path.join(data_dir, 'physiological', f'{sample_id}.npy')
        if os.path.exists(physio_path):
            try:
                physio = self.load_physiological_data(physio_path)
                processed_data['physiological'] = torch.from_numpy(physio).float()
            except Exception as e:
                logger.error(f"Error processing physiological data {physio_path}: {e}")
        else:
            logger.debug(f"Physiological data not found for sample {sample_id}")
            
        # 4. Text
        text_path = os.path.join(data_dir, 'text', f'{sample_id}.txt')
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                text = self.clean_text(text)
                input_ids, attn_mask = self.tokenize_text(text)
                processed_data['text_input_ids'] = input_ids
                processed_data['text_attention_mask'] = attn_mask
            except Exception as e:
                logger.error(f"Error processing text {text_path}: {e}")
        else:
            logger.debug(f"Text file not found for sample {sample_id}")
            
        return processed_data

# 测试代码
if __name__ == "__main__":
    # 模拟配置
    dummy_config = {
        'data': {
            'video': {'fps': 30, 'num_frames': 16, 'frame_size': 224},
            'audio': {'sample_rate': 16000, 'duration': 3.0},
            'physiological': {'sampling_rate': 128, 'window_size': 5.0}
        },
        'model': {
            'text': {'backbone': 'bert-base-uncased'}
        }
    }
    
    preprocessor = DataPreprocessor(dummy_config)
    print("Preprocessor tests passed (initialized successfully).")
