"""
多模态特征提取器模块

本模块定义了四个特征提取器类，分别用于提取视频、音频、生理信号和文本的特征。
每个特征提取器都使用预训练的深度学习模型作为骨干网络，然后通过投影层将特征维度统一到512维。

作者：项目开发团队
日期：2024年
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model, Wav2Vec2Processor, BertModel, BertTokenizer  # type: ignore
import numpy as np


class VideoFeatureExtractor(nn.Module):
    """
    视频特征提取器 - 使用ResNet-50提取面部表情特征
    
    功能说明：
    - 从视频帧中提取面部表情相关的视觉特征
    - 使用ImageNet预训练的ResNet-50作为骨干网络
    - 支持单帧图像和时序视频帧两种输入格式
    - 通过投影层将特征维度统一到output_dim（默认512）
    
    使用场景：
    - 驾驶员面部表情识别
    - 情绪相关的视觉特征提取
    
    输入格式：
    - 单帧：(B, C, H, W) - 批次大小B，通道数C=3，高度H，宽度W
    - 时序：(B, T, C, H, W) - 额外的时间维度T
    
    输出格式：
    - 单帧：(B, output_dim)
    - 时序：(B, T, output_dim)
    
    示例：
        >>> extractor = VideoFeatureExtractor(backbone='resnet50', pretrained=True, output_dim=512)
        >>> video_frames = torch.randn(2, 16, 3, 224, 224)  # 批次2，16帧，RGB，224x224
        >>> features = extractor(video_frames)  # 输出: (2, 16, 512)
    """
    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=2048, output_dim=512):
        """
        初始化视频特征提取器
        
        参数说明：
            backbone (str): 骨干网络类型，目前仅支持'resnet50'
            pretrained (bool): 是否使用ImageNet预训练权重，建议设为True
            feature_dim (int): ResNet-50的特征维度，固定为2048
            output_dim (int): 输出特征维度，默认512，用于与其他模态对齐
        """
        super(VideoFeatureExtractor, self).__init__()
        
        # 加载预训练的ResNet-50
        # ResNet-50在ImageNet上预训练，能够提取通用的视觉特征
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            # 移除最后的分类层（fc层），只保留特征提取部分
            # resnet.children()返回所有子模块，[:-1]表示除了最后一个（分类层）之外的所有层
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 2048  # ResNet-50的最终特征维度
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 投影层，将ResNet-50的2048维特征投影到统一的output_dim维度
        # 这样所有模态的特征维度一致，便于后续融合
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，将特征图池化为1x1
            nn.Flatten(),                  # 展平为1D向量
            nn.Linear(backbone_dim, output_dim),  # 线性投影到目标维度
            nn.ReLU(),                     # 激活函数
            nn.Dropout(0.1)                # Dropout防止过拟合
        )
        
    def forward(self, video_frames):
        """
        前向传播，提取视频特征
        
        参数：
            video_frames (torch.Tensor): 
                - 时序视频：(B, T, C, H, W) - 批次大小B，时间步T，通道C=3，高度H，宽度W
                - 单帧图像：(B, C, H, W) - 批次大小B，通道C=3，高度H，宽度W
                
        返回：
            features (torch.Tensor):
                - 时序视频：(B, T, output_dim) - 每个时间步的特征
                - 单帧图像：(B, output_dim) - 单帧特征
                
        处理流程：
            1. 判断输入是时序还是单帧
            2. 如果是时序，将时间维度合并到批次维度，批量处理所有帧
            3. 通过ResNet-50骨干网络提取特征
            4. 通过投影层统一维度
            5. 如果是时序，恢复时间维度
        """
        if len(video_frames.shape) == 5:
            # 处理时序视频 (B, T, C, H, W)
            B, T, C, H, W = video_frames.shape
            # 将时间维度合并到批次维度，方便批量处理
            # (B, T, C, H, W) -> (B*T, C, H, W)
            video_frames = video_frames.view(B * T, C, H, W)
            
            # 通过ResNet-50提取特征
            features = self.backbone(video_frames)  # (B*T, 2048, H', W')
            # 通过投影层统一维度
            features = self.projection(features)  # (B*T, output_dim)
            # 恢复时间维度
            features = features.view(B, T, -1)  # (B, T, output_dim)
        else:
            # 处理单帧图像 (B, C, H, W)
            # 直接通过骨干网络和投影层
            features = self.backbone(video_frames)  # (B, 2048, H', W')
            features = self.projection(features)  # (B, output_dim)
        
        return features


class AudioFeatureExtractor(nn.Module):
    """
    音频特征提取器 - 使用Wav2Vec2提取语音情感特征
    
    功能说明：
    - 从音频波形中提取语音情感相关的声学特征
    - 使用Hugging Face的Wav2Vec2.0预训练模型
    - Wav2Vec2是自监督学习的语音表示模型，能够捕捉语音的语义信息
    - 通过时序平均池化得到固定长度的特征向量
    
    使用场景：
    - 驾驶员语音情感识别
    - 语音中的情绪特征提取
    
    输入格式：
    - audio_waveform: (B, T) - 批次大小B，时间步T（采样点数）
    
    输出格式：
    - features: (B, output_dim) - 固定长度的音频特征向量
    
    示例：
        >>> extractor = AudioFeatureExtractor(backbone='facebook/wav2vec2-base', output_dim=512)
        >>> audio = torch.randn(2, 48000)  # 批次2，采样率16000的3秒音频
        >>> features = extractor(audio)  # 输出: (2, 512)
    """
    def __init__(self, backbone='facebook/wav2vec2-base', pretrained=True, output_dim=512):
        """
        初始化音频特征提取器
        
        参数说明：
            backbone (str): Wav2Vec2模型名称，默认'facebook/wav2vec2-base'
            pretrained (bool): 是否使用预训练权重，建议设为True
            output_dim (int): 输出特征维度，默认512
        """
        super(AudioFeatureExtractor, self).__init__()
        
        # 加载预训练的Wav2Vec2模型
        # Wav2Vec2是Facebook开发的语音表示学习模型，在大量无标签语音数据上预训练
        # 能够学习到语音的语义表示，对情感识别很有帮助
        self.backbone = Wav2Vec2Model.from_pretrained(backbone) if pretrained else Wav2Vec2Model.from_pretrained(backbone)
        self.processor = Wav2Vec2Processor.from_pretrained(backbone)  # 用于音频预处理
        
        # Wav2Vec2的输出维度（base模型为768维）
        backbone_dim = self.backbone.config.hidden_size
        
        # 投影层，将Wav2Vec2的特征维度投影到统一的output_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),  # 线性投影
            nn.ReLU(),                            # 激活函数
            nn.Dropout(0.1)                       # Dropout防止过拟合
        )
        
    def forward(self, audio_waveform):
        """
        前向传播，提取音频特征
        
        参数：
            audio_waveform (torch.Tensor): (B, T) - 音频波形数据
                - B: 批次大小
                - T: 时间步数（采样点数），例如16000采样率*3秒=48000
        
        返回：
            features (torch.Tensor): (B, output_dim) - 音频特征向量
        
        处理流程：
            1. 通过Wav2Vec2提取时序特征（每个时间步都有特征）
            2. 通过投影层统一维度
            3. 使用时序平均池化得到固定长度的特征向量
        """
        # Wav2Vec2处理音频波形
        # 使用torch.no_grad()可以节省内存，因为Wav2Vec2在推理时不需要梯度
        with torch.no_grad():
            outputs = self.backbone(audio_waveform)
            # 使用最后一层的隐藏状态，包含最丰富的语义信息
            # Wav2Vec2的输出是时序的，每个时间步都有特征
            hidden_states = outputs.last_hidden_state  # (B, T, hidden_dim=768)
        
        # 投影到统一维度
        features = self.projection(hidden_states)  # (B, T, output_dim=512)
        
        # 时序平均池化，将时序特征聚合为固定长度的向量
        # 这是常用的时序特征聚合方法，也可以使用最大池化或注意力机制
        features = torch.mean(features, dim=1)  # (B, output_dim=512)
        
        return features


class PhysiologicalFeatureExtractor(nn.Module):
    """
    生理信号特征提取器 - 使用1D-CNN或LSTM处理EEG、ECG、GSR等时序信号
    
    功能说明：
    - 从生理信号时序数据中提取情绪相关的生理特征
    - 支持两种架构：LSTM（适合长时序）和1D-CNN（适合局部特征）
    - LSTM能够捕捉长期依赖关系，适合处理长时序生理信号
    - CNN能够提取局部模式，计算效率更高
    
    使用场景：
    - 驾驶员生理信号监测（心率、皮肤电导、脑电等）
    - 情绪相关的生理特征提取
    
    输入格式：
    - physiological_data: (B, T, input_dim) - 批次B，时间步T，特征维度input_dim
    
    输出格式：
    - features: (B, output_dim) - 固定长度的生理特征向量
    
    示例：
        >>> extractor = PhysiologicalFeatureExtractor(input_dim=64, use_lstm=True, output_dim=512)
        >>> physiological = torch.randn(2, 100, 64)  # 批次2，100个时间步，64维特征
        >>> features = extractor(physiological)  # 输出: (2, 512)
    """
    def __init__(self, input_dim=64, hidden_dim=256, num_layers=2, output_dim=512, use_lstm=True):
        """
        初始化生理信号特征提取器
        
        参数说明：
            input_dim (int): 输入特征维度，例如EEG通道数、ECG特征数等，默认64
            hidden_dim (int): 隐藏层维度，默认256
            num_layers (int): LSTM层数或CNN层数，默认2
            output_dim (int): 输出特征维度，默认512
            use_lstm (bool): 是否使用LSTM架构，True使用LSTM，False使用CNN
        """
        super(PhysiologicalFeatureExtractor, self).__init__()
        
        self.use_lstm = use_lstm
        
        if use_lstm:
            # LSTM架构：适合处理长时序数据，能够捕捉长期依赖关系
            # 双向LSTM能够同时利用前向和后向信息
            self.lstm = nn.LSTM(
                input_size=input_dim,      # 输入特征维度
                hidden_size=hidden_dim,    # 隐藏层维度
                num_layers=num_layers,      # LSTM层数
                batch_first=True,           # 输入格式为(B, T, input_dim)
                bidirectional=True         # 双向LSTM，输出维度翻倍
            )
            lstm_output_dim = hidden_dim * 2  # 双向LSTM，前向+后向
        else:
            # 1D-CNN架构：适合提取局部特征，计算效率高
            # 使用一维卷积在时间维度上进行特征提取
            self.conv_layers = nn.Sequential(
                # 第一层卷积：提取局部特征
                nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),  # 保持时间维度不变
                nn.ReLU(),
                nn.MaxPool1d(2),  # 时间维度减半
                # 第二层卷积：提取更高层特征
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),  # 时间维度再减半
                # 第三层卷积：最终特征提取
                nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，输出固定长度
            )
            lstm_output_dim = hidden_dim
        
        # 投影层，将LSTM或CNN的输出投影到统一的output_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),  # 线性投影
            nn.ReLU(),                               # 激活函数
            nn.Dropout(0.1)                          # Dropout防止过拟合
        )
        
    def forward(self, physiological_data):
        """
        前向传播，提取生理信号特征
        
        参数：
            physiological_data (torch.Tensor): (B, T, input_dim) - 生理信号时序数据
                - B: 批次大小
                - T: 时间步数
                - input_dim: 每个时间步的特征维度（例如EEG通道数）
        
        返回：
            features (torch.Tensor): (B, output_dim) - 生理特征向量
        
        处理流程：
            1. 根据use_lstm选择LSTM或CNN架构
            2. LSTM：处理时序数据，使用最后一个时间步的输出
            3. CNN：转置后使用1D卷积提取特征，最后池化
            4. 通过投影层统一维度
        """
        if self.use_lstm:
            # LSTM处理时序数据
            # LSTM会处理整个时序，输出每个时间步的隐藏状态
            lstm_out, (h_n, c_n) = self.lstm(physiological_data)
            # lstm_out: (B, T, hidden_dim*2) - 每个时间步的双向LSTM输出
            # h_n: (num_layers*2, B, hidden_dim) - 最后一个时间步的隐藏状态
            # c_n: (num_layers*2, B, hidden_dim) - 最后一个时间步的细胞状态
            
            # 使用最后一个时间步的输出（包含整个序列的信息）
            features = lstm_out[:, -1, :]  # (B, hidden_dim * 2)
        else:
            # 1D-CNN处理
            # Conv1d需要输入格式为(B, channels, time)，所以需要转置
            # (B, T, input_dim) -> (B, input_dim, T)
            x = physiological_data.transpose(1, 2)
            # 通过卷积层提取特征
            features = self.conv_layers(x).squeeze(-1)  # (B, hidden_dim)
            # squeeze(-1)移除最后一个维度（AdaptiveAvgPool1d输出为(B, hidden_dim, 1)）
        
        # 投影到统一维度
        features = self.projection(features)  # (B, output_dim)
        
        return features


class TextFeatureExtractor(nn.Module):
    """
    文本特征提取器 - 使用BERT提取文本语义特征
    
    功能说明：
    - 从文本中提取语义特征，用于情绪识别
    - 使用Hugging Face的BERT预训练模型
    - BERT是双向编码器表示模型，能够理解文本的上下文语义
    - 使用[CLS] token的表示作为整个文本的语义表示
    
    使用场景：
    - 驾驶员语音转文本后的情绪分析
    - 文本对话中的情绪识别
    
    输入格式：
    - input_ids: (B, seq_len) - 文本token IDs
    - attention_mask: (B, seq_len) - 注意力掩码，1表示有效token，0表示padding
    
    输出格式：
    - features: (B, output_dim) - 文本特征向量
    
    示例：
        >>> extractor = TextFeatureExtractor(backbone='bert-base-uncased', output_dim=512)
        >>> input_ids = torch.randint(0, 1000, (2, 128))  # 批次2，序列长度128
        >>> attention_mask = torch.ones(2, 128)
        >>> features = extractor(input_ids, attention_mask)  # 输出: (2, 512)
    """
    def __init__(self, backbone='bert-base-uncased', pretrained=True, output_dim=512):
        """
        初始化文本特征提取器
        
        参数说明：
            backbone (str): BERT模型名称，默认'bert-base-uncased'（不区分大小写）
            pretrained (bool): 是否使用预训练权重，建议设为True
            output_dim (int): 输出特征维度，默认512
        """
        super(TextFeatureExtractor, self).__init__()
        
        # 加载预训练的BERT模型
        # BERT是Google开发的预训练语言模型，在大量文本数据上预训练
        # 能够理解文本的语义和上下文信息，对情绪识别很有帮助
        self.backbone = BertModel.from_pretrained(backbone) if pretrained else BertModel.from_pretrained(backbone)
        self.tokenizer = BertTokenizer.from_pretrained(backbone)  # 用于文本tokenization
        
        # BERT的输出维度（base模型为768维）
        backbone_dim = self.backbone.config.hidden_size
        
        # 投影层，将BERT的768维特征投影到统一的output_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),  # 线性投影
            nn.ReLU(),                            # 激活函数
            nn.Dropout(0.1)                       # Dropout防止过拟合
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播，提取文本特征
        
        参数：
            input_ids (torch.Tensor): (B, seq_len) - 文本token IDs
                - B: 批次大小
                - seq_len: 序列长度（token数量）
                - 每个token ID对应词汇表中的一个词
            attention_mask (torch.Tensor, optional): (B, seq_len) - 注意力掩码
                - 1表示有效token，0表示padding token
                - 如果为None，则所有token都视为有效
        
        返回：
            features (torch.Tensor): (B, output_dim) - 文本特征向量
        
        处理流程：
            1. 通过BERT提取文本的语义表示
            2. 使用[CLS] token的表示（pooler_output）作为整个文本的语义表示
            3. 通过投影层统一维度
        
        注意：
            - [CLS] token是BERT在序列开头添加的特殊token，其表示被训练为包含整个序列的语义信息
            - pooler_output是BERT对[CLS] token的表示进行进一步处理后的输出
        """
        # BERT处理文本
        # 使用torch.no_grad()可以节省内存，因为BERT在推理时不需要梯度
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # BERT的输出包含：
            # - last_hidden_state: (B, seq_len, hidden_dim) - 每个token的隐藏状态
            # - pooler_output: (B, hidden_dim) - [CLS] token的池化表示（整个序列的语义）
            # 使用pooler_output作为整个文本的语义表示
            pooled_output = outputs.pooler_output  # (B, hidden_dim=768)
        
        # 投影到统一维度
        features = self.projection(pooled_output)  # (B, output_dim=512)
        
        return features

