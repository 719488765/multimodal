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
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor  # type: ignore
import numpy as np

from .precomputed_encoder import TemporalNpyEncoder


def _freeze_pretrained_transformer(backbone: nn.Module, unfreeze_last_n: int = 0) -> None:
    """冻结 HuggingFace 预训练骨干；可选解冻 encoder 最后 N 层（中文 Agent 微调）。"""
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    cfg = getattr(backbone, "config", None)
    if cfg is not None and hasattr(cfg, "gradient_checkpointing"):
        cfg.gradient_checkpointing = False
    for param in backbone.parameters():
        param.requires_grad = False
    n = max(0, int(unfreeze_last_n or 0))
    if n > 0 and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        for layer in backbone.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        backbone.train()
    else:
        backbone.eval()


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
    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        feature_dim=2048,
        output_dim=512,
        input_type='auto',
        max_seq_len=32,
        temporal_hidden_dim=256,
    ):
        """
        初始化视频特征提取器
        
        参数说明：
            backbone (str): 骨干网络类型，目前仅支持'resnet50'
            pretrained (bool): 是否使用ImageNet预训练权重，建议设为True
            feature_dim (int): ResNet-50的特征维度，固定为2048
            output_dim (int): 输出特征维度，默认512，用于与其他模态对齐
            input_type (str): 'auto' | 'cnn' | 'npy' — MOSEI OpenFace 预提取特征用 npy
        """
        super(VideoFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.input_type = input_type
        self.max_seq_len = int(max_seq_len or 32)

        self.backbone = None
        self.cnn_projection = None
        if input_type != 'npy':
            if backbone == 'resnet50':
                resnet = models.resnet50(pretrained=pretrained)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                backbone_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            self.cnn_projection = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(backbone_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # MOSEI OpenFace2 (B,T,713) 等预提取特征
        self.temporal_encoder = TemporalNpyEncoder(
            input_dim=self.feature_dim,
            output_dim=output_dim,
            hidden_dim=temporal_hidden_dim,
            max_seq_len=self.max_seq_len,
            num_layers=1,
            dropout=0.1,
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        # npy 模式：时序 BiLSTM 编码
        if self.input_type == 'npy':
            if len(video_frames.shape) in (2, 3):
                return self.temporal_encoder(video_frames, return_pooled=False)

        # 情况1：时序视频帧 (B, T, C, H, W)
        if len(video_frames.shape) == 5:
            B, T, C, H, W = video_frames.shape
            video_frames = video_frames.view(B * T, C, H, W)
            features = self.backbone(video_frames)              # (B*T, 2048, H', W')
            features = self.cnn_projection(features)            # (B*T, output_dim)
            features = features.view(B, T, -1)                  # (B, T, output_dim)
            return features
        
        # 情况2：单帧图像 (B, C, H, W)
        if len(video_frames.shape) == 4 and video_frames.shape[1] == 3:
            features = self.backbone(video_frames)              # (B, 2048, H', W')
            features = self.cnn_projection(features)            # (B, output_dim)
            return features
        
        # 情况3：特征序列 (B, T, F) - 例如 MOSEI 的 OpenFace2 特征
        if len(video_frames.shape) == 3:
            B, T, F = video_frames.shape
            if F != self.feature_dim:
                raise ValueError(
                    f"Video feature dim {F} != expected {self.feature_dim}"
                )
            x = video_frames.view(B * T, F)
            x = self.feature_projection(x)
            x = x.view(B, T, -1)
            return x
        
        # 情况4：特征向量 (B, F)
        if len(video_frames.shape) == 2:
            F = video_frames.shape[-1]
            if F != self.feature_dim:
                raise ValueError(
                    f"Video feature dim {F} != expected {self.feature_dim}"
                )
            return self.feature_projection(video_frames)
        
        # 其他情况：返回 None，交由上游处理
        return None


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
    def __init__(
        self,
        backbone='facebook/wav2vec2-base',
        pretrained=True,
        output_dim=512,
        precomputed_feature_dim=74,
        audio_mode='auto',
        max_seq_len=32,
        temporal_hidden_dim=256,
    ):
        """
        初始化音频特征提取器
        
        参数说明：
            backbone (str): Wav2Vec2模型名称，默认'facebook/wav2vec2-base'
            pretrained (bool): 是否使用预训练权重，建议设为True
            output_dim (int): 输出特征维度，默认512
            precomputed_feature_dim (int): MOSEI COVAREP 等预提取声学特征维度，默认74
        """
        super(AudioFeatureExtractor, self).__init__()
        
        self.precomputed_feature_dim = int(precomputed_feature_dim or 74)
        self.audio_mode = str(audio_mode or 'auto').lower()
        self.max_seq_len = int(max_seq_len or 32)

        self.backbone = None
        self.processor = None
        self.projection = None
        if self.audio_mode != 'precomputed':
            self.backbone = Wav2Vec2Model.from_pretrained(backbone) if pretrained else Wav2Vec2Model.from_pretrained(backbone)
            self.processor = Wav2Vec2Processor.from_pretrained(backbone)
            _freeze_pretrained_transformer(self.backbone)
            backbone_dim = self.backbone.config.hidden_size
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

        self.temporal_encoder = TemporalNpyEncoder(
            input_dim=self.precomputed_feature_dim,
            output_dim=output_dim,
            hidden_dim=temporal_hidden_dim,
            max_seq_len=self.max_seq_len,
            num_layers=1,
            dropout=0.1,
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(self.precomputed_feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, audio_waveform):
        """
        前向传播，提取音频特征
        
        参数：
            audio_waveform (torch.Tensor):
                - (B, T) 原始波形
                - (B, T, F) 预提取声学特征（如 COVAREP，F=74）
        
        返回：
            features (torch.Tensor): (B, output_dim)
        """
        if audio_waveform is None:
            return None

        # 预提取特征序列 (B, T, F) — 返回时序 (B, T', output_dim) 供融合对齐
        if audio_waveform.dim() == 3:
            if audio_waveform.shape[-1] != self.precomputed_feature_dim:
                raise ValueError(
                    f"precomputed audio feature dim {audio_waveform.shape[-1]} "
                    f"!= expected {self.precomputed_feature_dim}"
                )
            x = torch.nan_to_num(
                audio_waveform.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            # MOSEI COVAREP: stable mean-pool + MLP (BiLSTM backward was NaN-prone).
            if self.audio_mode == "precomputed":
                pooled = x.mean(dim=1)
                return self.feature_projection(pooled)
            seq_feats, pooled = self.temporal_encoder(x, return_pooled=True)
            return seq_feats if seq_feats.shape[1] > 1 else pooled

        # 原始波形 (B, T)
        if audio_waveform.dim() == 2:
            if self.backbone is None:
                return None
            with torch.no_grad():
                outputs = self.backbone(audio_waveform)
                hidden_states = outputs.last_hidden_state
            features = self.projection(hidden_states)
            return torch.mean(features, dim=1)

        return None


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
    def __init__(
        self,
        backbone='bert-base-uncased',
        pretrained=True,
        output_dim=512,
        unfreeze_encoder_layers: int = 0,
    ):
        """
        初始化文本特征提取器
        
        参数说明：
            backbone (str): BERT模型名称，默认'bert-base-uncased'（不区分大小写）
            pretrained (bool): 是否使用预训练权重，建议设为True
            output_dim (int): 输出特征维度，默认512
        """
        super(TextFeatureExtractor, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.unfreeze_encoder_layers = max(0, int(unfreeze_encoder_layers or 0))
        _freeze_pretrained_transformer(self.backbone, unfreeze_last_n=self.unfreeze_encoder_layers)
        
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
        bert_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if bert_trainable:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            # RoBERTa 等模型无 pooler；使用 [CLS] / 首 token 表示
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 投影到统一维度
        features = self.projection(pooled_output)  # (B, output_dim=512)
        
        return features

