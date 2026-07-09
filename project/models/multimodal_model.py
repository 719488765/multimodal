"""
完整的多模态情绪分析模型
集成了多篇顶会论文的优秀方法：
- CFN-ESA: 情感转变感知机制
- MFMC: 多模态功能最大相关
- Continuous Emotion Recognition: 领导-跟随注意力
- GA2MIF: 两阶段融合策略
"""

import torch
import torch.nn as nn
from .feature_extractors import (
    VideoFeatureExtractor,
    AudioFeatureExtractor,
    PhysiologicalFeatureExtractor,
    TextFeatureExtractor
)
from .attention_modules import MultimodalFusion
from .emotion_shift import EmotionShiftFusion
from .leader_follower_attention import MultimodalLeaderFollowerFusion
from .two_stage_fusion import TwoStageFusion
from .functional_correlation import MultimodalCorrelationLoss


class MultimodalEmotionModel(nn.Module):
    """
    多模态驾驶员情绪分析模型
    整合视频、语音、生理信号和文本四种模态
    """
    def __init__(self, config):
        super(MultimodalEmotionModel, self).__init__()
        
        self.config = config
        model_config = config['model']
        modalities = model_config.get('modalities', {})
        self.use_video = bool(modalities.get('use_video', True))
        self.use_audio = bool(modalities.get('use_audio', True))
        self.use_physiological = bool(modalities.get('use_physiological', True))
        self.use_text = bool(modalities.get('use_text', True))
        
        # 特征提取器
        self.video_extractor = VideoFeatureExtractor(
            backbone=model_config['video']['backbone'],
            pretrained=model_config['video']['pretrained'],
            feature_dim=model_config['video']['feature_dim'],
            output_dim=model_config['video']['output_dim']
        )
        
        self.audio_extractor = AudioFeatureExtractor(
            backbone=model_config['audio']['backbone'],
            pretrained=model_config['audio']['pretrained'],
            output_dim=model_config['audio']['output_dim']
        )
        
        self.physiological_extractor = PhysiologicalFeatureExtractor(
            input_dim=model_config['physiological']['input_dim'],
            hidden_dim=model_config['physiological']['hidden_dim'],
            num_layers=model_config['physiological']['num_layers'],
            output_dim=model_config['physiological']['output_dim'],
            use_lstm=model_config['physiological']['use_lstm']
        )
        
        self.text_extractor = TextFeatureExtractor(
            backbone=model_config['text']['backbone'],
            pretrained=model_config['text']['pretrained'],
            output_dim=model_config['text']['output_dim'],
            unfreeze_encoder_layers=model_config['text'].get('unfreeze_encoder_layers', 0),
        )
        
        # 注意力融合模块（支持多种融合策略）
        attention_config = model_config['attention']
        output_config = model_config['output']
        fusion_strategy = attention_config.get('fusion_strategy', 'standard')  # standard, emotion_shift, leader_follower, two_stage
        
        if fusion_strategy == 'emotion_shift':
            # CFN-ESA: 情感转变感知融合
            self.fusion_module = EmotionShiftFusion(
                hidden_dim=attention_config['hidden_dim'],
                num_emotion_classes=output_config['emotion_classes'],
                num_heads=attention_config['num_heads'],
                dropout=attention_config['dropout'],
                leader_modal=attention_config.get('leader_modal', 'text'),
            )
            self.use_emotion_shift = True
        elif fusion_strategy == 'leader_follower':
            # Continuous Emotion Recognition: 领导-跟随注意力
            self.fusion_module = MultimodalLeaderFollowerFusion(
                hidden_dim=attention_config['hidden_dim'],
                num_heads=attention_config['num_heads'],
                leader_modal=attention_config.get('leader_modal', 'text'),
                dropout=attention_config['dropout']
            )
            self.use_emotion_shift = False
        elif fusion_strategy == 'two_stage':
            # GA2MIF: 两阶段融合
            self.fusion_module = TwoStageFusion(
                hidden_dim=attention_config['hidden_dim'],
                num_heads=attention_config['num_heads'],
                num_gat_layers=attention_config.get('num_gat_layers', 2),
                num_fusion_layers=attention_config['num_layers'],
                dropout=attention_config['dropout'],
                leader_modal=attention_config.get('leader_modal', 'text'),
            )
            self.use_emotion_shift = False
        else:
            # 标准融合（原始方法）
            self.fusion_module = MultimodalFusion(
                hidden_dim=attention_config['hidden_dim'],
                num_heads=attention_config['num_heads'],
                num_layers=attention_config['num_layers'],
                dropout=attention_config['dropout']
            )
            self.use_emotion_shift = False
        
        self.fusion_strategy = fusion_strategy
        
        # 功能最大相关损失（用于预训练，可选）
        if model_config.get('use_fmc_loss', False):
            self.fmc_loss = MultimodalCorrelationLoss(
                hidden_dim=attention_config['hidden_dim'],
                num_projections=attention_config.get('fmc_projections', 64),
                weight=attention_config.get('fmc_weight', 0.1)
            )
        else:
            self.fmc_loss = None
        
        # 输出层
        hidden_dim = attention_config['hidden_dim']
        
        # 情绪分类器（离散情绪类别）
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_config['emotion_classes'])
        )
        
        # 情绪强度回归器（连续情绪维度：效价和唤醒度）
        self.emotion_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_config['emotion_dimensions'])
        )
        
        # 情绪趋势预测器（可选）
        if output_config.get('use_trend_prediction', False):
            self.trend_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, output_config['emotion_classes'])
            )
        else:
            self.trend_predictor = None

    def _infer_batch_size(self, video, audio, physiological, text_input_ids, audio_precomputed):
        for tensor in (video, audio, physiological, text_input_ids, audio_precomputed):
            if tensor is not None:
                return tensor.shape[0]
        return 1

    def _zero_feature(self, batch_size, output_dim, device):
        return torch.zeros(batch_size, output_dim, device=device)

    def _build_active_mask(self, video, audio, physiological, text_input_ids, audio_precomputed):
        return {
            'video': self.use_video and video is not None,
            'audio': self.use_audio and (audio is not None or audio_precomputed is not None),
            'physiological': self.use_physiological and physiological is not None,
            'text': self.use_text and text_input_ids is not None,
        }
        
    def forward(self, video=None, audio=None, physiological=None, text=None,
                text_input_ids=None, text_attention_mask=None, previous_emotions=None,
                audio_precomputed=None, dataset_ids=None, context_text_input_ids=None,
                context_text_attention_mask=None, return_fmc_loss=False, return_domain_logits=False, **kwargs):
        """
        前向传播
        
        Args:
            video: (B, T, C, H, W) 或 (B, C, H, W) - 视频帧
            audio: (B, T) - 音频波形
            physiological: (B, T, input_dim) - 生理信号
            text: 文本字符串列表（用于tokenization）
            text_input_ids: (B, seq_len) - 文本token IDs（如果已tokenize）
            text_attention_mask: (B, seq_len) - 文本注意力掩码
            previous_emotions: (B, T_prev, num_classes) - 之前时间步的情感状态（用于情感转变感知）
            return_fmc_loss: bool - 是否返回功能最大相关损失（用于预训练）
        Returns:
            outputs: dict包含
                - emotion_logits: (B, num_classes) - 情绪分类logits
                - emotion_probs: (B, num_classes) - 情绪分类概率
                - emotion_dimensions: (B, 2) - 效价和唤醒度
                - trend_prediction: (B, num_classes) - 情绪趋势预测（如果启用）
                - shift_weights: (B,) - 情感转变权重（如果使用情感转变感知）
                - fmc_loss: 标量 - 功能最大相关损失（如果启用）
        """
        # 特征提取
        features = {}
        device = next(self.parameters()).device
        batch_size = self._infer_batch_size(
            video, audio, physiological, text_input_ids, audio_precomputed
        )
        active_mask = self._build_active_mask(
            video, audio, physiological, text_input_ids, audio_precomputed
        )

        if self.use_video and video is not None:
            features['video'] = self.video_extractor(video)
        else:
            features['video'] = self._zero_feature(
                batch_size, self.config['model']['video']['output_dim'], device
            )

        if self.use_audio and audio_precomputed is not None:
            features['audio'] = self.audio_extractor(audio_precomputed)
        elif self.use_audio and audio is not None:
            features['audio'] = self.audio_extractor(audio)
        else:
            features['audio'] = self._zero_feature(
                batch_size, self.config['model']['audio']['output_dim'], device
            )

        if self.use_physiological and physiological is not None:
            features['physiological'] = self.physiological_extractor(physiological)
        else:
            features['physiological'] = self._zero_feature(
                batch_size, self.config['model']['physiological']['output_dim'], device
            )

        if self.use_text and text_input_ids is not None:
            features['text'] = self.text_extractor(text_input_ids, text_attention_mask)
        else:
            features['text'] = self._zero_feature(
                batch_size, self.config['model']['text']['output_dim'], device
            )
        
        # 多模态融合（根据策略选择不同的融合方法）
        if self.use_emotion_shift:
            # 情感转变感知融合
            fused_features, emotion_logits_shift, shift_weights = self.fusion_module(
                features['video'],
                features['audio'],
                features['physiological'],
                features['text'],
                previous_emotions=previous_emotions,
                active_mask=active_mask,
            )
        else:
            # 标准融合、领导-跟随或两阶段融合
            fused_features = self.fusion_module(
                features['video'],
                features['audio'],
                features['physiological'],
                features['text'],
                active_mask=active_mask,
            )
            emotion_logits_shift = None
            shift_weights = None
        
        # 输出预测：分类始终走 fused_features + emotion_classifier（避免 shift 内部 logits NaN 塌缩）
        fused_features = torch.nan_to_num(
            fused_features, nan=0.0, posinf=1e4, neginf=-1e4
        )
        emotion_logits = self.emotion_classifier(fused_features)
        
        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        emotion_dimensions = self.emotion_regressor(fused_features)
        
        outputs = {
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'emotion_dimensions': emotion_dimensions,
            'fused_features': fused_features
        }
        
        # 添加情感转变权重（如果使用情感转变感知）
        if shift_weights is not None:
            outputs['shift_weights'] = shift_weights
        
        # 情绪趋势预测（如果启用）
        if self.trend_predictor is not None:
            trend_prediction = self.trend_predictor(fused_features)
            outputs['trend_prediction'] = trend_prediction
        
        # 功能最大相关损失（用于预训练）
        if return_fmc_loss and self.fmc_loss is not None:
            fmc_loss, correlations = self.fmc_loss(
                features['video'],
                features['audio'],
                features['physiological'],
                features['text']
            )
            outputs['fmc_loss'] = fmc_loss
            outputs['correlations'] = correlations
        
        return outputs
    
    def get_emotion_label(self, emotion_probs):
        """
        从概率分布中获取情绪标签
        
        Args:
            emotion_probs: (B, num_classes) - 情绪概率分布
        Returns:
            emotion_labels: (B,) - 情绪类别索引
        """
        return torch.argmax(emotion_probs, dim=-1)

