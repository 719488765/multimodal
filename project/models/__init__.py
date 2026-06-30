"""
多模态驾驶员情绪分析模型
集成了多篇顶会论文的优秀方法
"""

from .feature_extractors import (
    VideoFeatureExtractor,
    AudioFeatureExtractor,
    PhysiologicalFeatureExtractor,
    TextFeatureExtractor
)

from .attention_modules import (
    MultiHeadSelfAttention,
    CrossModalAttention,
    TemporalAttention,
    MultimodalFusion
)

from .emotion_shift import (
    EmotionShiftAwareness,
    EmotionShiftFusion
)

from .leader_follower_attention import (
    LeaderFollowerAttention,
    BidirectionalLeaderFollower,
    MultimodalLeaderFollowerFusion
)

from .two_stage_fusion import (
    GraphAttentionLayer,
    TwoStageFusion
)

from .functional_correlation import (
    FunctionalMaximumCorrelation,
    MultimodalCorrelationLoss
)

from .multimodal_model import MultimodalEmotionModel

__all__ = [
    # 特征提取器
    'VideoFeatureExtractor',
    'AudioFeatureExtractor',
    'PhysiologicalFeatureExtractor',
    'TextFeatureExtractor',
    # 基础注意力模块
    'MultiHeadSelfAttention',
    'CrossModalAttention',
    'TemporalAttention',
    'MultimodalFusion',
    # 情感转变感知（CFN-ESA）
    'EmotionShiftAwareness',
    'EmotionShiftFusion',
    # 领导-跟随注意力（Continuous Emotion Recognition）
    'LeaderFollowerAttention',
    'BidirectionalLeaderFollower',
    'MultimodalLeaderFollowerFusion',
    # 两阶段融合（GA2MIF）
    'GraphAttentionLayer',
    'TwoStageFusion',
    # 功能最大相关（MFMC）
    'FunctionalMaximumCorrelation',
    'MultimodalCorrelationLoss',
    # 完整模型
    'MultimodalEmotionModel'
]

