"""MultimodalDataset custom collate: variable-length video/audio and MOSEI precomputed features."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def _is_precomputed_audio(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.size(-1) <= 256


def _pad_videos(videos: List[torch.Tensor]) -> torch.Tensor:
    if not videos:
        raise ValueError("empty video list")
    if videos[0].dim() == 4:
        return torch.stack(videos, dim=0)
    if videos[0].dim() == 2:
        return pad_sequence(videos, batch_first=True)
    raise ValueError(f"unsupported video dim {videos[0].dim()}")


def _collate_audio(audios: List[torch.Tensor]):
    """Return (waveform_batch|None, precomputed_batch|None)."""
    wave_items: List[torch.Tensor] = []
    feat_items: List[torch.Tensor] = []
    for a in audios:
        if a.dim() == 1:
            wave_items.append(a)
        elif _is_precomputed_audio(a):
            feat_items.append(a)
        elif a.dim() == 2 and a.size(-1) > 256:
            wave_items.append(a.reshape(-1))
        else:
            raise ValueError(f"unsupported audio shape {tuple(a.shape)}")

    audio_wave = pad_sequence(wave_items, batch_first=True) if wave_items else None
    audio_pre = pad_sequence(feat_items, batch_first=True) if feat_items else None
    return audio_wave, audio_pre


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = [b["video"] for b in batch]
    audio_wave, audio_pre = _collate_audio([b["audio"] for b in batch])
    out: Dict[str, Any] = {
        "video": _pad_videos(videos),
        "physiological": torch.stack([b["physiological"] for b in batch], dim=0),
        "text_input_ids": torch.stack([b["text_input_ids"] for b in batch], dim=0),
        "text_attention_mask": torch.stack([b["text_attention_mask"] for b in batch], dim=0),
        "emotion_label": torch.tensor([b["emotion_label"] for b in batch], dtype=torch.long),
        "emotion_dimensions": torch.stack([b["emotion_dimensions"] for b in batch], dim=0),
        "sample_id": [b.get("sample_id", "") for b in batch],
        "dataset_id": torch.tensor([b.get("dataset_id", -1) for b in batch], dtype=torch.long),
    }

    if audio_wave is not None and audio_pre is not None:
        bsz = len(batch)
        max_len = audio_wave.size(1)
        max_t = audio_pre.size(1)
        feat_dim = audio_pre.size(2)
        merged_wave = torch.zeros(bsz, max_len, dtype=audio_wave.dtype)
        merged_pre = torch.zeros(bsz, max_t, feat_dim, dtype=audio_pre.dtype)
        wi = fi = 0
        for i, a in enumerate([b["audio"] for b in batch]):
            if (a.dim() == 1 or a.dim() == 2) and a.size(-1) > 256:
                merged_wave[i, : audio_wave.size(1)] = audio_wave[wi]
                wi += 1
            else:
                merged_pre[i, : audio_pre.size(1), :] = audio_pre[fi]
                fi += 1
        out["audio"] = merged_wave
        out["audio_precomputed"] = merged_pre
    elif audio_pre is not None:
        out["audio"] = audio_pre
        out["audio_precomputed"] = audio_pre
    else:
        out["audio"] = audio_wave

    if batch and "context_text_input_ids" in batch[0]:
        out["context_text_input_ids"] = torch.stack(
            [b["context_text_input_ids"] for b in batch], dim=0
        )
        out["context_text_attention_mask"] = torch.stack(
            [b["context_text_attention_mask"] for b in batch], dim=0
        )
    return out
