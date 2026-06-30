# P2 ES 抗过拟合重训验收 (2026-06-25 20:09:17)

**总体**: PASS

| Job | Dataset | Pass | Best F1@ep | Last F1 | Epochs | Issues |
|-----|---------|------|------------|---------|--------|--------|
| F_M_ES | meld | ✓ | 0.6109 @ 3 | 0.4522 | 16 | — |
| F_C_ES | crema | ✓ | 0.5412 @ 34 | 0.5289 | 47 | — |

**F_M_ES warnings**: post-best collapse: best=0.6109@ep3 last=0.4522 (drop=0.1586) (expected with early stopping)

**F_C_ES warnings**: F1=0.5412 below pre-retrain peak 0.579 (curve may still be cleaner)
