# SDAVT v3 R4 实验结果汇总

*生成时间: 2026-06-30 18:10:56*

## 队列概览

| Status | Count |
|--------|-------|
| done | 41 |

## p0_fix

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| F_C_TS | crema | done | 0.2579 @ 15 | 0.2836 | ✓ | `SDAVT_R4_F_C_TS_20260622_140751` |
| F_M_STD | meld | done | 0.4595 @ 4 | 0.4991 | ✓ | `SDAVT_R4_F_M_STD_20260622_140751` |
| F_M_TS | meld | done | 0.3826 @ 9 | 0.4368 | ✓ | `SDAVT_R4_F_M_TS_20260622_163603` |
| F_O_LFT | mosei | done | 0.5980 @ 7 | 0.7162 | ✓ | `SDAVT_R4_F_O_LFT_20260622_214330` |
| F_O_STD | mosei | done | 0.5988 @ 7 | 0.7162 | ✓ | `SDAVT_R4_F_O_STD_20260622_222125` |
| F_O_TS | mosei | done | 0.5978 @ 0 | 0.7162 | ✓ | `SDAVT_R4_F_O_TS_20260622_232044` |

## p1_baseline

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| R4_B_C0 | crema | done | 0.5889 @ 19 | 0.5874 | ✓ | `SDAVT_R4_R4_B_C0_20260623_005627` |
| R4_B_M1 | meld | done | 0.5680 @ 3 | 0.5966 | △ | `SDAVT_R4_R4_B_M1_20260623_005627` |
| R4_B_O0 | mosei | done | 0.6792 @ 12 | 0.7269 | ✓ | `SDAVT_R4_R4_B_O0_20260623_032138` |

## p2_fusion

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| F_C_ES | crema | done | 0.5412 @ 34 | 0.5470 | ✓ | `SDAVT_R4_F_C_ES` |
| F_C_LFA | crema | done | 0.2978 @ 26 | 0.3065 | ✓ | `SDAVT_R4_F_C_LFA_20260624_070022` |
| F_C_LFT | crema | done | 0.3081 @ 15 | 0.3320 | ✓ | `SDAVT_R4_F_C_LFT_20260624_073903` |
| F_C_STD | crema | done | 0.2405 @ 14 | 0.2594 | ✓ | `SDAVT_R4_F_C_STD_20260624_081739` |
| F_C_TS | crema | done | 0.2436 @ 29 | 0.3051 | ✓ | `SDAVT_R4_F_C_TS_20260624_085905` |
| F_M_ES | meld | done | 0.6109 @ 3 | 0.6245 | ✓ | `SDAVT_R4_F_M_ES` |
| F_M_LFA | meld | done | 0.4562 @ 3 | 0.4937 | ✓ | `SDAVT_R4_F_M_LFA_20260623_210443` |
| F_M_LFT | meld | done | 0.4207 @ 5 | 0.4486 | ✓ | `SDAVT_R4_F_M_LFT_20260623_225301` |
| F_M_STD | meld | done | 0.4447 @ 5 | 0.4973 | ✓ | `SDAVT_R4_F_M_STD_20260624_005729` |
| F_M_TS | meld | done | 0.3682 @ 12 | 0.4314 | ✓ | `SDAVT_R4_F_M_TS_20260624_030111` |
| F_O_ES | mosei | done | 0.6792 @ 12 | 0.7269 | ✓ | `SDAVT_R4_F_O_ES_20260624_101647` |
| F_O_LFT | mosei | done | 0.5980 @ 7 | 0.7162 | ✓ | `SDAVT_R4_F_O_LFT_20260624_110504` |
| F_O_STD | mosei | done | 0.5988 @ 7 | 0.7162 | ✓ | `SDAVT_R4_F_O_STD_20260624_115119` |
| F_O_TS | mosei | done | 0.5978 @ 0 | 0.7162 | ✓ | `SDAVT_R4_F_O_TS_20260624_123518` |

## p3_c3

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| C3_C1_baseline | crema | done | 0.5336 @ 29 | 0.5417 | △ | `SDAVT_R4_C3_C1_baseline_20260625_200937` |
| C3_C2_w2v_large | crema | done | 0.5629 @ 31 | 0.5672 | △ | `SDAVT_R4_C3_C2_w2v_large_20260626_004150` |
| C3_C3_focal | crema | done | 0.5526 @ 41 | 0.5565 | △ | `SDAVT_R4_C3_C3_focal_20260626_043125` |

## p3_m3

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| M3_M0_baseline | meld | done | 0.6080 @ 3 | 0.6218 | ✓ | `SDAVT_R4_M3_M0_baseline_20260625_200937` |
| M3_M1_roberta | meld | done | 0.6823 @ 17 | 0.6968 | ✓ | `SDAVT_R4_M3_M1_roberta` |
| M3_M2_w2v_large | meld | done | 0.5572 @ 3 | 0.6020 | △ | `SDAVT_R4_M3_M2_w2v_large_20260625_233919` |
| M3_M3_uniform | meld | done | 0.6105 @ 3 | 0.6245 | ✓ | `SDAVT_R4_M3_M3_uniform_20260626_031222` |
| M3_M4_focal | meld | done | 0.6079 @ 3 | 0.6209 | ✓ | `SDAVT_R4_M3_M4_focal_20260626_062916` |
| M3_M5_context | meld | done | 0.5725 @ 3 | 0.5912 | △ | `SDAVT_R4_M3_M5_context_20260626_073046` |
| M3_M6_moddrop | meld | done | 0.6079 @ 6 | 0.6245 | ✓ | `SDAVT_R4_M3_M6_moddrop_20260626_113831` |
| M3_M7_combo | meld | done | 0.6957 @ 31 | 0.7121 | ✓ | `SDAVT_R4_M3_M7_combo` |

## p4_modal

| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |
|-----|---------|--------|------------|-------------|--------|-----|
| R4_A_M_A | meld | done | 0.4659 @ 31 | 0.5090 | ✓ | `SDAVT_R4_R4_A_M_A_20260626_141343` |
| R4_A_M_AT | meld | done | 0.6232 @ 32 | 0.6399 | ✓ | `SDAVT_R4_R4_A_M_AT_20260627_112907` |
| R4_A_M_AV | meld | done | 0.4336 @ 2 | 0.4865 | ✓ | `SDAVT_R4_R4_A_M_AV_20260627_050317` |
| R4_A_M_AVT | meld | done | 0.6080 @ 3 | 0.6218 | ✓ | `SDAVT_R4_R4_A_M_AVT_20260627_191824` |
| R4_A_M_T | meld | done | 0.6262 @ 13 | 0.6444 | ✓ | `SDAVT_R4_R4_A_M_T_20260627_031444` |
| R4_A_M_V | meld | done | 0.3840 @ 8 | 0.4377 | ✓ | `SDAVT_R4_R4_A_M_V_20260626_192132` |
| R4_A_M_VT | meld | done | 0.6120 @ 3 | 0.6245 | ✓ | `SDAVT_R4_R4_A_M_VT_20260627_122022` |

---

*本文档由 `scripts/build_sdavt_r4_report.py` 自动生成。*
