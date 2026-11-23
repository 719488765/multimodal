#!/bin/bash
set -u
cd /mnt/sda1/lizhichun_24/code/multimodal/project
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
PY=/home/lizhichun_24/.conda/envs/myenv310/bin/python
LOG=$PWD/logs_e2
mkdir -p "$LOG" checkpoints_e2 outputs_e2
for cfg in f4_es_noseq_meld g2_es_seq_noprev_meld g3_es_seq_prev_meld f1_standard_meld f2_lf_text_meld f3_two_stage_meld; do
  echo "=== start $cfg $(date '+%F %T') ===" >> "$LOG/run_all.log"
  "$PY" scripts/train.py --config "config/e2/${cfg}.yaml" --mode finetune --dataset meld >> "$LOG/${cfg}.log" 2>&1
  rc=$?
  echo "=== done  $cfg rc=$rc $(date '+%F %T') ===" >> "$LOG/run_all.log"
done
echo "ALL_DONE $(date '+%F %T')" >> "$LOG/run_all.log"