# Author: AI
# Date: 2026-03-31
# Description: 基线与消融重跑实验总清单（含命令、TensorBoard与日志归档流程）

## 0. 本次已完成的预处理（已执行）

1) 已将历史日志迁移到：

- `logs_archived/history_before_rerun_20260331_195746/`

2) 当前 `logs/` 仅保留：

- `logs/archived/`

3) 训练脚本已增强：

- `scripts/train.py` 新增 `best_f1` 检查点保存：`checkpoint_pretrain_best_f1.pth`
- 原有 `best_val_loss` 保存逻辑保持不变：`checkpoint_pretrain_best.pth`

4) 重跑配置已自动生成完毕（可直接用）：

- 目录：`config/rerun/`
- 已生成文件：
  - `config_AT_noDA.yaml`
  - `config_AT_DA.yaml`
  - `config_VT_noDA.yaml`
  - `config_AVT_noDA.yaml`
  - `config_AVT_DA.yaml`
  - `config_AVT_noDA_emotion_shift.yaml`
  - `config_text_only.yaml`
  - `config_audio_only.yaml`
  - `config_video_only.yaml`
  - `config_AVT_DA_w002.yaml`
  - `config_AVT_DA_w005.yaml`
  - `config_AVT_DA_w010.yaml`
- 以上配置已统一改到：
  - `paths.checkpoint_dir: "checkpoints_rerun/"`
  - `paths.log_dir: "logs_rerun/"`
  - `paths.output_dir: "outputs_rerun/"`
- `experiment.name` 已统一改为 `RERUN_*` 前缀。

5) 新 TensorBoard 已启动（重跑专用）：

- 服务地址：`http://0.0.0.0:6010/`
- 监控目录：`logs_rerun/`

---

## 1. 统一规范（本轮重跑必须遵守）

### 1.1 环境与目录

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv310
mkdir -p checkpoints_rerun logs_rerun outputs_rerun
```

### 1.2 TensorBoard 新看板（只看重跑数据）

> 建议本轮重跑全部写入 `logs_rerun/`，避免与旧历史 run 混在一起。

```bash
tensorboard --logdir logs_rerun --host 0.0.0.0 --port 6010
```

本地端口转发（Windows PowerShell）：

```powershell
ssh -N -L 6010:localhost:6010 -p 1022 lizhichun_24@49.233.89.203
```

浏览器访问：

- `http://localhost:6010`

### 1.3 为重跑创建专用配置目录（避免污染原配置）

```bash
mkdir -p config/rerun
cp config/config_AT_DA.yaml config/rerun/
cp config/config_AVT_noDA.yaml config/rerun/
cp config/config_AVT_DA.yaml config/rerun/
cp config/config_AVT_noDA_emotion_shift.yaml config/rerun/
cp config/config_VT_noDA.yaml config/rerun/
cp config/config_text_only.yaml config/rerun/
cp config/config_audio_only.yaml config/rerun/
cp config/config_video_only.yaml config/rerun/
```

将重跑配置统一改为新的输出路径（逐个文件执行）：

```bash
sed -i 's|checkpoint_dir: "checkpoints/"|checkpoint_dir: "checkpoints_rerun/"|g' config/rerun/*.yaml
sed -i 's|log_dir: "logs/"|log_dir: "logs_rerun/"|g' config/rerun/*.yaml
sed -i 's|output_dir: "outputs/"|output_dir: "outputs_rerun/"|g' config/rerun/*.yaml
```

---

## 2. 先做冒烟验证（必做，10-15 epoch）

目标：先确认本轮代码（含标签映射修复）无异常，再进入50 epoch长跑。

### 2.1 AVT_noDA 冒烟

```bash
python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain
```

> 冒烟阶段可临时把 `num_epochs` 改为 `10~15`，通过后恢复 `50`。

### 2.2 AVT_DA 冒烟

```bash
python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain
```

---

## 3. 正式重跑主线（论文主表核心）

> 推荐使用 `tmux`，每个实验一个会话。

### 3.1 关键会话命名

```bash
tmux new -s rerun_at_noda
tmux new -s rerun_at_da
tmux new -s rerun_vt_noda
tmux new -s rerun_avt_noda
tmux new -s rerun_avt_da
tmux new -s rerun_avt_es
```

### 3.2 必跑实验顺序（先基线再消融）

1. `AT_noDA`（基线锚点）  
2. `AT_DA`（DA 对照）  
3. `VT_noDA`（视频引入中间态）  
4. `AVT_noDA`（全模态主结果）  
5. `AVT_DA`（全模态 DA 对照）  
6. `AVT_noDA_emotion_shift`（融合策略对照）

### 3.3 命令清单

#### [1] AT_noDA

> 使用 `config/rerun/config.yaml` 不够稳妥，建议从 `config_AT_DA.yaml` 复制一份 `config_AT_noDA.yaml` 后关闭 DA：

```bash
cp config/config_AT_DA.yaml config/rerun/config_AT_noDA.yaml
sed -i 's/enabled: true/enabled: false/g' config/rerun/config_AT_noDA.yaml
sed -i 's/use_domain_adaptation: true/use_domain_adaptation: false/g' config/rerun/config_AT_noDA.yaml
```

执行：

```bash
python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain
```

#### [2] AT_DA

```bash
python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain
```

#### [3] VT_noDA

```bash
python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain
```

#### [4] AVT_noDA

```bash
python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain
```

#### [5] AVT_DA

```bash
python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain
```

#### [6] AVT_noDA_emotion_shift

```bash
python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain
```

---

## 4. 扩展实验（建议补充，提升论文说服力）

### 4.1 模态组合维度

- `T_only`：

```bash
python3 scripts/train.py --config config/rerun/config_text_only.yaml --mode pretrain
```

- `A_only`：

```bash
python3 scripts/train.py --config config/rerun/config_audio_only.yaml --mode pretrain
```

- `V_only`：

```bash
python3 scripts/train.py --config config/rerun/config_video_only.yaml --mode pretrain
```

### 4.2 域适应维度（建议加1组调参）

在 `config/rerun/config_AVT_DA.yaml` 上做 3 组：

- `domain_loss_weight = 0.02`
- `domain_loss_weight = 0.05`
- `domain_loss_weight = 0.10`（当前默认）

建议复制三个配置并分别运行：

```bash
cp config/rerun/config_AVT_DA.yaml config/rerun/config_AVT_DA_w002.yaml
cp config/rerun/config_AVT_DA.yaml config/rerun/config_AVT_DA_w005.yaml
cp config/rerun/config_AVT_DA.yaml config/rerun/config_AVT_DA_w010.yaml
```

然后分别手工修改 `domain_loss_weight` 后执行：

```bash
python3 scripts/train.py --config config/rerun/config_AVT_DA_w002.yaml --mode pretrain
python3 scripts/train.py --config config/rerun/config_AVT_DA_w005.yaml --mode pretrain
python3 scripts/train.py --config config/rerun/config_AVT_DA_w010.yaml --mode pretrain
```

### 4.3 融合策略维度

至少完成：

- `AVT_noDA_standard` vs `AVT_noDA_emotion_shift`

可选扩展（资源允许）：

- `leader_follower`
- `two_stage`

### 4.4 训练范式维度（from_pretrain vs scratch）

保留你现有 CREMA 路线，建议在本轮重跑后再补一组 MELD 或 MOSEI：

```bash
# from_pretrain（示意）
python3 scripts/train.py --config config/config_crema_finetune_from_pretrain.yaml --mode finetune --resume checkpoints_rerun/checkpoint_pretrain_best_f1.pth

# scratch（示意）
python3 scripts/train.py --config config/config_crema_finetune_from_scratch.yaml --mode finetune
```

---

## 5. 结果固化与统一评估（必须）

### 5.1 每个实验至少保留三类 checkpoint

- `checkpoint_pretrain_best.pth`（best val loss）
- `checkpoint_pretrain_best_f1.pth`（best val f1）
- `checkpoint_pretrain_epoch_49.pth`（last）

### 5.2 统一口径重算指标（写论文主表前必须执行）

```bash
conda run -n myenv310 python3 scripts/recompute_val_metrics.py --config <对应yaml> --checkpoint <best_or_last_ckpt> --split val --batch_size 2
```

建议每组至少重算：

1. `best_f1`
2. `best_loss`
3. `last`

---

## 6. 运行管理与故障恢复

### 6.1 tmux 常用

```bash
tmux ls
tmux attach -t <session_name>
# 退出但不停止：Ctrl+b 然后 d
```

### 6.2 断点续训

```bash
python3 scripts/train.py --config <yaml> --mode pretrain --resume checkpoints_rerun/checkpoint_pretrain_epoch_<N>.pth
```

### 6.3 日志健康检查

```bash
python3 scripts/check_media_health_dir.py --data_dir data
```

---

## 7. 论文主表建议字段（重跑完成后直接填）

- 实验名（run）
- 模态组合（T/A/V/AT/VT/AVT）
- 融合策略（standard/emotion_shift/...）
- 是否 DA
- 训练范式（pretrain/finetune；from_pretrain/scratch）
- Best Acc / Best F1 / Last Acc / Last F1
- 对应 checkpoint 路径
- 备注（异常、重跑原因）

---

## 8. 本清单执行优先级

1. 冒烟验证（AVT_noDA, AVT_DA）  
2. 六项主线必跑（AT_noDA → AT_DA → VT_noDA → AVT_noDA → AVT_DA → AVT_noDA_emotion_shift）  
3. 单模态补充（T/A/V-only）  
4. AVT_DA 权重调参（0.02/0.05/0.1）  
5. 统一重算 + 论文主表固化  

> 原则：先“可比主线完整”，再“扩展加分项”。

---

## 9. 一键启动主线 6 实验（tmux 脚本，已就绪）

已提供脚本：

- `scripts/start_rerun_mainline_tmux.sh`

### 9.1 一键启动命令

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_mainline_tmux.sh
```

该脚本会自动创建并启动以下会话：

- `rerun_at_noda`
- `rerun_at_da`
- `rerun_vt_noda`
- `rerun_avt_noda`
- `rerun_avt_da`
- `rerun_avt_es`

### 9.2 运行状态检查命令

```bash
tmux ls
tmux attach -t rerun_at_noda
```

### 9.3 如果你只想手动逐个开跑（不用脚本）

可继续使用本清单第 3 节的逐条命令。

---

## 10. 一键启动第 2 批扩展实验（tmux 脚本，已就绪）

已提供脚本：

- `scripts/start_rerun_expansion_tmux.sh`

### 10.1 一键启动命令

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_expansion_tmux.sh
```

该脚本会自动创建并启动以下会话：

- `rerun_t_only`
- `rerun_a_only`
- `rerun_v_only`
- `rerun_avt_da_w002`
- `rerun_avt_da_w005`
- `rerun_avt_da_w010`

### 10.2 运行状态检查命令

```bash
tmux ls
tmux attach -t rerun_t_only
```

### 10.3 推荐执行顺序（避免 GPU 资源冲突）

1. 先完成主线 6 组（第 9 节）  
2. 再执行第 2 批扩展组（本节）  
3. 若单卡资源有限，建议一次仅保留 1~2 个训练会话同时运行

---

## 10.5 扩展实验 GPU 感知启动（推荐）

脚本：

- `scripts/start_rerun_expansion_gpuaware.sh`

用法：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_expansion_gpuaware.sh auto
```

可选模式：

```bash
./scripts/start_rerun_expansion_gpuaware.sh single
./scripts/start_rerun_expansion_gpuaware.sh dual
```

说明：

- 与主线 GPU 感知脚本一致，自动根据显存空闲量选择单卡/双卡
- 会自动设置 `CUDA_VISIBLE_DEVICES`
- 默认阈值：两卡空闲显存均 >= 12000MB 时走 dual，否则 single

---

## 10.4 新增建议维度（已生成对应配置）

基于当前配置与历史结果，建议再补 3 个低成本高价值维度（已生成配置）：

1) **采样策略维度（proportional vs uniform）**  
- `config/rerun/config_AVT_noDA_uniform.yaml`  
- `config/rerun/config_AVT_DA_uniform.yaml`  

2) **学习率维度（1e-4 vs 5e-5）**  
- `config/rerun/config_AVT_noDA_lr5e5.yaml`  
- `config/rerun/config_AVT_DA_w005_lr5e5.yaml`  

3) **随机种子稳健性维度（seed=3407）**  
- `config/rerun/config_AVT_noDA_seed3407.yaml`  
- `config/rerun/config_AVT_DA_seed3407.yaml`  

建议执行顺序：先跑主线与第2批扩展，再跑上述新增 6 组对照。

---

## 11. 一键结果汇总（CSV + Markdown，对论文表格友好）

已提供脚本：

- `scripts/summarize_rerun_results.py`

功能：

- 自动扫描 `logs_rerun/*/metrics.csv`
- 汇总每个 run 的：
  - last 指标（acc / precision / recall / f1 / loss）
  - best_acc（对应 epoch）
  - best_f1（对应 epoch）
  - min_val_loss（对应 epoch）
- 输出到：
  - `outputs_rerun/rerun_results_summary.csv`
  - `outputs_rerun/rerun_results_summary.md`

执行命令：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python3 scripts/summarize_rerun_results.py
```

指定目录（可选）：

```bash
python3 scripts/summarize_rerun_results.py --logs_dir logs_rerun --output_csv outputs_rerun/rerun_results_summary.csv --output_md outputs_rerun/rerun_results_summary.md
```

> 建议每跑完 1~2 组实验就执行一次汇总，避免后期手动查表。

---

## 12. 自动生成论文主表初稿（CSV + Markdown）

已提供脚本：

- `scripts/build_paper_table_main.py`

功能：

- 从 `outputs_rerun/rerun_results_summary.csv` 自动解析 run 名称
- 生成论文主表初稿字段：
  - `modality`（T/A/V/AT/VT/AVT）
  - `da`（noDA / DA）
  - `fusion`（standard / emotion_shift / ...）
  - `training_regime`（pretrain / finetune）
  - `last_acc/last_f1`
  - `best_acc / best_f1` 及对应 epoch
- 输出到：
  - `outputs_rerun/paper_table_main.csv`
  - `outputs_rerun/paper_table_main.md`

执行命令：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python3 scripts/build_paper_table_main.py
```

推荐串联（先汇总再生成主表）：

```bash
python3 scripts/summarize_rerun_results.py
python3 scripts/build_paper_table_main.py
```

> 注意：论文最终数字建议仍以 `recompute_val_metrics.py` 复核后的关键 checkpoint 为准。

---

## 13. 自动重算关键 checkpoint 并回填主表（最终定稿用）

已提供脚本：

- `scripts/recompute_and_fill_paper_table.py`

功能：

- 读取：
  - `outputs_rerun/paper_table_main.csv`
  - `outputs_rerun/rerun_results_summary.csv`
- 自动定位每个 run 的关键 epoch：
  - `best_f1_epoch`
  - `best_acc_epoch`
  - `last_val_epoch`
- 自动调用 `scripts/recompute_val_metrics.py` 重算对应 checkpoint 指标
- 输出回填后的终稿表：
  - `outputs_rerun/paper_table_main_recomputed.csv`
  - `outputs_rerun/paper_table_main_recomputed.md`

执行命令：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python3 scripts/recompute_and_fill_paper_table.py
```

推荐完整链路（实验跑完后一条龙）：

```bash
python3 scripts/summarize_rerun_results.py
python3 scripts/build_paper_table_main.py
python3 scripts/recompute_and_fill_paper_table.py
```

> 注意：本脚本依赖 `checkpoints_rerun/<run>/checkpoint_pretrain_epoch_<N>.pth` 的目录结构，因此请使用当前已改造的 `train.py`（按 run 分目录保存 checkpoint）。

---

## 14. 关于 tmux 并发启动是否会互相干扰（重要）

可以同时启动多个 tmux 会话，但是否“互不干扰”取决于硬件资源：

1) **代码/日志层面**：  
- 现在已按 run 分目录保存 checkpoint（`checkpoints_rerun/<run>/...`），日志也在 `logs_rerun/<run>/...`，不会互相覆盖。  

2) **算力层面（GPU）**：  
- 同一块 GPU 上并行跑多个训练会话会竞争显存与算力，可能导致：  
  - OOM  
  - 训练显著变慢  
  - 曲线抖动加大（间接影响效果稳定性）  

3) **推荐策略**：  
- 单卡：一次运行 1 组（最多 2 组轻量实验）  
- 多卡：可按 GPU 绑定分配（`CUDA_VISIBLE_DEVICES`）并行  
- 论文主结果（AT/VT/AVT 主线）建议**串行跑**，保证可比性与稳定性

---

## 15. 结合你服务器真实配置的参数与启动建议（已实测）

### 15.1 服务器真实配置结论

- GPU：**2 x NVIDIA RTX 4090（24GB）**
- PyTorch 可见设备数：`torch.cuda.device_count() = 2`
- 但采样时两张卡利用率接近满载（约 99%~100%，空闲显存约 4.6~5.6GB），说明存在并发任务

结论：

- 你是**双卡服务器**，但是否能并行重跑取决于当时显存空闲量
- 若显存紧张，应使用单卡串行策略（更稳、更少 OOM）

### 15.2 已提供 GPU 感知启动脚本（主线）

脚本：

- `scripts/start_rerun_mainline_gpuaware.sh`

用法：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_mainline_gpuaware.sh auto
```

可选模式：

```bash
./scripts/start_rerun_mainline_gpuaware.sh single
./scripts/start_rerun_mainline_gpuaware.sh dual
```

说明：

- `auto`：当两卡空闲显存都高于阈值（默认 12000MB）自动 dual，否则 single
- 启动时自动设置 `CUDA_VISIBLE_DEVICES`

### 15.3 为提升指标已做的配置优化（已落地）

1) `train.py` 新增 `training.seed` 支持（可复现）  
2) `train.py` 让 `weight_decay` 真正生效（之前配置值未用于优化器）  
3) `train.py` 新增 `gradient_accumulation_steps` 支持（小 batch 稳定优化）  
4) `config/rerun/config_AVT*.yaml` 已补：
   - `seed: 3407`
   - `gradient_accumulation_steps: 2`
5) 新增 6 组扩展维度配置（uniform / lr5e-5 / seed3407），用于低成本提效探索

### 15.4 推荐参数（按单卡/双卡）

单卡（稳妥，建议默认）：

- `batch_size: 1`
- `gradient_accumulation_steps: 2`
- `learning_rate: 1e-4`（DA 可试 `5e-5`）
- `num_workers: 4`（若 IO 成瓶颈可试 6~8）

双卡（资源空闲时）：

- 两个独立实验分别绑卡运行（不是 DDP）
- 每卡仍建议 `batch_size: 1`，优先用并行实验提升吞吐，不急于放大单实验 batch

### 15.5 已补齐 GPU 感知启动脚本（扩展组 + 新增维度组）

扩展组脚本（T/A/V-only + DA 权重网格）：

- `scripts/start_rerun_expansion_gpuaware.sh`

新增维度脚本（uniform/lr/seed）：

- `scripts/start_rerun_extra_dims_gpuaware.sh`

统一用法（推荐 `auto`）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_expansion_gpuaware.sh auto
./scripts/start_rerun_extra_dims_gpuaware.sh auto
```

可选模式：

```bash
./scripts/start_rerun_expansion_gpuaware.sh single
./scripts/start_rerun_expansion_gpuaware.sh dual
./scripts/start_rerun_extra_dims_gpuaware.sh single
./scripts/start_rerun_extra_dims_gpuaware.sh dual
```

### 15.6 总控入口（一步启动）

脚本：

- `scripts/start_rerun_all_gpuaware.sh`

参数：

- 第 1 个参数：实验分组  
  - `mainline`：主线 6 组  
  - `expansion`：扩展 6 组  
  - `extra`：新增维度 6 组  
  - `all`：按顺序启动全部分组
- 第 2 个参数：GPU 模式  
  - `auto` / `single` / `dual`
- 第 3 个参数：预检查开关  
  - `0`：正常启动  
  - `1`：dry-run（仅打印计划，不创建 tmux 会话）

示例：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_all_gpuaware.sh all auto
./scripts/start_rerun_all_gpuaware.sh mainline single
./scripts/start_rerun_all_gpuaware.sh expansion dual
./scripts/start_rerun_all_gpuaware.sh all auto 1
```

说明：

- 总控脚本会调用三个已存在的 gpuaware 脚本，不重复维护训练命令
- 已存在同名 tmux session 时会自动跳过，避免误覆盖
- 建议每次正式启动前先执行一次 `dry-run`，确认 session 名和 GPU 绑定是否符合预期

### 15.7 分批节流总控入口（推荐单卡高占用场景）

脚本：

- `scripts/start_rerun_all_throttled_gpuaware.sh`

参数：

- 第 1 个参数：实验分组  
  - `mainline` / `expansion` / `extra` / `all`
- 第 2 个参数：GPU 模式  
  - `auto` / `single` / `dual`
- 第 3 个参数：预检查开关  
  - `0` 正常启动，`1` dry-run
- 第 4 个参数：批次大小（每批最多启动多少个会话）  
  - 例如 `2`
- 第 5 个参数：批次轮询间隔秒数  
  - 例如 `30`
- 第 6/7 个参数：兼容保留（当前默认不自动重试）
  - 失败后由你手动判断是否重试，便于捕捉报错现场

示例（你的服务器当前负载较高，建议从这里开始）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_all_throttled_gpuaware.sh all auto 1 2 30
./scripts/start_rerun_all_throttled_gpuaware.sh mainline single 0 1 30 1 20
```

说明：

- 每批启动后会等待该批任务写出状态文件，再进入下一批，降低 OOM 风险
- 会话失败后默认保留（不自动关闭），便于你直接 `tmux attach -t <session>` 捕捉错误栈
- 若你希望更保守，可把批次大小设为 `1`
- 当前默认关闭自动重试（手动重试更利于定位问题）
- 正式运行结束后，会导出失败清单到：`logs_rerun/.launcher_status/failed_items_*.txt`
- 同时刷新最新失败清单软约定文件：`logs_rerun/.launcher_status/failed_items_latest.txt`

### 15.8 失败任务一键补跑（基于失败清单）

脚本：

- `scripts/start_rerun_failed_from_list.sh`

参数：

- 第 1 个参数：失败清单路径（默认使用 latest）  
  - 每行格式：`session:gpu:cmd`
- 第 2 个参数：dry-run  
  - `0` 正式补跑，`1` 仅预检查
- 第 3 个参数：轮询间隔秒数  
  - 例如 `20`

示例：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_failed_from_list.sh logs_rerun/.launcher_status/failed_items_latest.txt 1
./scripts/start_rerun_failed_from_list.sh logs_rerun/.launcher_status/failed_items_latest.txt 0 20
```

说明：

- 建议先 `dry-run` 确认会话名与绑卡，再正式补跑
- 若 latest 文件不存在，先执行一次“正式”分批脚本（`dry_run=0`）生成失败清单
- 正式补跑会等待任务结束，并自动刷新下一轮失败清单到 `failed_items_latest.txt`

### 15.9 循环补跑直到清零（或达到最大轮次）

脚本：

- `scripts/start_rerun_failed_until_clear.sh`

参数：

- 第 1 个参数：失败清单路径（默认 latest）
- 第 2 个参数：最大轮次（例如 `5`）
- 第 3 个参数：每轮轮询间隔秒数（例如 `20`）
- 第 4 个参数：dry-run（`0|1`）

示例：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_failed_until_clear.sh logs_rerun/.launcher_status/failed_items_latest.txt 5 20 1
./scripts/start_rerun_failed_until_clear.sh logs_rerun/.launcher_status/failed_items_latest.txt 5 20 0
```

补充：

- 若你当前机器 conda 路径不同，可临时跳过 conda 激活：
  - `USE_CONDA=0 ./scripts/start_rerun_failed_until_clear.sh ...`
- 正常训练建议保持 conda 激活（`USE_CONDA=1`，默认）

### 15.10 全自动总控（一条命令跑到底）

脚本：

- `scripts/start_rerun_full_autopilot.sh`

能力：

- Step 1：调用分批节流主流程（默认关闭自动重试，保留失败会话）
- Step 2：是否自动失败补跑由开关控制（默认关闭，建议手动）

参数：

- 第 1 个参数：分组（`mainline|expansion|extra|all`）
- 第 2 个参数：模式（`auto|single|dual`）
- 第 3 个参数：dry-run（`0|1`）
- 第 4 个参数：批次大小
- 第 5 个参数：轮询秒数
- 第 6/7 个参数：兼容保留（当前默认不自动重试）
- 第 8 个参数：失败循环补跑最大轮次
- 第 9 个参数：是否启用训练后自动汇总出表（`1|0`）
- 第 10 个参数：是否启用重算回填表格（`1|0`，较耗时）
- 第 11 个参数：重算回填 batch_size
- 环境变量：`AUTO_RERUN_FAILED=0|1`（默认 `0`，即不自动补跑）

示例：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/start_rerun_full_autopilot.sh all auto 1 1 30 1 20 5 1 0 2
./scripts/start_rerun_full_autopilot.sh mainline single 0 1 30 1 20 5 1 0 2
AUTO_RERUN_FAILED=1 ./scripts/start_rerun_full_autopilot.sh mainline single 0 1 30 1 20 5 1 0 2
./scripts/start_rerun_full_autopilot.sh mainline single 0 1 30 1 20 5 1 1 2
```

建议：

- 先 `dry-run` 看完整执行链路
- 当前你服务器负载较高，优先 `mainline single`
- 建议把第 4 个参数（批次大小）固定为 `1`，避免共享 GPU 场景并发炸显存
- 建议默认 `AUTO_RERUN_FAILED=0`，先抓到稳定错误栈，再手动决定重试策略
- 建议先用 `ENABLE_RECOMPUTE=0` 跑通，再在空闲时用 `ENABLE_RECOMPUTE=1` 做最终口径复核

---

