# 准确率优化实验序列：操作指南与论文记录规范

**文档版本**：v2（2026-04-11）  
**目的**：在**主线重跑已闭环**（见 `docs/EXPERIMENT_RERUN_FULL_RECORD_20260407.md` v3 与 `logs_rerun/` 归档）的前提下，开展**新一轮**准确率优化序列；**所有新训练与 TensorBoard 只写入** `*_accuracy_seq/` 目录，与历史重跑**完全隔离**。  
**配置根目录**：`project/config/rerun/accuracy_plan/`  

### 零、当前状态与策略调整（相对 v1）

| 事项 | 说明 |
|------|------|
| **重跑阶段** | 已完成；`logs_rerun/`、`checkpoints_rerun/`、`outputs_rerun/`（若存在）**仅作查阅与论文引用，禁止再写入新训练**。 |
| **新序列阶段** | 从本指南 **AP0～AP4**（及可选 AP5）在隔离目录执行；与重跑数值对比时**显式标注数据来源**（重跑表 vs accuracy_seq 表）。 |
| **AP0 定位** | v3 记录中 **AVT noDA standard 已满 50 epoch** 已在重跑侧留档。**AP0 改为可选**：若需在**同一套当前代码与日志字段**（如 `cls_ce_unweighted`）下再留一份 standard 对照，可跑 `ap0_*.yaml`；若论文直接引用重跑 standard，**可跳过 AP0，从 AP1 或 AP2 起跑**。 |
| **交叉引用** | 重跑事实锚点、差分表、选点规则以 **`EXPERIMENT_RERUN_FULL_RECORD_20260407.md` v3** 为准；本序列产出单独建表，勿与重跑目录混写。 |

**本轮专用根目录**（相对仓库中 `project/`，即训练时 `cd project` 后的工作目录）：

| 类型 | 相对路径 | TensorBoard / 脚本约定 |
|------|----------|-------------------------|
| 事件与指标 | `logs_accuracy_seq/` | `tensorboard_accuracy_seq.sh` 的 `--logdir` 指向此处 |
| Checkpoint | `checkpoints_accuracy_seq/` | 各 `accuracy_plan/*.yaml` 中 `paths.checkpoint_dir` |
| 汇总输出 | `outputs_accuracy_seq/` | 自建 CSV/表格、脚本导出；**勿**写入 `outputs_rerun/` |

**绝对路径示例**（将 `<REPO>` 换为你的克隆根；若 `project` 为子目录则含 `/project`）：

- 日志：`<REPO>/project/logs_accuracy_seq`
- 权重：`<REPO>/project/checkpoints_accuracy_seq`
- 汇总：`<REPO>/project/outputs_accuracy_seq`

---

## 一、背景与要解决的问题

### 1.1 本周重跑已暴露的现象（事实锚点）

- 混合三数据集、统一 7 类验证上，**末轮 Acc 约 0.18–0.38**；**VT+noDA** 与 **AVT+noDA+emotion_shift** 的**峰值 Acc**相对更优（详见 `docs/EXPERIMENT_RERUN_FULL_RECORD_20260407.md`）。
- 重跑 AVT 多为 **batch_size=1、gradient_accumulation_steps=2**（有效 batch≈2），梯度噪声大，不利于混合域收敛。
- **ClassBalancedLoss 按 batch 重算权重**，可导致 **val 总 loss 与 Acc 脱钩**；论文与调参应优先依据 **Acc / F1 / 不加权 CE**（若 `metrics.csv` 含 `cls_ce_unweighted` 列）。
- 融合策略上，主线重跑**未系统覆盖** `leader_follower`、`two_stage`，与「多模块/多论文复现」叙事相比消融矩阵不完整。

### 1.2 本轮序列的设计目标

1. **基线可比**：重跑侧已有 AVT **standard** 满程（v3）；本序列中 **可选** 在 `logs_accuracy_seq` 再跑同协议 standard，或与 **emotion_shift** 在 AP2/AP3 中直接对照（见阶段 0 说明）。  
2. **单数据集上界**：在 **CREMA / MELD / MOSEI** 各自上得到「同域上限」，与混合结果**分表报告**，避免混口径。  
3. **配方消融**：在固定强基线（**AVT+emotion_shift 三混合**）上，单独改变 batch、学习率、采样、损失形式。  
4. **融合消融**：在三混合 + 较优配方下，扫描 `standard / leader_follower / two_stage`（**emotion_shift** 由阶段 2 基线承担）。  
5. **域适应**：在较优混合配方确定后，扫描 `domain_loss_weight` 等（**以 best F1 选 checkpoint**，避免只报末轮）。  
6. **工程可复现**：统一 `training.seed: 3407`（除另有说明），日志与 checkpoint **全部写入新目录**。

---

## 二、目录与产物约定

### 2.1 单次 run 的目录命名

训练脚本会按 `experiment.name` 与时间戳创建子目录，例如：

- 日志：`logs_accuracy_seq/<experiment_name>_<timestamp>/`
  - `metrics.csv`、`metrics.jsonl`、TensorBoard 事件文件
- 权重：`checkpoints_accuracy_seq/<experiment_name>_<timestamp>/`
  - `checkpoint_pretrain_best_f1.pth`、`checkpoint_pretrain_best.pth`、按 epoch 保存等

### 2.2 TensorBoard（必须与 `logs_rerun` 分离）

**推荐**（与仓库脚本一致，默认端口 6006）：

```bash
cd /path/to/project    # 进入含 scripts/ 与 logs_accuracy_seq/ 的 project 目录
chmod +x scripts/tensorboard_accuracy_seq.sh   # 首次
./scripts/tensorboard_accuracy_seq.sh 6006
```

等价一行（自行替换绝对路径）：

```bash
tensorboard --logdir /path/to/project/logs_accuracy_seq --port 6006 --bind_all
```

浏览器打开 `http://<服务器IP>:6006`。**仅**加载 `logs_accuracy_seq`；若误打开仍指向 `logs_rerun` 的旧进程，请停掉后按上式重启。

**多 run 并列**：同一 `logdir` 下每个 `experiment_*` 子目录会作为独立 run 出现；脚本 `tensorboard_accuracy_seq.sh` 固定为 `PROJECT_DIR/logs_accuracy_seq`（见 `scripts/tensorboard_accuracy_seq.sh` 内 `echo [INFO] logdir=...` 可自检）。

**可选**：若需自定义子目录组合，可使用 `tensorboard --logdir_spec name1:path1,name2:path2`（仍建议路径均落在 `logs_accuracy_seq` 之下，避免与重跑混淆）。

**建议同步关注的标量**（若已实现落盘）：

- `val/f1`、`val/accuracy`（主指标）
- `val/cls_ce_unweighted`（与 ClassBalanced 解耦的监控）
- `val/loss_total`（总训练准则，跨实验仅作趋势参考）
- `val/loss_classification`（分项均值，便于解释总 loss）

---

## 三、阶段化实验方案（配置与理由）

### 阶段 0（AP0）：可选 — AVT standard 满程（隔离目录留档）

| 配置文件 | 要点 |
|----------|------|
| `accuracy_plan/ap0_AVT_noDA_standard_full50_s3407.yaml` | `fusion_strategy: standard`，`num_epochs: 50`，`seed: 3407`，三数据集混合；`paths.*` 已指向 `*_accuracy_seq/` |

**理由（更新）**：重跑记录 v3 中 **standard 满 50 epoch 已存在**（`logs_rerun`）。本阶段用于在**新目录**下用当前代码再跑一条同协议曲线，便于与 AP2/AP3 的日志字段、选点规则完全一致；**非强制**。

**何时跳过**：论文主表已采用重跑 standard，且不需要在 `logs_accuracy_seq` 中并列 standard 曲线时，可直接进入 **AP1** 或 **AP2**。

**操作**：

```bash
cd project
python3 scripts/train.py --config config/rerun/accuracy_plan/ap0_AVT_noDA_standard_full50_s3407.yaml --mode pretrain
```

---

### 阶段 1（AP1）：单数据集上界（VT ×3 + AVT+ES ×3）

| 文件模式 | 模态 | 数据子集 | 类别数 |
|----------|------|----------|--------|
| `ap1_VT_*_only_s3407.yaml` | VT | 仅 crema / meld / mosei | CREMA 6，其余 7 |
| `ap1_AVT_ES_*_only_s3407.yaml` | AVT + emotion_shift | 同上 | 同上 |

**理由**：单域、单标注空间通常 **Acc/F1 高于混合验证**，用于证明管线与模块「在理想条件下有效」，与混合实验**分开展示**。

**依赖代码行为**：`train.py` 在 **pretrain** 模式下会按 `training.pretrain.datasets` **过滤 train/val**，与 finetune 一致。

---

### 阶段 2（AP2）：混合训练配方消融（固定 AVT + emotion_shift）

| ID | 文件 | 变更意图 |
|----|------|----------|
| 基线 | `ap2_ES_baseline_3ds_s3407.yaml` | 与历史 emotion_shift 协议对齐 |
| M1 | `ap2_M1_effbatch8_ES_3ds_s3407.yaml` | 有效 batch 8（2×4），OOM 则退回基线 batch |
| M2 | `ap2_M2_lr5e5_ES_3ds_s3407.yaml` | 全局 lr 5e-5，抑制小 batch 高方差 |
| M3 | `ap2_M3_uniform_ES_3ds_s3407.yaml` | 采样 `uniform`，减轻某一域主导 |
| M4a | `ap2_M4_plain_ce_ES_3ds_s3407.yaml` | 关闭 ClassBalanced，纯 CE |
| M4b | `ap2_M4_focal_ES_3ds_s3407.yaml` | Focal 替代 CB |

**可选子阶段（AP2opt）**

| 文件 | 配置项 |
|------|--------|
| `ap2_opt_fixed_cb_ES_3ds_s3407.yaml` | `use_fixed_class_balanced_weights: true` |
| `ap2_opt_label_smooth_ES_3ds_s3407.yaml` | `label_smoothing: 0.05` |
| `ap2_opt_backbone_lr_ES_3ds_s3407.yaml` | `backbone_lr_multiplier: 0.1` |

**执行顺序建议**：先跑 **基线 + M1 + M2 + M3**，根据显存与曲线再开 M4a/M4b 与 AP2opt。

---

### 阶段 3（AP3）：融合策略消融（三混合，配方建议用阶段 2 最优）

| 文件 | `fusion_strategy` |
|------|-------------------|
| `ap3_fusion_standard_3ds_s3407.yaml` | standard |
| `ap3_fusion_leader_text_3ds_s3407.yaml` | leader_follower，`leader_modal: text` |
| `ap3_fusion_leader_audio_3ds_s3407.yaml` | leader_follower，`leader_modal: audio` |
| `ap3_fusion_two_stage_3ds_s3407.yaml` | two_stage |

**emotion_shift 对照**：使用阶段 2 的 `ap2_ES_baseline_3ds_s3407.yaml`，勿重复开跑除非改配方。

**论文写法**：固定「数据划分 + epoch + seed + 有效 batch 协议」，只变融合模块，报告 **Best F1 / Best Acc** 与 **Last**。

---

### 阶段 4（AP4）：域适应扫描

本轮已将原 `config_AVT_DA*.yaml` **复制**为 `accuracy_plan/ap4_config_AVT_DA*_accuracy_seq.yaml`，仅修改：

- `paths.log_dir` → `logs_accuracy_seq/`
- `paths.checkpoint_dir` → `checkpoints_accuracy_seq/`
- `experiment.name` 前缀 → `AP4_...`（避免与旧 run 混淆）

清单见 `accuracy_plan/ap4_da_sweep_manifest.yaml`。

**理由**：DA 在既往重跑中出现「峰值优于 noDA、末轮变差」——论文需 **同时报告 Best 与 Last**，并明确 early stopping 策略。

---

### 阶段 5（后续）：预训练 → 单域微调

仓库内 `config_crema_finetune_from_pretrain.yaml` 等默认 `paths` 为 `logs/`、`checkpoints/`。若希望**微调日志与权重也与重跑、与 accuracy 预训练同桶隔离**，请**复制**该模板到 `accuracy_plan/`（例如 `ap5_crema_finetune_from_accuracy_pretrain.yaml`），并显式设置：

- `paths.log_dir` → `logs_accuracy_seq/`
- `paths.checkpoint_dir` → `checkpoints_accuracy_seq/`
- `finetune` 中预训练权重路径 → 指向 `checkpoints_accuracy_seq/.../checkpoint_pretrain_best_f1.pth`（或你选定的 AP3/AP4 产出）

（若需 MELD/MOSEI 对称微调，可复制 CREMA 模板改 `datasets` 与 `emotion_classes`。）

---

## 四、一键启动（tmux，推荐）

### 4.1 单卡：按阶段启动

```bash
cd /path/to/project
export ENV_NAME=myenv310          # 改成你的 conda 环境
export GPU_ID=0                     # 单卡 ID
chmod +x scripts/start_accuracy_seq_tmux.sh
./scripts/start_accuracy_seq_tmux.sh ap0
./scripts/start_accuracy_seq_tmux.sh ap1
# … 依次 ap2、ap2opt、ap3、ap4；勿一上来用 all
```

查看会话：`tmux ls`；进入：`tmux attach -t aseq_ap0_std`。

**仅打印命令（不启动）**：

```bash
./scripts/start_accuracy_seq_tmux.sh list
```

### 4.2 双卡：自动/强制分配 GPU

```bash
chmod +x scripts/start_accuracy_seq_gpuaware.sh
./scripts/start_accuracy_seq_gpuaware.sh ap2 auto 0        # 正常启动
./scripts/start_accuracy_seq_gpuaware.sh ap2 dual 1      # dry-run：仅打印计划
```

环境变量：`GPU0`、`GPU1`、`GPU_SINGLE`、`MIN_FREE_MB` 与主线 `start_rerun_mainline_gpuaware.sh` 含义相同。

### 4.3 并行与显存建议

- **AVT 全模态**：每卡**同时 1 个**训练任务通常更安全；阶段内多会话并行仅在你确认显存充足时使用。  
- `ap2_M1`（有效 batch 8）更易 OOM，若失败保持 `ap2` 基线 batch 或减小 `batch_size`。

### 4.4 AP4 分会话手动启动（完整步骤，推荐）

本节与 **4.1 / 4.2 的一键脚本**并列：当你需要**错开显存高峰**、**避免同卡叠七路 AP4** 时，请**只按本节**操作。七个实验彼此独立，配置均写入 `logs_accuracy_seq/` 与 `checkpoints_accuracy_seq/`（见 `accuracy_plan/ap4_da_sweep_manifest.yaml`）。

#### 4.4.1 原则与节奏

- **不要**对 AP4 使用 `./scripts/start_accuracy_seq_gpuaware.sh ap4 …` 或 `./scripts/start_accuracy_seq_tmux.sh ap4` 一次拉起全部会话（会同时启动 7 条训练，极易 OOM）。  
- **每条 AP4 训练单独一个 tmux 会话**；**强烈建议同一物理 GPU 上同一时间只跑一条 AVT+DA 主训练**（若同卡还有 `train_SAC.py` 等常驻进程，用 `nvidia-smi` 观察峰值显存）。  
- **启动顺序（示例，可按机器调整）**：  
  - 在 **GPU0** 上先跑完 **实验 ①**，再启动 **实验 ②**（或等 ① 进入稳定期且显存余量极大时再考虑同卡第二条，风险自负）。  
  - 若 **GPU1** 仍被 AP3（如 `two_stage`）或其它大训练占用，**不要将新的 AP4 绑到 GPU1**，直到该任务结束、`nvidia-smi` 显存安全为止。  
  - **实验 ③～⑦** 在 ①② 完成后，按表中顺序轮流使用已空闲的 GPU（例如交替使用 GPU0 / GPU1，但仍保持**每卡一单任务**）。  
- **TensorBoard**：沿用 `./scripts/tensorboard_accuracy_seq.sh <端口>`，`logdir` 为 `logs_accuracy_seq/`；新开 run 会多一个带时间戳的子目录。  
- **续训**：若中断，可用 `python3 scripts/train.py ... --resume checkpoints_accuracy_seq/<run目录>/checkpoint_pretrain_best.pth`（或 `best_f1`）；若需复用原日志目录名，见仓库 `utils/helpers.py` 中环境变量 **`MULTIMODAL_LOG_RUN_DIR_NAME`** 的说明。

#### 4.4.2 开跑前一次性准备（每个终端 / 每台机器做一次）

以下路径、环境名请改成你的实际值。

**粘贴与续行提示（重要）**

- 请**一次性整段复制**下面第一个代码块（从 `# 1）` 到 `export GPU_AP4=0`），不要从中间的 `if` 行开始粘贴；否则容易漏掉 **`export CONDA_SH=""`**，`CONDA_SH` 可能带着旧值或根本未初始化。  
- 若提示符变成 **`>`**（续行），说明上一段 shell 语法未闭合（常见于只贴了 `if` 没贴 `fi`）：按 **`Ctrl+C`** 取消，再重新粘贴**完整**代码块。  
- 若当前 shell **已经能执行** `conda`（例如提示符前有 `(myenv310)`），可用**捷径**先得到 `conda.sh`，再跳过下方长逻辑：

```bash
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] && echo "OK: $CONDA_SH" || echo "[ERROR] 无 conda 或路径无效"
```

```bash
# 1）进入 project 根目录（含 scripts/、config/、logs_accuracy_seq/）
export PROJECT="/mnt/sda1/lizhichun_24/code/multimodal/project"
cd "$PROJECT"

# 2）Conda 环境名（与 4.1 一致）
export ENV_NAME="myenv310"

# 3）定位 conda.sh（与仓库脚本逻辑一致；找不到则安装/配置 conda）
export CONDA_SH=""
if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
else
  for _c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    [[ -f "$_c" ]] && CONDA_SH="$_c" && break
  done
fi
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi
[[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]] && { echo "[ERROR] 未找到 conda.sh，请设置 CONDA_BASE 或安装 conda"; exit 1; }

# 4）本实验使用的物理 GPU 编号（0 或 1）；每个实验启动前按需修改
export GPU_AP4=0
```

自检：

```bash
nvidia-smi
tmux ls
```

#### 4.4.3 通用模板（两种方式二选一）

**方式 A：交互式（适合调试）**

```bash
tmux new-session -s "<会话名>"
```

进入窗格后**逐行**执行（与下面「方式 B」中 `send-keys` 的内容相同）：`cd` → `source conda.sh` → `conda activate` → `export CUDA_VISIBLE_DEVICES=$GPU_AP4` → `python3 scripts/train.py ...`。  
 detached：`Ctrl+B` 再按 `D`。

**方式 B：后台创建会话并自动敲命令（适合远程、可复现）**

将下面各小节中的整段复制到 shell；把开头的 `SESSION`、`CFG`、`GPU_AP4` 换成该实验对应值（见 4.4.4 表）。

```bash
SESSION="<会话名>"
CFG="<config/rerun/accuracy_plan/...yaml>"
GPU_AP4=0   # 或 1

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m

# 查看训练日志（可选）
tmux attach -t "$SESSION"
```

训练结束后在窗格内 `exit` 退出 shell，或在外部执行：

```bash
tmux kill-session -t "<会话名>"
```

#### 4.4.4 七个 AP4 实验：会话名、配置与推荐命令块

| 序号 | tmux 会话名 | 配置文件（相对 `project/`） | manifest 备注 |
|------|-------------|-----------------------------|---------------|
| ① | `aseq_ap4_da_def` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_accuracy_seq.yaml` | AVT + DA 默认 |
| ② | `aseq_ap4_da_w02` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_w002_accuracy_seq.yaml` | domain_loss_weight 0.02 |
| ③ | `aseq_ap4_da_w05` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_accuracy_seq.yaml` | domain_loss_weight 0.05 |
| ④ | `aseq_ap4_da_w10` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_w010_accuracy_seq.yaml` | domain_loss_weight 0.10 |
| ⑤ | `aseq_ap4_da_w05lr` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml` | w005 + lr 5e-5 |
| ⑥ | `aseq_ap4_da_uni` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_uniform_accuracy_seq.yaml` | DA + uniform 采样 |
| ⑦ | `aseq_ap4_da_s34` | `config/rerun/accuracy_plan/ap4_config_AVT_DA_seed3407_accuracy_seq.yaml` | DA + seed 3407 |

**实验 ① — DA 默认**

```bash
SESSION="aseq_ap4_da_def"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_accuracy_seq.yaml"
GPU_AP4=0

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ② — domain_loss_weight 0.02**

```bash
SESSION="aseq_ap4_da_w02"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_w002_accuracy_seq.yaml"
GPU_AP4=0

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ③ — domain_loss_weight 0.05**

```bash
SESSION="aseq_ap4_da_w05"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_accuracy_seq.yaml"
GPU_AP4=1

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ④ — domain_loss_weight 0.10**

```bash
SESSION="aseq_ap4_da_w10"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_w010_accuracy_seq.yaml"
GPU_AP4=0

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ⑤ — w005 + lr 5e-5**

```bash
SESSION="aseq_ap4_da_w05lr"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml"
GPU_AP4=1

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ⑥ — DA + uniform 采样**

```bash
SESSION="aseq_ap4_da_uni"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_uniform_accuracy_seq.yaml"
GPU_AP4=0

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

**实验 ⑦ — DA + seed 3407（配置内已强调种子时请与论文一致）**

```bash
SESSION="aseq_ap4_da_s34"
CFG="config/rerun/accuracy_plan/ap4_config_AVT_DA_seed3407_accuracy_seq.yaml"
GPU_AP4=1

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$PROJECT\"" C-m
tmux send-keys -t "$SESSION" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION" "conda activate $ENV_NAME" C-m
tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU_AP4" C-m
tmux send-keys -t "$SESSION" "python3 scripts/train.py --config $CFG --mode pretrain" C-m
```

> **说明**：上表中 `GPU_AP4=0` / `1` 仅为「前几条在 GPU0、后续在 GPU1」的**示例**；你必须在每次启动前用 `nvidia-smi` 确认目标卡上无其它大训练，并随 AP3 结束与否**自行改写 `GPU_AP4`**。若某会话名已存在，`tmux new-session -d -s` 会失败，可先 `tmux kill-session -t <会话名>` 或换名。

#### 4.4.5 常用运维命令

```bash
tmux ls
tmux attach -t aseq_ap4_da_def
tmux kill-session -t aseq_ap4_da_def
```

---

## 五、论文级评价维度与消融对比规范

### 5.1 主表指标（混合三数据集 7 类验证）

对**每个 run**至少记录：

| 字段 | 来源 | 说明 |
|------|------|------|
| Best val Acc | `metrics.csv` 中 `phase=val` 的 `accuracy` 最大值及对应 `epoch` | 主结果之一 |
| Best val F1 | 同上，`f1` | **与 `checkpoint_pretrain_best_f1.pth` 对齐** |
| Last val Acc/F1 | 最后一行 val | 反映末段稳定性；DA 必报 |
| Best `cls_ce_unweighted` | 若 CSV 有该列 | 与 ClassBalanced 尺度解耦的校准参考 |
| 协议 | 配置中的 `batch_size`、`gradient_accumulation_steps`、`learning_rate`、`sampling.mode`、`fusion_strategy` | 表注必须写清 |

### 5.2 单数据集上界表（AP1）

- **验证集仅含该数据集**（由过滤逻辑保证）。  
- **CREMA 行必须注明 6 类**；MELD/MOSEI 为 7 类。  
- **禁止**与混合 7 类表合并为「一张总 Acc 排名表」而不加脚注。

### 5.3 消融对比的「控制变量」原则

- **配方消融（AP2）**：仅改 `training.*` / `loss.*` / `sampling.*`，**固定** `fusion_strategy: emotion_shift` 与三数据集列表。  
- **融合消融（AP3）**：**固定**阶段 2 选出的最优配方（batch、lr、采样、损失），**只改** `model.attention.fusion_strategy`（及 `leader_modal` 等从属项）。  
- **DA（AP4）**：在确定的三混合配方上，**只扫**域损失权重及相关项；报告 **Best vs Last**。

### 5.4 统计与可复现

- **种子**：默认 `3407`；若因硬件/CUDA 非确定性导致微小偏差，在方法节说明「近似复现」。  
- **多次运行**：若时间允许，关键配置可 **2 个种子** 报告均值±方差（选做）。

### 5.5 重算校验（推荐）

对写入论文的数值，任选 checkpoint 使用 `scripts/recompute_val_metrics.py` 交叉验证（与 CSV 差异应在数值误差量级内）。

---

## 六、与历史重跑目录的边界（强制隔离）

| 目录 | 用途 |
|------|------|
| `logs_rerun/`、`checkpoints_rerun/`、`outputs_rerun/` | **历史主线重跑归档**；只读查阅，**禁止新训练或新 TensorBoard 默认定根到此** |
| `logs_accuracy_seq/`、`checkpoints_accuracy_seq/`、`outputs_accuracy_seq/` | **本轮准确率优化序列唯一写入区** |

**开跑前自检**（在 `project/` 下）：

```bash
grep -R "logs_rerun\|checkpoints_rerun" config/rerun/accuracy_plan/*.yaml
```

**期望**：无匹配；或**仅** `ap4_da_sweep_manifest.yaml` 首行注释中出现「与 logs_rerun 隔离」字样。若在**非注释**的 `log_dir` / `checkpoint_dir` 中出现 `*_rerun`，必须改回 `*_accuracy_seq` 后再启动。

---

## 七、命令速查

```bash
# 打印全部单条训练命令（无 tmux）
python3 scripts/print_accuracy_plan_commands.py

# TensorBoard
./scripts/tensorboard_accuracy_seq.sh

# 单卡分阶段
./scripts/start_accuracy_seq_tmux.sh ap2

# 双卡 ap2 示例
./scripts/start_accuracy_seq_gpuaware.sh ap2 auto 0

# AP4 建议分会话手动启动（勿一键 ap4 七路）：见第四节「4.4」
```

---

## 八、实施检查清单（开跑前）

- [ ] `project/data/` 下数据与 `train.csv` / `val.csv` 就绪  
- [ ] Conda 环境中 `torch`、`cv2`、`librosa`、`transformers` 等可用  
- [ ] 已读 **重跑 v3** 与本文 **「零、当前状态」**：明确 AP0 是否执行  
- [ ] 所有将用于训练的 YAML 中 `paths.log_dir` / `checkpoint_dir` / `output_dir` 均为 `*_accuracy_seq/`（见第六节 `grep`）  
- [ ] 磁盘空间：`logs_accuracy_seq` 与 `checkpoints_accuracy_seq` 预计随阶段增长，勿与已满的 `logs_rerun` 混用清理策略  
- [ ] `CUDA_VISIBLE_DEVICES` / 多卡脚本与 `start_accuracy_seq_gpuaware.sh` 约定一致  
- [ ] 先小阶段试跑（**AP2 基线** 或 **AP0** 若需要）1～2 epoch 确认无 OOM  
- [ ] TensorBoard 使用 `tensorboard_accuracy_seq.sh` 或等价 `--logdir .../logs_accuracy_seq`，刷新后可见新 `experiment_*` 子目录  

**推荐顺序**：**（可选 AP0）→ AP1 → AP2 → AP3 → AP4**；AP0 跳过则从 AP1 或 AP2 起。确认无问题后再扩大 tmux 并行与阶段内多任务。
