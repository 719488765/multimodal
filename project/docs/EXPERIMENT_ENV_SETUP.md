## 混合数据集实验环境与复现实验清单

> 本文档用于记录当前项目在服务器上的实验环境配置，便于后续在本机或其他机器上**一键复现**。  
> 建议你在新环境上重跑实验时，严格按照本清单逐项检查。

---

### 一、代码与目录结构

- **代码根目录**：`/home/lizhichun_24/sda1/code/multimodal`
- **项目主目录**：`/home/lizhichun_24/sda1/code/multimodal/project`
- 关键子目录：
  - `config/`：实验配置文件（尤其是 `config/config.yaml`）
  - `data/`：统一后的数据集目录结构  
    - `data/train|val|test/{video,audio,text,labels,physiological}`
  - `models/`：主模型、特征提取器、融合模块、域适应等
  - `scripts/`：训练与数据整理脚本（`train.py` 等）
  - `utils/`：工具函数（日志、指标、checkpoint 管理）
  - `docs/`：项目说明与实验策略文档  
    - `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md`  
    - `MIXED_DATASET_TRAINING_ANALYSIS.md`  
    - 本文档 `EXPERIMENT_ENV_SETUP.md`

---

### 二、Conda 环境与依赖

#### 2.1 Python 与 Conda 环境

- Python 版本：**3.10**
- Conda 环境名称：**`myenv310`**

创建环境示例：

```bash
conda create -n myenv310 python=3.10 -y
conda activate myenv310
```

#### 2.2 关键第三方依赖（分类列出）

> 实际环境中建议在当前服务器执行：  
> `conda list > myenv310_conda_list.txt`  
> `pip freeze > myenv310_pip_freeze.txt`  
> 并将这两个文件一并保存，便于后续 1:1 复现实验环境。

- **深度学习与 GPU 支持**
  - `torch`（支持 CUDA 的版本，例如 2.x）
  - `torchvision`
  - `torchaudio`（如有）

- **预训练模型与 NLP**
  - `transformers`
  - `huggingface_hub`

- **音频 / 图像处理**
  - `librosa`
  - `opencv-python`
  - `Pillow`

- **数据处理与通用工具**
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `pyyaml`
  - `tqdm`

- **可视化与日志**
  - `tensorboard`

---

### 三、当前推荐的 Baseline 配置（AT 组合，无域适应）

> 该配置已在 `config/config.yaml` 中写明，也是 `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第 7.1.1 节所描述的“当前工程基线”。

#### 3.1 模态与模型相关配置

```yaml
model:
  modalities:
    use_video: false
    use_audio: true
    use_physiological: false
    use_text: true

  attention:
    fusion_strategy: "standard"

  domain_adaptation:
    enabled: false
```

- 说明：
  - 当前基线仅使用 **音频 + 文本（AT 组合）**，视频模态与生理模态暂时关闭；
  - 使用标准多头注意力融合作为稳定 Baseline；
  - 域适应模块暂时关闭，留待后续单独做“有/无 DA”的消融对比。

#### 3.2 数据与预处理配置（与显存相关）

```yaml
data:
  video:
    fps: 30
    frame_size: 160    # 由 224 下调，以降低显存占用
    num_frames: 8      # 由 16 下调，以降低显存占用
```

#### 3.3 训练与损失配置

```yaml
training:
  batch_size: 4
  num_epochs: 50
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler: "cosine"

  loss:
    use_class_balanced: true
    use_focal_loss: false
    use_domain_adaptation: false
```

---

### 四、标准运行命令（预训练）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

python scripts/train.py --config config/config.yaml --mode pretrain
```

- 实验日志：
  - 终端中会打印当前使用的配置、采样统计、每个 epoch 的训练/验证 loss 与指标；
  - 训练过程中会在 `logs/` 目录下创建带时间戳的子目录，包含：
    - `metrics.jsonl` / `metrics.csv`（数值日志）
    - TensorBoard 日志文件。

---

### 五、在新环境重跑实验时的检查清单

1. **代码是否与当前仓库版本一致？**
   - 提交 ID / 分支是否与服务器上使用的一致；
   - `config/config.yaml` 与两个 docs 文件是否已同步过去。

2. **Conda 环境是否正确？**
   - `python --version` 是否为 3.10；
   - `conda list` 与 `myenv310_conda_list.txt` 是否基本一致；
   - `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"` 是否正常。

3. **数据是否就绪？**
   - `data/train|val|test/{audio,text,labels}` 是否存在；
   - 文件前缀是否为 `crema_*/meld_*/mosei_*`；
   - 不依赖视频模态时，可以暂不检查 `video/` 目录。

4. **配置是否完全符合当前 Baseline？**
   - `use_video=false`、`use_audio/use_text=true`；
   - `fusion_strategy="standard"`；
   - 域适应关闭，类别平衡损失开启；
   - `batch_size=4`，`frame_size=160`，`num_frames=8`；
   - `learning_rate=1e-4`，`optimizer=adamw`。

5. **日志与 TensorBoard 是否工作正常？**
   - 训练开始后，终端是否打印 `Experiment log directory: logs/...`；
   - `logs/.../metrics.csv` 是否被持续写入；
   - TensorBoard 是否能读取并显示 `train/loss_total`、`val/loss_total`、`val/accuracy`、`val/f1` 等曲线。

---

### 六、关于当前服务器环境的注意事项（客观限制）

- GPU：NVIDIA RTX 4090（24GB 显存），但通常有其他进程同时占用一部分显存；
- 在 VAT + 域适应 + EmotionShift 的完整配置下，显存与 autograd 稳定性存在问题，当前文档和代码已记录了为解决这些问题而做出的妥协（关闭视频模态、关闭域适应、使用标准融合等）；
- 如果后续在新环境（显存更大或独占 GPU）中重跑实验，可以按本清单先复现 AT Baseline，再逐步打开视频模态、域适应和高级融合策略，完成全文计划中的完整消融实验。

