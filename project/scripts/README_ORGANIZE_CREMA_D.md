# CREMA-D数据集整理快速指南

## 快速开始

### 在远程服务器上执行：

```bash
# 1. SSH连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project

# 3. 激活虚拟环境
conda activate myenv310

# 4. 检查数据集是否存在
ls -lh downloads/crema-d-emotional-multimodal-dataset/

# 5. 运行整理脚本
python scripts/organize_crema_d.py
```

## 详细步骤说明

### 步骤1：确认数据集位置

```bash
# 检查数据集是否在正确位置
ls -lh /home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 如果数据集在其他位置，脚本会自动查找
# 或者可以修改脚本中的 DOWNLOAD_DIR 变量
```

### 步骤2：运行整理脚本

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/organize_crema_d.py
```

### 步骤3：验证整理结果

```bash
# 检查各数据集的文件数量
echo "训练集:"
ls data/train/video | wc -l
ls data/train/audio | wc -l
ls data/train/labels | wc -l

echo "验证集:"
ls data/val/video | wc -l
ls data/val/audio | wc -l

echo "测试集:"
ls data/test/video | wc -l
ls data/test/audio | wc -l

# 查看一个样本的标签
cat data/train/labels/crema_train_0001.txt
```

## 脚本功能说明

### 自动完成的任务

1. **扫描数据集**：自动查找所有视频和音频文件
2. **解析标签**：从文件名提取情感类别（Happy, Sad, Angry, Fear, Disgust, Neutral）
3. **数据集划分**：按 8:1:1 划分 train/val/test
4. **文件整理**：复制文件到 `data/` 目录并统一命名
5. **生成标签**：为每个样本生成标签文件（情感类别 + 效价/唤醒度）

### 输出结构

整理后的目录结构：

```
data/
├── train/
│   ├── video/          # 训练视频文件
│   ├── audio/          # 训练音频文件
│   ├── text/           # 文本文件（占位）
│   ├── physiological/  # 生理信号（空）
│   └── labels/         # 标签文件
├── val/                # 验证集（结构同train）
└── test/               # 测试集（结构同train）
```

### 标签文件格式

每个标签文件（如 `data/train/labels/crema_train_0001.txt`）包含：

```
angry
-0.7,0.8
```

- 第一行：情感类别（happy, sad, angry, fear, disgust, neutral）
- 第二行：效价,唤醒度（范围通常在 [-1, 1]）

## 常见问题

### 问题1：找不到数据集

**错误信息**：`错误：找不到CREMA-D数据集目录`

**解决方案**：
```bash
# 1. 检查数据集是否已下载
ls -la downloads/

# 2. 如果数据集在其他位置，修改脚本中的路径
# 编辑 scripts/organize_crema_d.py，修改 DOWNLOAD_DIR 变量
```

### 问题2：无法解析情感标签

**症状**：所有样本都被标记为 'neutral'

**解决方案**：
```bash
# 1. 查看实际的文件名格式
find downloads/crema-d-emotional-multimodal-dataset -type f | head -10

# 2. 根据实际文件名格式，修改脚本中的 parse_crema_d_filename 函数
```

### 问题3：文件数量不匹配

**症状**：视频和音频文件数量不一致

**解决方案**：
- 这是正常的，CREMA-D可能有些样本只有视频或只有音频
- 脚本会处理这种情况，只复制存在的文件

### 问题4：存储空间不足

**症状**：`No space left on device`

**解决方案**：
```bash
# 检查可用空间
df -h /home/lizhichun_24/sda1

# 如果空间不足，可以考虑：
# 1. 只整理部分数据（修改脚本，限制样本数量）
# 2. 使用符号链接而不是复制文件（修改脚本使用 os.symlink）
```

## 自定义配置

### 修改数据集划分比例

编辑 `scripts/organize_crema_d.py`，修改 `organize_files` 函数调用：

```python
stats = organize_files(files, DATA_ROOT, 
                       train_ratio=0.8,  # 训练集比例
                       val_ratio=0.1,    # 验证集比例
                       test_ratio=0.1)   # 测试集比例
```

### 限制样本数量（用于快速测试）

在脚本的 `organize_files` 函数中，添加：

```python
# 只处理前1000个样本（用于快速测试）
if len(sample_list) > 1000:
    sample_list = sample_list[:1000]
```

### 修改情感标签映射

如果CREMA-D使用不同的标签格式，修改 `EMOTION_MAP` 字典：

```python
EMOTION_MAP = {
    'HAP': 'happy',
    'SAD': 'sad',
    # 添加其他映射...
}
```

## 验证数据可用性

整理完成后，可以使用项目的 `MultimodalDataset` 验证：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

python - <<'PY'
import os
import yaml
from data.dataset import MultimodalDataset

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
config_path = os.path.join(project_root, "config", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset = MultimodalDataset(
    data_dir=os.path.join(project_root, "data"),
    split="train",
    config=config,
)

print(f"数据集大小: {len(dataset)}")
sample = dataset[0]
print(f"样本键: {sample.keys()}")
print("数据加载成功！")
PY
```

## 下一步

数据整理完成后，可以：

1. **开始训练**：参考 `project/README.md` 和 `project/详细文档.md`
2. **验证数据**：使用 `MultimodalDataset` 加载数据，检查格式是否正确
3. **调整配置**：根据实际数据情况，调整 `config/config.yaml`

## 完整执行示例

```bash
# 1. 连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project

# 3. 激活环境
conda activate myenv310

# 4. 检查数据集
ls -lh downloads/crema-d-emotional-multimodal-dataset/ | head -10

# 5. 运行整理脚本
python scripts/organize_crema_d.py

# 6. 验证结果
ls data/train/video | wc -l
cat data/train/labels/crema_train_0001.txt

# 7. 测试数据加载
python -c "from data.dataset import MultimodalDataset; print('OK')"
```

## 注意事项

1. **备份数据**：整理前建议备份原始数据集
2. **检查空间**：确保有足够空间存储整理后的数据
3. **标签准确性**：脚本从文件名提取标签，如有官方标签文件，建议使用官方标签
4. **文本数据**：CREMA-D通常没有文本，脚本生成的是占位文本

