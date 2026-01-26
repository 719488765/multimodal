# CREMA-D视频文件整理修复说明

## 问题修复

已修复以下问题：
1. ✅ **添加.flv格式支持**：脚本现在可以识别和处理`.flv`格式的视频文件
2. ✅ **更新路径查找逻辑**：支持`content/CREMA-D/VideoFlash/`目录结构
3. ✅ **跳过音频处理**：自动检测已存在的音频文件，跳过重复处理
4. ✅ **视频音频匹配**：确保视频文件能正确匹配到已存在的音频文件

## 快速使用

### 如果音频文件已整理（你的情况）

```bash
# SSH连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project

# 激活虚拟环境
conda activate myenv310

# 运行整理脚本（会自动检测并跳过音频处理）
python scripts/organize_crema_d.py
```

脚本会自动：
- 检测到`data/audio`目录已有文件
- 只处理视频文件（.flv格式）
- 匹配视频和已存在的音频文件
- 生成标签文件

## 详细步骤

### 步骤1：确认数据集位置

```bash
# 检查数据集是否存在
ls -lh downloads/crema-d-emotional-multimodal-dataset/

# 查找.flv视频文件
find downloads/crema-d-emotional-multimodal-dataset -name "*.flv" | head -10

# 应该看到类似输出：
# downloads/crema-d-emotional-multimodal-dataset/content/CREMA-D/VideoFlash/1001_DFA_ANG_XX.flv
```

### 步骤2：确认音频文件已存在

```bash
# 检查音频文件
ls data/train/audio | head -10
ls data/val/audio | head -10
ls data/test/audio | head -10

# 统计数量
echo "训练集音频: $(ls data/train/audio | wc -l)"
echo "验证集音频: $(ls data/val/audio | wc -l)"
echo "测试集音频: $(ls data/test/audio | wc -l)"
```

### 步骤3：运行整理脚本

```bash
python scripts/organize_crema_d.py
```

**脚本输出示例**：

```
============================================================
CREMA-D数据集整理脚本
============================================================
扫描目录: /home/lizhichun_24/sda1/code/multimodal/project/downloads/crema-d-emotional-multimodal-dataset/content/CREMA-D
找到 7442 个文件

检测到 train 集已有音频文件，自动跳过音频处理
只处理视频文件，共 7442 个

整理 7442 个样本
数据集划分: train=5953, val=744, test=745

扫描已存在的音频文件，建立匹配索引...
  已建立 7442 个音频文件映射

处理 train 集...
  已处理 100/5953 个样本
  已处理 200/5953 个样本
  ...
处理 val 集...
处理 test 集...

============================================================
整理完成！
============================================================
训练集: 5953 个样本
验证集: 744 个样本
测试集: 745 个样本
```

### 步骤4：验证整理结果

```bash
# 检查视频文件数量
echo "训练集视频: $(ls data/train/video | wc -l)"
echo "验证集视频: $(ls data/val/video | wc -l)"
echo "测试集视频: $(ls data/test/video | wc -l)"

# 检查文件格式
ls data/train/video | head -5
# 应该看到：crema_train_0001.flv, crema_train_0002.flv 等

# 检查标签文件
cat data/train/labels/crema_train_0001.txt
# 输出示例：
# angry
# -0.7,0.8
```

## 脚本功能说明

### 自动检测功能

1. **路径自动查找**：
   - 尝试多个可能的路径
   - 支持`content/CREMA-D/VideoFlash/`结构
   - 自动找到数据集位置

2. **音频自动检测**：
   - 检查`data/train/audio`, `data/val/audio`, `data/test/audio`
   - 如果发现已有音频文件，自动跳过音频处理
   - 建立音频文件映射，确保视频和音频匹配

3. **格式支持**：
   - 视频：`.flv`, `.mp4`, `.avi`, `.mov`, `.mkv`
   - 音频：`.wav`, `.mp3`, `.flac`, `.m4a`

### 文件匹配逻辑

脚本通过以下方式匹配视频和音频：

1. **索引位置匹配**：如果音频和视频按相同顺序整理，通过索引位置匹配
2. **文件名base匹配**：如果文件名base相同（如`1001_DFA_ANG_XX`），自动匹配

## 常见问题

### 问题1：脚本找不到视频文件

**症状**：输出 `找到 0 个文件`

**解决方案**：
```bash
# 1. 检查数据集路径
find downloads/ -name "*.flv" | head -10

# 2. 如果文件在其他位置，修改脚本中的DOWNLOAD_DIR
# 编辑 scripts/organize_crema_d.py，修改第16行：
DOWNLOAD_DIR = "/实际/路径/到/数据集"
```

### 问题2：视频文件没有被复制

**症状**：脚本运行完成，但`data/video`目录为空

**解决方案**：
```bash
# 1. 检查文件权限
ls -ld data/train/video
chmod 755 data/train/video  # 如果需要

# 2. 检查磁盘空间
df -h /home/lizhichun_24/sda1

# 3. 查看脚本详细输出，检查是否有错误信息
python scripts/organize_crema_d.py 2>&1 | tee organize.log
```

### 问题3：视频和音频不匹配

**症状**：视频和音频文件数量不一致，或对应关系错误

**解决方案**：
- 脚本通过索引位置匹配，确保音频和视频按相同顺序整理
- 如果音频已按不同顺序整理，可能需要重新整理音频文件

### 问题4：.flv文件没有被识别

**症状**：脚本输出"未找到视频文件"

**解决方案**：
- 已修复：脚本现在支持`.flv`格式
- 如果仍有问题，检查文件扩展名是否正确（大小写不敏感）

## 手动验证

### 检查视频文件

```bash
# 查看视频文件
ls -lh data/train/video/ | head -10

# 检查文件大小（.flv文件通常几MB）
du -sh data/train/video/

# 验证文件完整性（尝试读取第一个文件）
file data/train/video/crema_train_0001.flv
```

### 检查标签文件

```bash
# 查看标签文件内容
head -5 data/train/labels/crema_train_0001.txt

# 统计各情感类别数量
grep -h "^" data/train/labels/*.txt | sort | uniq -c
```

### 检查文件对应关系

```bash
# 确保视频、音频、标签文件数量一致
echo "训练集:"
echo "  视频: $(ls data/train/video | wc -l)"
echo "  音频: $(ls data/train/audio | wc -l)"
echo "  标签: $(ls data/train/labels | wc -l)"
```

## 完整执行示例

```bash
# 1. 连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

# 3. 检查数据集
find downloads/crema-d-emotional-multimodal-dataset -name "*.flv" | wc -l

# 4. 检查音频文件（确认已存在）
ls data/train/audio | wc -l

# 5. 运行整理脚本
python scripts/organize_crema_d.py

# 6. 验证结果
ls data/train/video | wc -l
ls data/train/labels | wc -l

# 7. 检查一个样本
ls data/train/video/crema_train_0001.flv
ls data/train/audio/crema_train_0001.wav  # 应该存在
cat data/train/labels/crema_train_0001.txt
```

## 注意事项

1. **文件格式**：CREMA-D的视频文件是`.flv`格式，脚本已支持
2. **路径结构**：脚本支持`content/CREMA-D/VideoFlash/`目录结构
3. **音频匹配**：如果音频已整理，脚本会自动匹配，确保视频和音频使用相同的sample_id
4. **随机种子**：脚本使用固定随机种子（42），确保数据集划分可复现
5. **文件覆盖**：如果目标文件已存在，脚本会跳过，不会覆盖

## 下一步

整理完成后，可以：

1. **验证数据加载**：使用`MultimodalDataset`测试数据加载
2. **开始训练**：参考`project/README.md`开始模型训练
3. **检查数据质量**：查看标签分布，确保数据平衡

