# MELD数据集下载和整理快速指南

## 快速开始

### 完整执行流程（复制粘贴即可）

```bash
# 1. SSH连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 3. 下载MELD原始数据
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 4. 解压原始数据
tar -xzf MELD.Raw.tar.gz

# 5. 下载标注文件
git clone https://github.com/declare-lab/MELD.git

# 6. 解压视频文件
mkdir -p MELD/videos
tar -xzf train.tar.gz -C MELD/videos/
tar -xzf dev.tar.gz -C MELD/videos/
tar -xzf test.tar.gz -C MELD/videos/

# 7. 整理数据到data/目录
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/organize_meld.py
```

## 详细步骤说明

### 步骤1：下载MELD.Raw.tar.gz

**下载链接**：http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 下载（文件较大，可能需要较长时间）
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 检查下载结果
ls -lh MELD.Raw.tar.gz
```

**如果下载中断**：
```bash
# 使用断点续传
wget -c http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```

### 步骤2：解压MELD.Raw.tar.gz

```bash
# 解压（会得到train.tar.gz, dev.tar.gz, test.tar.gz）
tar -xzf MELD.Raw.tar.gz

# 验证
ls -lh *.tar.gz
```

### 步骤3：下载标注文件

**方法A：克隆GitHub仓库（推荐）**

```bash
git clone https://github.com/declare-lab/MELD.git

# 验证
ls MELD/data/MELD/*.csv
```

**方法B：直接下载CSV文件**

```bash
mkdir -p MELD/data/MELD
cd MELD/data/MELD

wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv
wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv
wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv
```

### 步骤4：解压视频文件

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 创建视频目录
mkdir -p MELD/videos

# 解压训练集（可能需要较长时间）
tar -xzf train.tar.gz -C MELD/videos/

# 解压验证集
tar -xzf dev.tar.gz -C MELD/videos/

# 解压测试集
tar -xzf test.tar.gz -C MELD/videos/

# 验证
find MELD/videos -name "*.mp4" | wc -l
```

### 步骤5：运行整理脚本

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/organize_meld.py
```

## 数据集结构

### 下载后的结构

```
downloads/
├── MELD.Raw.tar.gz          # 原始压缩包
├── train.tar.gz             # 训练集视频（解压后）
├── dev.tar.gz               # 验证集视频（解压后）
├── test.tar.gz              # 测试集视频（解压后）
└── MELD/
    ├── videos/
    │   ├── train/           # 训练集视频（.mp4）
    │   ├── dev/             # 验证集视频（.mp4）
    │   └── test/             # 测试集视频（.mp4）
    └── data/
        └── MELD/
            ├── train_sent_emo.csv
            ├── dev_sent_emo.csv
            └── test_sent_emo.csv
```

### 整理后的结构

```
data/
├── train/
│   ├── video/          # meld_train_0001.mp4, ...
│   ├── text/           # meld_train_0001.txt, ...
│   └── labels/         # meld_train_0001.txt, ...
├── val/                # 从dev映射而来
│   ├── video/
│   ├── text/
│   └── labels/
└── test/
    ├── video/
    ├── text/
    └── labels/
```

## 文件命名规则

### 原始视频文件

- **格式**：`dia{dialogue_id}_utt{utterance_id}.mp4`
- **示例**：`dia6_utt1.mp4` 表示对话6中的第1个话语

### 整理后的文件

- **视频**：`meld_train_0001.mp4`
- **文本**：`meld_train_0001.txt`
- **标签**：`meld_train_0001.txt`

## 验证数据

### 检查文件数量

```bash
# 检查原始数据
echo "训练集视频: $(find downloads/MELD/videos/train -name '*.mp4' | wc -l)"
echo "验证集视频: $(find downloads/MELD/videos/dev -name '*.mp4' | wc -l)"
echo "测试集视频: $(find downloads/MELD/videos/test -name '*.mp4' | wc -l)"

# 检查整理后的数据
echo "训练集:"
ls data/train/video | wc -l
ls data/train/text | wc -l
ls data/train/labels | wc -l
```

### 查看样本内容

```bash
# 查看文本
cat data/train/text/meld_train_0001.txt

# 查看标签
cat data/train/labels/meld_train_0001.txt
# 输出示例：
# happy
# 0.8,0.7
```

## 常见问题

### 问题1：下载速度慢

**解决方案**：
- 使用screen保持会话，避免SSH断开
- 考虑在本地下载后上传到服务器

### 问题2：解压失败

**症状**：`tar: Error is not recoverable: exiting now`

**解决方案**：
```bash
# 检查文件完整性
md5sum MELD.Raw.tar.gz  # 如果有MD5值，进行校验

# 重新下载
rm MELD.Raw.tar.gz
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```

### 问题3：视频和标注不匹配

**症状**：整理脚本显示很多样本找不到视频

**解决方案**：
- 检查视频文件命名格式是否正确
- 检查CSV文件中的Dialogue_ID和Utterance_ID格式
- 手动验证几个样本的对应关系

## 使用screen保持下载会话

```bash
# 创建screen会话
screen -S download_meld

# 在screen中执行下载
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 按 Ctrl+A 然后按 D 可以暂时离开（下载会继续）
# 重新连接：screen -r download_meld
```

## 下一步

数据整理完成后，可以：

1. **验证数据加载**：使用`MultimodalDataset`测试数据加载
2. **开始训练**：参考`project/README.md`开始模型训练
3. **检查数据质量**：查看标签分布，确保数据平衡

