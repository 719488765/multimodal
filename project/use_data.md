# 多模态驾驶员情绪项目数据集使用手册（use_data.md）

> 本文档面向**几乎没有数据/命令行基础**的使用者，结合仓库中的 `dataset_application_guide.md`、`project/README.md` 和 `project/详细文档.md`，一步步教你在**远程服务器**上准备本项目所需的数据集。
>
> 当前假设：
> - 你已经在远程服务器上完成了代码拉取（`git clone`）；
> - 已按照 `详细文档.md` / `README.md` 完成 **Python 环境与依赖安装**；
> - 项目路径为：`/home/<你的用户名>/sda1/code/multimodal/project`（下文以 `lizhichun_24` 为例）。

---

## 一、整体目标与数据结构要求

### 1.1 本文目标

1. 教你**从零**准备本项目所需的多模态情感数据集（预训练 + 微调）。  
2. 教你在**远程服务器上直接下载或准备数据**，并整理为项目可以直接读取的格式。  
3. 教你用项目中的 `MultimodalDataset` 做一次“试读”，验证数据是否准备正确。

### 1.2 项目需要的数据目录结构

根据 `project/详细文档.md` 与 `project/data/dataset.py`，项目最终期望的数据结构为（方式二：目录结构）：

```bash
project/
├── data/
│   ├── train/
│   │   ├── video/           # 训练视频文件（.mp4 等）
│   │   ├── audio/           # 训练音频文件（.wav）
│   │   ├── physiological/   # 训练生理信号（.npy）
│   │   ├── text/            # 训练文本（.txt）
│   │   └── labels/          # 训练标签（.txt）
│   ├── val/
│   │   ├── video/
│   │   ├── audio/
│   │   ├── physiological/
│   │   ├── text/
│   │   └── labels/
│   └── test/
│       ├── video/
│       ├── audio/
│       ├── physiological/
│       ├── text/
│       └── labels/
```

**标签文件格式**（`labels/sample_001.txt`），详见 `详细文档.md` 与 `dataset.py`：

```text
happy
0.8,0.6
```

- 第 1 行：情绪类别（`happy, sad, angry, fear, neutral, anxious, other`）  
- 第 2 行：情绪强度，`效价,唤醒度`（通常在 \[-1,1\] 或 \[0,1\] 区间）

**推荐样本命名规则：**

- 视频：`sample_001.mp4`
- 音频：`sample_001.wav`
- 生理信号：`sample_001.npy`
- 文本：`sample_001.txt`
- 标签：`sample_001.txt`

`project/data/dataset.py` 会基于 `sample_id` 自动推断这些路径。

---

## 二、前置准备：远程项目路径与虚拟环境

### 2.1 确认项目路径

在远程终端中执行（以下以你的项目路径为例）：

```bash
ssh -p 1022 lizhichun_24@49.233.89.203

cd /home/lizhichun_24/sda1/code/multimodal/project
pwd
```

输出应为：

```bash
/home/lizhichun_24/sda1/code/multimodal/project
```

### 2.2 激活虚拟环境

```bash
conda activate myenv310
python --version
```

确认 Python 版本满足 `requirements.txt`（推荐 3.10）。

### 2.3 创建基础目录结构

参考 `详细文档.md` 第 9.1.2 步骤：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

mkdir -p data/{train,val,test}/{video,audio,physiological,text,labels}

# 创建一个下载临时目录
mkdir -p downloads
```

---

## 三、本项目涉及的主要数据集（预训练 vs 微调）

根据 `project/README.md` 与 `dataset_application_guide.md`：

### 3.1 预训练阶段数据集（通用多模态情感）

用于学习通用的表情/语音/文本情感特征：

- **MAFW**：多模态情感数据库（面部表情 + 语音 + 文本）
- **AffectNet**：大规模面部表情数据库
- **IEMOCAP**：语音 + 文本情感对话数据
- （可选）CMU-MOSEI/MELD/M3ED/EmotionTalk 等，用于多模态融合预训练

> 这些数据集主要用于：预训练 ResNet-50 / Wav2Vec2 / BERT 等特征提取器，以及基础融合模块。

### 3.2 微调阶段数据集（驾驶员场景）

用于贴近智能驾驶场景的微调：

- **MPDB**：多模态生理驾驶员行为数据库（EEG、ECG、GSR 等），是**驾驶员情绪识别的核心数据集**。
- **DEFE**（如有）：驾驶员面部表情数据集。
- 其它驾驶场景类数据（HDD、BDD100K 等）多用于场景理解，而非驾驶员情绪本身。

### 3.3 数据集获取方式概览

> 具体官网链接和说明请查看仓库根目录的 `links.txt`（由 `README.md` 指出）。

通常流程：

1. 在浏览器打开 `links.txt` 中的官网链接。  
2. 注册 / 登陆，必要时填写数据使用申请。  
3. 审批通过后获得下载链接，下载得到一个或多个压缩包（`.zip` / `.tar.gz`）。  
4. 将压缩包放到远程服务器的 `project/downloads/` 中（**如果官网允许服务器直接下载，可以用 wget；否则先在本地下载再上传**）。  

后续章节会详细讲如何在**远程服务器上直接下载**（wget/curl），以及在**必须登录时如何配合本地浏览器与 scp**。

---

## 三（补充）、开源数据集替代方案（推荐）

> **重要说明**：由于原计划中的 MAFW、AffectNet、IEMOCAP（预训练阶段）和 MPDB、DEFE（微调阶段）都需要申请且难以获得，本节提供**完全开源、可直接下载**的替代数据集方案，让你能够立即开始实验。

### 3.4 预训练阶段开源替代数据集

#### 替代方案一：CREMA-D + MELD + CMU-MOSEI（推荐组合）

**1. CREMA-D 情感多模态数据集**
- **用途**：替代 MAFW + AffectNet（面部表情 + 语音）
- **规模**：91 名演员，6 种情感状态（快乐、悲伤、愤怒、恐惧、厌恶、中性）
- **模态**：视频 + 音频 + 文本
- **获取方式**：
  - **方式 A（Kaggle，推荐）**：注册 Kaggle 账户后可直接下载
  - **方式 B（TensorFlow Datasets）**：使用 Python 代码直接加载，无需下载文件
- **下载链接**：
  - Kaggle：https://www.kaggle.com/datasets/orvile/crema-d-emotional-multimodal-dataset
  - TensorFlow Datasets：https://www.tensorflow.org/datasets/catalog/crema_d

**2. MELD 多模态情感对话数据集**
- **用途**：替代 IEMOCAP（对话场景的多模态情感）
- **规模**：13,000+ 视频片段，包含对话中的情感表达
- **模态**：视频 + 音频 + 文本
- **获取方式**：GitHub 开源，可直接下载，**无需申请**
- **下载链接**：https://github.com/declare-lab/MELD

**3. CMU-MOSEI 多模态情感数据集**
- **用途**：替代 MAFW + IEMOCAP（大规模多模态融合预训练）
- **规模**：3,228 条视频，来自 1,000 位 YouTube 用户
- **模态**：视频 + 语音 + 文本，标注了情感和情绪强度
- **获取方式**：
  - **方式 A（Kaggle，推荐）**：注册 Kaggle 账户后可直接下载，已预处理
  - **方式 B（官方 SDK）**：使用 CMU-MultimodalSDK 工具包
- **下载链接**：
  - Kaggle：https://www.kaggle.com/datasets/samarwarsi/cmu-mosei
  - 官方 GitHub：https://github.com/A2Zadeh/CMU-MultimodalSDK

#### 替代方案二：FER2013 + RAVDESS + BUPT（轻量级组合）

**1. FER2013 面部表情数据集**
- **用途**：替代 AffectNet（面部表情识别）
- **规模**：35,887 张面部图像，7 种情感类别
- **模态**：图像（可转换为视频帧）
- **获取方式**：Kaggle 免费下载
- **下载链接**：https://www.kaggle.com/datasets/msambare/fer2013

**2. RAVDESS 情感语音数据集**
- **用途**：替代 IEMOCAP（语音情感识别）
- **规模**：24 名演员，8 种情感状态
- **模态**：音频 + 视频
- **获取方式**：Zenodo 公开下载
- **下载链接**：https://zenodo.org/record/1188976

**3. BUPT 多模态数据集**
- **用途**：替代部分多模态融合数据（中文场景）
- **规模**：适用于跨模态检索和情感分析
- **模态**：多模态
- **获取方式**：Hugging Face 直接加载，**无需下载文件**
- **下载链接**：https://huggingface.co/datasets/weioshino/bupt_dataset

### 3.5 微调阶段开源替代数据集

由于 MPDB 和 DEFE 都需要申请且难以获得，以下提供开源替代方案：

#### 替代方案：SWELL-KW + 合成数据增强

**1. SWELL-KW 数据集（如可获得）**
- **用途**：替代 MPDB（驾驶员生理信号）
- **规模**：包含驾驶员在驾驶过程中的生理信号和视频数据
- **模态**：EEG + 视频 + 音频
- **获取方式**：需要搜索相关论文并联系作者
- **注意**：该数据集可能也需要申请，但比 MPDB 更容易获得

**2. 使用通用数据集 + 数据增强策略**
- **策略**：使用 CREMA-D、MELD 等通用数据集，通过数据增强（光照变化、噪声添加等）模拟驾驶场景
- **优势**：无需申请，立即可用
- **适用场景**：如果无法获得专门的驾驶员数据集，可以使用通用数据集 + 领域适应技术

### 3.6 推荐的开源数据集组合方案

**方案 A（推荐）：CREMA-D + MELD + CMU-MOSEI**
```
预训练阶段：
  - CREMA-D：面部表情 + 语音（替代 MAFW + AffectNet）
  - MELD：对话场景多模态（替代 IEMOCAP）
  - CMU-MOSEI：大规模多模态融合（增强预训练效果）

微调阶段：
  - 使用 CREMA-D + MELD 的子集，通过数据增强模拟驾驶场景
  - 或等待获得 MPDB/DEFE 后再进行微调
```

**方案 B（轻量级）：FER2013 + RAVDESS + BUPT**
```
预训练阶段：
  - FER2013：面部表情（替代 AffectNet）
  - RAVDESS：语音情感（替代 IEMOCAP）
  - BUPT：多模态融合（补充数据）

微调阶段：
  - 同方案 A
```

---

## 四、在远程服务器上直接下载数据集（有公开链接时）

> 仅适用于：**有公开 HTTP/HTTPS 直链** 或 **你已经在服务器上配置了可用的 Cookie/Token** 的情况。  
> 许多学术数据集需要人工同意协议，通常**不提供简单直链**，这时需要“本地浏览器下载 + 上传”方式（见下一节）。

### 4.1 检查 wget / curl 是否可用

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

wget --version  # 若没有则安装
curl --version
```

若命令不存在，可先安装（需要 sudo 权限）：

```bash
sudo apt update
sudo apt install -y wget curl unzip tar
```

### 4.2 在远程下载压缩包到 downloads/

假设某数据集提供了可直接访问的链接 `https://example.com/mafw_video.zip` 和 `https://example.com/mafw_audio.zip`：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 下载视频压缩包
wget -O MAFW_video.zip "https://example.com/mafw_video.zip"

# 下载音频压缩包
wget -O MAFW_audio.zip "https://example.com/mafw_audio.zip"
```

如果是 `.tar.gz`：

```bash
wget -O MPDB_physio.tar.gz "https://example.com/mpdb_physio.tar.gz"
```

### 4.3 解压到临时目录

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

# 为每个数据集准备一个独立目录，方便整理
mkdir -p downloads/MAFW
mkdir -p downloads/MPDB

cd downloads/MAFW
unzip ../MAFW_video.zip      -d video_raw
unzip ../MAFW_audio.zip      -d audio_raw
# 若还有文本/标签
unzip ../MAFW_text_labels.zip -d text_labels_raw  # 文件名根据你实际情况修改

cd ../MPDB
tar -xzf ../MPDB_physio.tar.gz -C physio_raw      # 或 unzip ../MPDB_physio.zip -d physio_raw
unzip ../MPDB_labels.zip -d labels_raw
```

> 解压后，可以用 `ls` / `find` 查看原始数据集的文件结构，后续会根据这些结构编写简单的“整理脚本”将数据搬到 `data/` 下。

---

## 四（补充）、开源数据集详细下载步骤

### 4.4 下载 CREMA-D 数据集（Kaggle 方式）

**步骤 1：配置 Kaggle API**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

# 安装 Kaggle CLI
pip install kaggle

# 创建 Kaggle 配置目录
mkdir -p ~/.kaggle
```

**步骤 2：获取 Kaggle API Token**

**详细操作步骤**：

1. **登录Kaggle账户**
   - 在浏览器访问 https://www.kaggle.com/
   - 使用你的账户登录（用户名：zhichunli）

2. **进入Settings页面**（三种方法任选其一）：
   
   **方法A：通过用户菜单（推荐）**
   - 点击页面右上角的用户头像（或用户名）
   - 在弹出的菜单中点击 **"Settings"**（设置）
   
   **方法B：直接访问URL**
   - 在浏览器地址栏输入：`https://www.kaggle.com/settings`
   - 按回车键直接进入设置页面
   
   **方法C：通过个人资料**
   - 点击右上角头像 → 选择 **"Your Profile"**（你的个人资料）
   - 在个人资料页面找到并点击 **"Settings"** 或 **"Account"**

3. **创建API Token**
   - 在Settings页面中，向下滚动找到 **"API"** 部分
   - 在API部分，点击 **"Create New Token"** 按钮
   - 如果已有Token，可能需要先点击 **"Expire API Token"** 再创建新的

4. **获取kaggle.json文件**

   **情况A：浏览器自动下载（正常情况）**
   - 点击"Create New Token"后，浏览器会自动下载 `kaggle.json` 文件
   - 文件通常保存在浏览器的默认下载目录（如：`C:\Users\lizhichun\Downloads\kaggle.json`）

   **情况B：浏览器没有自动下载（手动创建）**
   
   如果浏览器没有自动下载文件，可以手动创建 `kaggle.json`：
   
   **步骤1：获取你的Kaggle用户名**
   - 在Kaggle页面右上角查看你的用户名（例如：zhichunli）
   - 或者在Settings页面的API部分，查看显示的用户名
   
   **步骤2：获取API Key（Token）**
   - 在Settings页面的API部分，创建Token后会显示一个类似 `KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` 的字符串
   - 这就是你的API Key（完整复制，包括 `KGAT_` 前缀）
   
   **步骤3：手动创建kaggle.json文件**
   
   在本地电脑上创建文件（使用文本编辑器，如记事本、VS Code等）：
   
   - **文件位置**：`C:\Users\lizhichun\.kaggle\kaggle.json`
   - **文件内容**（替换为你的实际信息）：
     ```json
     {"username":"zhichunli","key":"KGAT_9875c021e2565d1de0041c5c553109aa"}
     ```
   
   **Windows PowerShell创建方法**：
   ```powershell
   # 创建.kaggle目录（如果不存在）
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
   
   # 创建kaggle.json文件（替换为你的实际用户名和key）
   @"
   {"username":"zhichunli","key":"KGAT_9875c021e2565d1de0041c5c553109aa"}
   "@ | Out-File -FilePath "$env:USERPROFILE\.kaggle\kaggle.json" -Encoding utf8
   
   # 验证文件创建成功
   Get-Content "$env:USERPROFILE\.kaggle\kaggle.json"
   ```
   
   **或者使用记事本手动创建**：
   1. 打开记事本
   2. 输入以下内容（替换为你的实际信息）：
      ```json
      {"username":"zhichunli","key":"KGAT_9875c021e2565d1de0041c5c553109aa"}
      ```
   3. 保存文件为：`C:\Users\lizhichun\.kaggle\kaggle.json`
   4. 注意：保存时选择"所有文件"类型，确保文件名是 `kaggle.json`（不是 `kaggle.json.txt`）

**注意事项**：
- 如果找不到Settings或API选项，请确保已完全登录账户
- 某些账户可能需要先验证邮箱才能创建API Token
- Token创建后请妥善保管，不要泄露给他人
- 如果浏览器没有自动下载，使用手动创建方法即可
- 确保kaggle.json文件格式正确（JSON格式，无多余空格或换行）

**步骤 3：上传 kaggle.json 到远程服务器**

**目标路径**：`~/.kaggle/kaggle.json`（完整路径：`/home/lizhichun_24/.kaggle/kaggle.json`）

**方法一：使用 scp 命令（推荐）**

在**本地 PowerShell** 执行：

```powershell
# 如果kaggle.json在.kaggle文件夹（推荐位置）
scp -P 1022 "$env:USERPROFILE\.kaggle\kaggle.json" `
    lizhichun_24@49.233.89.203:~/.kaggle/

# 或者如果kaggle.json在Downloads文件夹
scp -P 1022 "C:\Users\lizhichun\Downloads\kaggle.json" `
    lizhichun_24@49.233.89.203:~/.kaggle/
```

**如果远程服务器上.kaggle目录不存在，先创建**：

```powershell
# 1. 先创建目录
ssh -p 1022 lizhichun_24@49.233.89.203 "mkdir -p ~/.kaggle"

# 2. 然后上传文件
scp -P 1022 "$env:USERPROFILE\.kaggle\kaggle.json" `
    lizhichun_24@49.233.89.203:~/.kaggle/
```

**方法二：使用 WinSCP 图形化工具**

1. 打开 WinSCP，连接到服务器：`lizhichun_24@49.233.89.203`，端口：`1022`
2. 导航到：`/home/lizhichun_24/.kaggle/`（如果不存在，右键创建新文件夹 `.kaggle`）
3. 将本地的 `kaggle.json` 文件拖拽上传到该目录

**验证上传成功**：

```bash
# SSH连接到服务器后执行
ls -la ~/.kaggle/kaggle.json

# 应该看到文件存在
```

**步骤 4：在远程服务器设置权限并验证**

```bash
# 在远程服务器执行
chmod 600 ~/.kaggle/kaggle.json

# 验证 Kaggle 配置
kaggle datasets list | head
```

**步骤 5：使用 kagglehub 下载 CREMA-D 数据集（推荐方法）**

kagglehub 是 Kaggle 官方提供的 Python 库，比 Kaggle CLI 更简单易用。

**5.1 安装 kagglehub**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

# 安装 kagglehub
pip install kagglehub

# 验证安装
python -c "import kagglehub; print(kagglehub.__version__)"
```

**5.2 使用下载脚本（推荐）**

项目已提供下载脚本 `scripts/download_crema_d.py`，支持多种下载方法：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

# 确保脚本有执行权限
chmod +x scripts/download_crema_d.py

# 运行下载脚本（使用默认路径）
python scripts/download_crema_d.py

# 或者指定自定义路径
python scripts/download_crema_d.py /home/lizhichun_24/sda1/code/multimodal/project/downloads
```

**脚本功能说明**：
- **方法1**：尝试使用 `path` 参数直接指定下载位置（如果 kagglehub 版本支持）
- **方法2**：如果 path 参数不支持，自动使用环境变量 `KAGGLEHUB_HOME` 指定缓存目录
- **方法3**：如果前两种方法失败，下载到默认位置后移动到目标路径

**5.3 直接使用 Python 代码下载**

如果不想使用脚本，可以直接在 Python 中执行：

```python
import kagglehub
import os

# 设置目标路径
target_path = "/home/lizhichun_24/sda1/code/multimodal/project/downloads"
os.makedirs(target_path, exist_ok=True)

# 方法1：尝试使用 path 参数
try:
    path = kagglehub.dataset_download(
        "orvile/crema-d-emotional-multimodal-dataset",
        path=target_path
    )
except TypeError:
    # 方法2：使用环境变量
    os.environ['KAGGLEHUB_HOME'] = target_path
    path = kagglehub.dataset_download("orvile/crema-d-emotional-multimodal-dataset")

print(f"数据集下载到: {path}")
```

**5.4 验证下载结果**

```bash
# 检查下载的文件
ls -lh /home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 查看数据集结构
find /home/lizhichun_24/sda1/code/multimodal/project/downloads -type f | head -20

# 检查数据集大小
du -sh /home/lizhichun_24/sda1/code/multimodal/project/downloads/*
```

**5.5 处理下载的数据**

如果下载的是压缩包，需要解压：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 查找压缩包
find . -name "*.zip" -o -name "*.tar.gz"

# 解压（根据实际文件名调整）
unzip crema-d-emotional-multimodal-dataset.zip -d CREMA_D_raw

# 或
tar -xzf crema-d-emotional-multimodal-dataset.tar.gz -C CREMA_D_raw
```

**步骤 6：使用 Kaggle CLI 下载（备选方法）**

如果 kagglehub 不可用，可以使用传统的 Kaggle CLI：

```bash
# 下载 CREMA-D 数据集
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
kaggle datasets download -d orvile/crema-d-emotional-multimodal-dataset -p .

# 解压
unzip crema-d-emotional-multimodal-dataset.zip -d CREMA_D_raw
```

**步骤 7：使用 TensorFlow Datasets 方式（无需下载文件，可选）**

如果你不想下载文件，可以直接在 Python 代码中加载：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

pip install tensorflow-datasets

python - <<'PY'
import tensorflow_datasets as tfds
import os

# 加载 CREMA-D 数据集
ds = tfds.load('crema_d', split='train')

# 保存到本地目录（可选）
output_dir = "downloads/CREMA_D_raw"
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集并保存（示例：保存前 100 个样本）
for i, example in enumerate(ds.take(100)):
    # 根据实际需要保存视频、音频、标签等
    # 这里只是示例，实际保存逻辑需要根据项目需求调整
    pass

print("CREMA-D 数据集加载完成")
PY
```

**kagglehub 下载 CREMA-D 的注意事项和故障排查**

**注意事项**：

1. **kagglehub 版本**：不同版本的 kagglehub 可能 API 不同，建议使用最新版本
   ```bash
   pip install --upgrade kagglehub
   ```

2. **网络连接**：确保远程服务器可以访问 Kaggle（可能需要代理或 VPN）

3. **存储空间**：CREMA-D 数据集可能较大（几GB），确保有足够空间
   ```bash
   df -h /home/lizhichun_24/sda1  # 检查可用空间
   ```

4. **认证问题**：如果遇到认证错误，检查 kaggle.json 是否正确配置
   ```bash
   # 检查文件是否存在
   ls -la ~/.kaggle/kaggle.json
   
   # 检查文件内容格式
   cat ~/.kaggle/kaggle.json
   
   # 检查文件权限（必须是600）
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **路径权限**：确保对目标路径有写权限
   ```bash
   ls -ld /home/lizhichun_24/sda1/code/multimodal/project/downloads
   chmod 755 /home/lizhichun_24/sda1/code/multimodal/project/downloads  # 如果需要
   ```

**故障排查**：

**问题1：kagglehub 找不到 path 参数**

**症状**：`TypeError: dataset_download() got an unexpected keyword argument 'path'`

**解决方案**：
- 脚本会自动切换到环境变量方法或下载后移动方法
- 或者手动使用环境变量：
  ```bash
  export KAGGLEHUB_HOME=/home/lizhichun_24/sda1/code/multimodal/project/downloads
  python scripts/download_crema_d.py
  ```

**问题2：认证失败**

**症状**：`401 Unauthorized` 或 `Authentication failed`

**解决方案**：
```bash
# 1. 检查 kaggle.json 是否存在且格式正确
cat ~/.kaggle/kaggle.json
# 应该看到：{"username":"your_username","key":"KGAT_xxxxx"}

# 2. 检查文件权限
chmod 600 ~/.kaggle/kaggle.json

# 3. 验证认证（使用 kaggle CLI）
kaggle datasets list | head

# 4. 如果 kaggle CLI 可用但 kagglehub 不可用，检查 kagglehub 版本
pip install --upgrade kagglehub
```

**问题3：下载中断或网络错误**

**症状**：下载过程中断，或 `ConnectionError`

**解决方案**：
- kagglehub 支持断点续传，重新运行脚本即可
- 如果网络不稳定，可以设置代理：
  ```bash
  export HTTP_PROXY=http://proxy.example.com:8080
  export HTTPS_PROXY=http://proxy.example.com:8080
  python scripts/download_crema_d.py
  ```

**问题4：存储空间不足**

**症状**：`No space left on device`

**解决方案**：
```bash
# 检查可用空间
df -h

# 清理不需要的文件
# 或者更改下载路径到有更多空间的目录
python scripts/download_crema_d.py /path/to/larger/disk/downloads
```

**问题5：权限错误**

**症状**：`Permission denied` 或 `Access denied`

**解决方案**：
```bash
# 检查目标目录权限
ls -ld /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 如果需要，创建目录并设置权限
mkdir -p /home/lizhichun_24/sda1/code/multimodal/project/downloads
chmod 755 /home/lizhichun_24/sda1/code/multimodal/project/downloads
```

**问题6：kagglehub 导入错误**

**症状**：`ModuleNotFoundError: No module named 'kagglehub'`

**解决方案**：
```bash
# 确保在正确的虚拟环境中
conda activate myenv310

# 安装 kagglehub
pip install kagglehub

# 验证安装
python -c "import kagglehub; print(kagglehub.__version__)"
```

### 4.5 下载 MELD 数据集（官方推荐方式）

**MELD数据集简介**：
- **全称**：Multimodal EmotionLines Dataset
- **规模**：13,000+ 个对话片段，来自Friends TV系列
- **模态**：视频（.mp4）+ 音频 + 文本
- **情感类别**：7种（Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear）
- **数据划分**：Train (9,989), Dev (1,109), Test (2,610)

**步骤 1：下载原始视频数据（MELD.Raw.tar.gz）**

根据MELD官方README，使用wget直接下载原始数据：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 下载MELD原始数据（包含train.tar.gz, dev.tar.gz, test.tar.gz）
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 检查下载是否成功
ls -lh MELD.Raw.tar.gz

# 注意：文件可能很大（几GB），确保有足够的存储空间和稳定的网络连接
```

**如果wget下载失败或速度慢**：

可以使用curl或分块下载：

```bash
# 使用curl下载
curl -L -o MELD.Raw.tar.gz http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 或使用断点续传（如果下载中断）
wget -c http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```

**步骤 2：解压MELD.Raw.tar.gz**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 解压MELD.Raw.tar.gz（会得到train.tar.gz, dev.tar.gz, test.tar.gz）
tar -xzf MELD.Raw.tar.gz

# 查看解压后的文件
ls -lh

# 应该看到：
# train.tar.gz  - 训练集视频文件
# dev.tar.gz    - 验证集视频文件（MELD使用dev而不是val）
# test.tar.gz   - 测试集视频文件
```

**步骤 3：下载标注文件（CSV格式）**

MELD的标注文件在GitHub仓库中，需要克隆仓库或直接下载CSV文件：

**方法A：克隆GitHub仓库（推荐，可获取完整信息）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 克隆MELD仓库
git clone https://github.com/declare-lab/MELD.git

# 查看标注文件位置
ls MELD/data/MELD/

# 应该看到：
# train_sent_emo.csv  - 训练集标注
# dev_sent_emo.csv    - 验证集标注
# test_sent_emo.csv   - 测试集标注
```

**方法B：直接下载CSV文件（如果只需要标注）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 创建标注文件目录
mkdir -p MELD/data/MELD

# 直接下载CSV标注文件
cd MELD/data/MELD

wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv
wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv
wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv

# 验证下载
ls -lh *.csv
```

**步骤 4：解压视频文件**

MELD的视频文件存储在三个tar.gz文件中，需要分别解压：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 创建解压目录
mkdir -p MELD/videos

# 解压训练集视频
echo "解压训练集视频（可能需要较长时间）..."
tar -xzf train.tar.gz -C MELD/videos/

# 解压验证集视频
echo "解压验证集视频..."
tar -xzf dev.tar.gz -C MELD/videos/

# 解压测试集视频
echo "解压测试集视频..."
tar -xzf test.tar.gz -C MELD/videos/

# 查看解压后的视频文件
find MELD/videos -name "*.mp4" | head -10

# 统计视频文件数量
echo "训练集视频: $(find MELD/videos/train -name '*.mp4' 2>/dev/null | wc -l)"
echo "验证集视频: $(find MELD/videos/dev -name '*.mp4' 2>/dev/null | wc -l)"
echo "测试集视频: $(find MELD/videos/test -name '*.mp4' 2>/dev/null | wc -l)"
```

**视频文件命名规则**：
- 格式：`diaX1_uttX2.mp4`
- `X1` = Dialogue_ID（对话ID，从0开始）
- `X2` = Utterance_ID（话语ID，从0开始）
- 示例：`dia6_utt1.mp4` 表示对话6中的第1个话语

**步骤 5：查看标注文件格式**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 查看训练集标注文件的前几行
head -5 MELD/data/MELD/train_sent_emo.csv

# 查看列名
head -1 MELD/data/MELD/train_sent_emo.csv

# 输出示例：
# Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
```

**标注文件列说明**：
- **Utterance**：文本内容
- **Speaker**：说话者名称
- **Emotion**：情感类别（neutral, joy, sadness, anger, surprise, fear, disgust）
- **Sentiment**：情感倾向（positive, neutral, negative）
- **Dialogue_ID**：对话ID（用于匹配视频文件）
- **Utterance_ID**：话语ID（用于匹配视频文件）
- **Season, Episode**：来自Friends TV的季数和集数
- **StartTime, EndTime**：时间戳

**步骤 6：验证数据完整性**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 检查视频文件数量
echo "训练集视频文件数:"
find MELD/videos/train -name "*.mp4" 2>/dev/null | wc -l

# 检查标注文件行数（减去标题行）
echo "训练集标注数:"
tail -n +2 MELD/data/MELD/train_sent_emo.csv | wc -l

# 验证视频和标注是否匹配（示例：检查dia6_utt1是否存在）
ls MELD/videos/train/dia6_utt1.mp4 2>/dev/null && echo "视频文件存在" || echo "视频文件不存在"
grep "6,1," MELD/data/MELD/train_sent_emo.csv | head -1
```

**步骤 7：整理MELD数据到data/目录**

MELD数据集需要从CSV标注文件和视频文件整理到项目的数据格式。项目提供了整理脚本：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

# 运行MELD数据整理脚本
python scripts/organize_meld.py
```

**手动整理方法**（如果脚本不可用）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

python - <<'PY'
import os
import shutil
import csv
import random

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
meld_root = os.path.join(project_root, "downloads", "MELD")
data_root = os.path.join(project_root, "data")

# 情感类别映射（MELD使用7种情感）
emotion_map = {
    'neutral': 'neutral',
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust'
}

# 处理每个数据集划分
for split_name, csv_file in [
    ('train', 'train_sent_emo.csv'),
    ('val', 'dev_sent_emo.csv'),  # MELD使用dev，映射到val
    ('test', 'test_sent_emo.csv')
]:
    csv_path = os.path.join(meld_root, "data", "MELD", csv_file)
    video_dir = os.path.join(meld_root, "videos", split_name if split_name != 'val' else 'dev')
    
    if not os.path.exists(csv_path):
        print(f"警告：标注文件不存在: {csv_path}")
        continue
    
    if not os.path.exists(video_dir):
        print(f"警告：视频目录不存在: {video_dir}")
        continue
    
    # 读取CSV文件
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            emotion = row['Emotion'].lower()
            utterance_text = row['Utterance']
            
            # 视频文件名格式：dia{dialogue_id}_utt{utterance_id}.mp4
            video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            video_path = os.path.join(video_dir, video_filename)
            
            if os.path.exists(video_path):
                samples.append({
                    'video_path': video_path,
                    'emotion': emotion_map.get(emotion, 'neutral'),
                    'text': utterance_text,
                    'dialogue_id': dialogue_id,
                    'utterance_id': utterance_id
                })
    
    print(f"{split_name}集: 找到 {len(samples)} 个有效样本")
    
    # 复制文件到data目录
    for idx, sample in enumerate(samples, start=1):
        sample_id = f"meld_{split_name}_{idx:04d}"
        
        # 复制视频
        dst_video = os.path.join(data_root, split_name, "video", f"{sample_id}.mp4")
        os.makedirs(os.path.dirname(dst_video), exist_ok=True)
        shutil.copy2(sample['video_path'], dst_video)
        
        # 生成文本文件
        dst_text = os.path.join(data_root, split_name, "text", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_text), exist_ok=True)
        with open(dst_text, 'w', encoding='utf-8') as f:
            f.write(sample['text'] + "\n")
        
        # 生成标签文件
        emotion = sample['emotion']
        # 简化的效价和唤醒度映射
        valence_arousal_map = {
            'happy': (0.8, 0.7),
            'sad': (-0.6, -0.3),
            'angry': (-0.7, 0.8),
            'fear': (-0.5, 0.9),
            'disgust': (-0.7, 0.5),
            'surprise': (0.3, 0.8),
            'neutral': (0.0, 0.0)
        }
        valence, arousal = valence_arousal_map.get(emotion, (0.0, 0.0))
        
        dst_label = os.path.join(data_root, split_name, "labels", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        with open(dst_label, 'w', encoding='utf-8') as f:
            f.write(f"{emotion}\n")
            f.write(f"{valence},{arousal}\n")
        
        # 音频文件（MELD视频包含音频，可以从视频提取，这里先跳过）
        # 生理信号（MELD不包含，保持为空）

print("MELD数据整理完成！")
PY
```

**MELD数据整理注意事项**：

1. **数据集划分**：MELD使用train/dev/test，项目使用train/val/test，dev映射到val
2. **视频格式**：MELD视频是.mp4格式，包含音频
3. **文本数据**：MELD有完整的文本标注（Utterance列）
4. **情感类别**：MELD有7种情感，需要映射到项目的7种情感类别
5. **文件匹配**：通过Dialogue_ID和Utterance_ID匹配视频和标注

**验证整理结果**：

```bash
# 检查文件数量
echo "训练集:"
ls data/train/video | wc -l
ls data/train/text | wc -l
ls data/train/labels | wc -l

# 查看一个样本
cat data/train/text/meld_train_0001.txt
cat data/train/labels/meld_train_0001.txt
```

**MELD数据集下载和整理的完整流程总结**：

```bash
# 1. 下载原始数据
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 2. 解压原始数据
tar -xzf MELD.Raw.tar.gz  # 得到train.tar.gz, dev.tar.gz, test.tar.gz

# 3. 下载标注文件
git clone https://github.com/declare-lab/MELD.git
# 或直接下载CSV文件（见步骤3）

# 4. 解压视频文件
mkdir -p MELD/videos
tar -xzf train.tar.gz -C MELD/videos/
tar -xzf dev.tar.gz -C MELD/videos/
tar -xzf test.tar.gz -C MELD/videos/

# 5. 整理数据到data/目录
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/organize_meld.py
```

**MELD数据集注意事项和故障排查**：

**注意事项**：
1. **存储空间**：MELD数据集较大（几GB），确保有足够空间
2. **下载时间**：根据网络速度，下载可能需要较长时间
3. **解压时间**：解压视频文件可能需要较长时间
4. **数据集划分**：MELD使用dev，项目使用val，脚本会自动映射

**常见问题**：

**问题1：wget下载失败**

**症状**：`Connection refused` 或 `404 Not Found`

**解决方案**：
```bash
# 1. 检查网络连接
ping web.eecs.umich.edu

# 2. 尝试使用curl
curl -L -o MELD.Raw.tar.gz http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz

# 3. 如果仍然失败，检查URL是否更新
# 访问MELD GitHub仓库查看最新下载链接
```

**问题2：解压后找不到视频文件**

**症状**：解压后没有看到train/dev/test目录

**解决方案**：
```bash
# 检查解压后的结构
ls -la downloads/
find downloads/ -name "*.tar.gz" -type f

# 可能需要再次解压
tar -tzf train.tar.gz | head -10  # 查看压缩包内容
tar -xzf train.tar.gz  # 解压到当前目录
```

**问题3：视频和标注不匹配**

**症状**：整理脚本显示很多样本找不到视频文件

**解决方案**：
```bash
# 1. 检查视频文件命名格式
ls downloads/MELD/videos/train/ | head -10

# 2. 检查CSV文件中的Dialogue_ID和Utterance_ID格式
head -5 downloads/MELD/data/MELD/train_sent_emo.csv

# 3. 手动验证一个样本
# 例如：CSV中Dialogue_ID=6, Utterance_ID=1
# 应该对应视频文件：dia6_utt1.mp4
ls downloads/MELD/videos/train/dia6_utt1.mp4
```

**问题4：CSV文件下载失败**

**症状**：GitHub raw链接无法访问

**解决方案**：
```bash
# 1. 使用GitHub完整URL
wget https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv

# 2. 或克隆整个仓库
git clone https://github.com/declare-lab/MELD.git

# 3. 如果GitHub访问困难，可以手动在浏览器下载后上传
```

### 4.6 下载 CMU-MOSEI 数据集（Kaggle 方式，推荐）

**步骤 1：使用 Kaggle API 下载**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 下载 CMU-MOSEI（Kaggle 版本，已预处理）
kaggle datasets download -d samarwarsi/cmu-mosei -p .

# 解压（注意：文件可能很大，确保有足够空间）
unzip cmu-mosei.zip -d CMU_MOSEI_raw
```

**步骤 2：使用官方 SDK 方式（可选）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 克隆 CMU-MultimodalSDK
git clone https://github.com/A2Zadeh/CMU-MultimodalSDK.git

cd CMU-MultimodalSDK

# 安装 SDK
pip install -r requirements.txt
python setup.py install

# 使用 SDK 下载数据（按照 SDK 文档说明）
python download_mosei.py  # 示例命令，实际命令请查看 SDK 文档
```

### 4.7 下载 FER2013 数据集（Kaggle 方式）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 下载 FER2013
kaggle datasets download -d msambare/fer2013 -p .

# 解压
unzip fer2013.zip -d FER2013_raw
```

### 4.8 下载 RAVDESS 数据集（Zenodo 直链）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# RAVDESS 数据集在 Zenodo 上，需要找到具体的下载链接
# 访问 https://zenodo.org/record/1188976 获取最新下载链接

# 示例下载命令（需要替换为实际链接）
wget -O RAVDESS.zip "https://zenodo.org/record/1188976/files/RAVDESS.zip?download=1"

# 解压
unzip RAVDESS.zip -d RAVDESS_raw
```

### 4.9 使用 BUPT 数据集（Hugging Face，无需下载文件）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

pip install datasets

python - <<'PY'
from datasets import load_dataset
import os

# 直接加载 BUPT 数据集（无需下载文件）
dataset = load_dataset('weioshino/bupt_dataset')

# 如果需要保存到本地
output_dir = "downloads/BUPT_raw"
os.makedirs(output_dir, exist_ok=True)

# 保存数据集（根据实际需要调整）
dataset.save_to_disk(output_dir)

print("BUPT 数据集加载并保存完成")
PY
```

---

## 五、无法在服务器直接 wget 时：本地下载 + scp 上传

很多数据集（尤其是 MAFW、MPDB 等）要求你在网页上「同意协议 / 手动点击下载」，这类链接往往：

- 受 Cookie/Session 保护；
- 或需要 POST 请求/表单提交；
- 不容易直接用 `wget` 下载。

这时推荐的安全流程是：

1. 在**你本地电脑的浏览器**上，根据 `links.txt` 的说明登录/申请，**把压缩包下载到本地磁盘**。  
2. 用 `scp`（或 WinSCP 等图形工具）把压缩包上传到远程服务器的 `project/downloads/` 目录。  

### 5.1 本地使用 scp 上传（以 Windows + PowerShell 为例）

假设你本地已下载：

- `D:\datasets\MAFW\MAFW_video.zip`
- `D:\datasets\MAFW\MAFW_audio.zip`
- `D:\datasets\MAFW\MAFW_text_labels.zip`
- `D:\datasets\MPDB\MPDB_physio.zip`
- `D:\datasets\MPDB\MPDB_labels.zip`

在本地 PowerShell 中执行（注意端口 1022）：

```powershell
scp -P 1022 "D:\datasets\MAFW\MAFW_video.zip" `
    lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

scp -P 1022 "D:\datasets\MAFW\MAFW_audio.zip" `
    lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

scp -P 1022 "D:\datasets\MAFW\MAFW_text_labels.zip" `
    lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

scp -P 1022 "D:\datasets\MPDB\MPDB_physio.zip" `
    lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

scp -P 1022 "D:\datasets\MPDB\MPDB_labels.zip" `
    lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/
```

上传完成后，在远程终端中检查：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
ls downloads
```

接下来的“解压 + 整理到 data/”步骤与前一节完全相同。

---

## 六、将解压后的数据整理到 data/ 目录

> 这一部分是**关键**：需要从原始数据集的文件结构中，挑选出**视频 / 音频 / 生理信号 / 文本 / 标签**五类文件，按统一命名规则放到 `data/{train,val,test}/...` 对应目录中。
>
> 不同数据集的原始结构不一样，因此下面给出一个通用“示例脚本骨架”，你可以根据具体数据集的文件名稍作修改。

### 6.1 MAFW 示例：整理视频 + 音频 + 文本/标签

假设解压后的结构大致如下（伪示例）：

```bash
downloads/MAFW/
├── video_raw/
│   ├── clip_0001.mp4
│   ├── clip_0002.mp4
│   └── ...
├── audio_raw/
│   ├── clip_0001.wav
│   ├── clip_0002.wav
│   └── ...
└── text_labels_raw/
    ├── clip_0001.txt
    ├── clip_0002.txt
    └── ...
```

你可以在远程终端中运行如下脚本（只需改少量路径/前缀即可）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

python - <<'PY'
import os, shutil, math

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
mafw_root    = os.path.join(project_root, "downloads", "MAFW")
data_root    = os.path.join(project_root, "data")

# 1. 获取视频文件列表（作为样本 ID 的基准）
video_dir  = os.path.join(mafw_root, "video_raw")
video_files = sorted([
    f for f in os.listdir(video_dir)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
])

print("发现视频文件数量：", len(video_files))

# 可选：为了快速实验，先只用前 100 个样本
max_samples = 100
video_files = video_files[:max_samples]

# 2. 划分 train/val/test（8:1:1）
n       = len(video_files)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)
n_test  = n - n_train - n_val

splits = (
    ("train", video_files[:n_train]),
    ("val",   video_files[n_train:n_train + n_val]),
    ("test",  video_files[n_train + n_val:]),
)

audio_dir       = os.path.join(mafw_root, "audio_raw")
text_labels_dir = os.path.join(mafw_root, "text_labels_raw")

for split, files in splits:
    print(f"处理 {split} 集，样本数：{len(files)}")
    for idx, vf in enumerate(files, start=1):
        # 生成统一的 sample_id，例如 mafw_train_0001
        sample_id = f"mafw_{split}_{idx:04d}"

        # ------- 视频 -------
        src_video = os.path.join(video_dir, vf)
        dst_video = os.path.join(data_root, split, "video", f"{sample_id}.mp4")
        os.makedirs(os.path.dirname(dst_video), exist_ok=True)
        shutil.copy2(src_video, dst_video)

        base = os.path.splitext(vf)[0]

        # ------- 音频（如存在）-------
        if os.path.isdir(audio_dir):
            candidates = [
                f for f in os.listdir(audio_dir)
                if f.startswith(base) and f.lower().endswith(".wav")
            ]
            if candidates:
                src_audio = os.path.join(audio_dir, candidates[0])
                dst_audio = os.path.join(data_root, split, "audio", f"{sample_id}.wav")
                os.makedirs(os.path.dirname(dst_audio), exist_ok=True)
                shutil.copy2(src_audio, dst_audio)

        # ------- 文本与标签 -------
        # 实际 MAFW 的文本/标签格式需要根据官方文档解析。
        # 这里给出一个“占位示例”，默认写 neutral + 0.0,0.0，
        # 方便你先跑通代码，后续可根据官方标注文件再生成真实标签。
        dst_text  = os.path.join(data_root, split, "text",   f"{sample_id}.txt")
        dst_label = os.path.join(data_root, split, "labels", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_text), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        # 占位文本
        with open(dst_text, "w", encoding="utf-8") as f:
            f.write("This is a placeholder text for MAFW sample.\n")

        # 占位标签（请日后用真实 MAFW 标注替换）
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("neutral\n")
            f.write("0.0,0.0\n")

print("MAFW 数据整理到 data/ 目录完成（当前使用的是占位标签）。")
PY
```

> 重要：上述脚本中**标签是占位的**，为了先把项目跑通，后续需要你根据 MAFW 官方给的标签文件生成真实的 `labels/*.txt`。

### 6.2 MPDB 示例：整理生理信号 + 标签

假设 MPDB 解压后结构类似：

```bash
downloads/MPDB/
├── physio_raw/
│   ├── subj01_trial01.npy
│   ├── subj01_trial02.npy
│   └── ...
└── labels_raw/
    ├── subj01_trial01.txt
    ├── subj01_trial02.txt
    └── ...
```

可以运行：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

python - <<'PY'
import os, shutil

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
mpdb_root    = os.path.join(project_root, "downloads", "MPDB")
data_root    = os.path.join(project_root, "data")

physio_dir = os.path.join(mpdb_root, "physio_raw")
label_dir  = os.path.join(mpdb_root, "labels_raw")

physio_files = sorted([
    f for f in os.listdir(physio_dir)
    if f.lower().endswith(".npy")
])

print("发现生理信号文件数量：", len(physio_files))

n       = len(physio_files)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)
n_test  = n - n_train - n_val

splits = (
    ("train", physio_files[:n_train]),
    ("val",   physio_files[n_train:n_train + n_val]),
    ("test",  physio_files[n_train + n_val:]),
)

for split, files in splits:
    print(f"处理 {split} 集，样本数：{len(files)}")
    for idx, pf in enumerate(files, start=1):
        sample_id = f"mpdb_{split}_{idx:04d}"

        # 生理信号
        src_phys = os.path.join(physio_dir, pf)
        dst_phys = os.path.join(data_root, split, "physiological", f"{sample_id}.npy")
        os.makedirs(os.path.dirname(dst_phys), exist_ok=True)
        shutil.copy2(src_phys, dst_phys)

        # 标签：假设同名 .txt 存在
        base = os.path.splitext(pf)[0]
        candidates = [
            f for f in os.listdir(label_dir)
            if f.startswith(base) and f.lower().endswith(".txt")
        ]
        dst_label = os.path.join(data_root, split, "labels", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        if candidates:
            src_label = os.path.join(label_dir, candidates[0])
            shutil.copy2(src_label, dst_label)
        else:
            # 若暂时找不到标签，同样写占位 neutral + 0.0,0.0
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("neutral\n")
                f.write("0.0,0.0\n")

print("MPDB 生理信号整理到 data/ 目录完成。")
PY
```

### 6.3 CREMA-D 示例：整理视频 + 音频 + 标签

**CREMA-D数据集特点**：
- 包含视频和音频文件
- 文件名包含情感标签信息（如：`1001_DFA_ANG_XX.wav` 表示 Angry）
- 6种情感类别：Happy, Sad, Angry, Fear, Disgust, Neutral

**使用整理脚本（推荐）**：

项目已提供专门的整理脚本 `scripts/organize_crema_d.py`：

**如果音频文件已整理（推荐）**：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

# 运行整理脚本（自动检测并跳过音频处理）
python scripts/organize_crema_d.py

# 或者显式指定跳过音频
python scripts/organize_crema_d.py --skip-audio
```

**如果音频文件未整理**：

```bash
# 正常运行，会同时整理视频和音频
python scripts/organize_crema_d.py
```

**脚本功能**：
1. 自动扫描 `downloads/crema-d-emotional-multimodal-dataset/` 目录（支持 `content/CREMA-D/VideoFlash/` 结构）
2. 识别视频文件（支持 `.flv`, `.mp4`, `.avi`, `.mov`, `.mkv` 格式）
3. 识别音频文件（支持 `.wav`, `.mp3`, `.flac`, `.m4a` 格式）
4. 从文件名提取情感标签（如 `ANG` → `angry`）
5. 自动检测已存在的音频文件，跳过重复处理
6. 按 8:1:1 划分 train/val/test
7. 复制文件到 `data/` 目录并统一命名
8. 生成标签文件（情感类别 + 效价/唤醒度）

**重要特性**：
- **支持.flv格式**：CREMA-D的视频文件是`.flv`格式，脚本已支持
- **自动跳过音频**：如果检测到`data/audio`目录已有文件，自动跳过音频处理
- **路径自动查找**：支持多种数据集路径结构，包括`content/CREMA-D/VideoFlash/`

**脚本执行步骤**：

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

**脚本输出示例**：

```
============================================================
CREMA-D数据集整理脚本
============================================================
扫描目录: /home/lizhichun_24/sda1/code/multimodal/project/downloads/crema-d-emotional-multimodal-dataset
找到 7442 个文件

整理 7442 个样本
数据集划分: train=5953, val=744, test=745

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

数据已整理到: /home/lizhichun_24/sda1/code/multimodal/project/data
```

**验证整理结果**：

```bash
# 检查文件数量
ls data/train/video | wc -l
ls data/train/audio | wc -l
ls data/train/labels | wc -l

# 查看一个样本
ls data/train/video/ | head -5
ls data/train/labels/ | head -5

# 查看标签文件内容
cat data/train/labels/crema_train_0001.txt
# 输出示例：
# angry
# -0.7,0.8
```

**手动整理方法（如果脚本不适用）**：

如果数据集结构不同，可以手动整理：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

python - <<'PY'
import os, shutil, random

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
crema_root = os.path.join(project_root, "downloads", "crema-d-emotional-multimodal-dataset")
data_root = os.path.join(project_root, "data")

# 1. 查找所有视频和音频文件
video_files = []
audio_files = []

for root, dirs, files in os.walk(crema_root):
    for f in files:
        filepath = os.path.join(root, f)
        if f.lower().endswith(('.mp4', '.avi', '.mov')):
            video_files.append(filepath)
        elif f.lower().endswith(('.wav', '.mp3')):
            audio_files.append(filepath)

print(f"找到 {len(video_files)} 个视频文件")
print(f"找到 {len(audio_files)} 个音频文件")

# 2. 按文件名匹配视频和音频（假设文件名前缀相同）
# 这里需要根据实际文件名格式调整

# 3. 划分数据集（8:1:1）
all_files = video_files  # 以视频为主
random.seed(42)
random.shuffle(all_files)

n = len(all_files)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

splits = {
    'train': all_files[:n_train],
    'val': all_files[n_train:n_train + n_val],
    'test': all_files[n_train + n_val:]
}

# 4. 复制文件并生成标签
for split, files in splits.items():
    print(f"处理 {split} 集，样本数：{len(files)}")
    for idx, filepath in enumerate(files, start=1):
        sample_id = f"crema_{split}_{idx:04d}"
        filename = os.path.basename(filepath)
        
        # 复制视频
        ext = os.path.splitext(filename)[1]
        dst_video = os.path.join(data_root, split, "video", f"{sample_id}{ext}")
        os.makedirs(os.path.dirname(dst_video), exist_ok=True)
        shutil.copy2(filepath, dst_video)
        
        # 查找对应的音频文件（根据文件名匹配）
        base = os.path.splitext(filename)[0]
        audio_match = None
        for audio_file in audio_files:
            if os.path.splitext(os.path.basename(audio_file))[0] == base:
                audio_match = audio_file
                break
        
        if audio_match:
            audio_ext = os.path.splitext(audio_match)[1]
            dst_audio = os.path.join(data_root, split, "audio", f"{sample_id}{audio_ext}")
            os.makedirs(os.path.dirname(dst_audio), exist_ok=True)
            shutil.copy2(audio_match, dst_audio)
        
        # 从文件名提取情感标签
        emotion = 'neutral'  # 默认值
        filename_upper = filename.upper()
        if 'HAP' in filename_upper or 'HAPPY' in filename_upper:
            emotion = 'happy'
        elif 'SAD' in filename_upper:
            emotion = 'sad'
        elif 'ANG' in filename_upper or 'ANGRY' in filename_upper:
            emotion = 'angry'
        elif 'FEA' in filename_upper or 'FEAR' in filename_upper:
            emotion = 'fear'
        elif 'DIS' in filename_upper or 'DISGUST' in filename_upper:
            emotion = 'disgust'
        
        # 生成标签文件
        valence_arousal = {
            'happy': (0.8, 0.7),
            'sad': (-0.6, -0.3),
            'angry': (-0.7, 0.8),
            'fear': (-0.5, 0.9),
            'disgust': (-0.7, 0.5),
            'neutral': (0.0, 0.0)
        }
        valence, arousal = valence_arousal.get(emotion, (0.0, 0.0))
        
        dst_label = os.path.join(data_root, split, "labels", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write(f"{emotion}\n")
            f.write(f"{valence},{arousal}\n")
        
        # 生成文本文件（占位）
        dst_text = os.path.join(data_root, split, "text", f"{sample_id}.txt")
        os.makedirs(os.path.dirname(dst_text), exist_ok=True)
        with open(dst_text, "w", encoding="utf-8") as f:
            f.write(f"Audio transcription for {sample_id}\n")

print("CREMA-D 数据整理到 data/ 目录完成。")
PY
```

**注意事项**：

1. **视频格式**：CREMA-D的视频文件是`.flv`格式，脚本已支持（也支持其他常见视频格式）
2. **文件名格式**：CREMA-D的文件名通常包含情感标签，如 `1001_DFA_ANG_XX.flv`，脚本会自动解析
3. **音频已整理**：如果音频文件已经整理到`data/audio`目录，脚本会自动检测并跳过音频处理
4. **路径结构**：脚本支持多种数据集路径结构：
   - `downloads/crema-d-emotional-multimodal-dataset/content/CREMA-D/VideoFlash/`
   - `downloads/crema-d-emotional-multimodal-dataset/`
   - 其他常见路径
5. **标签映射**：如果文件名格式不同，可能需要修改脚本中的 `parse_crema_d_filename` 函数
6. **文本数据**：CREMA-D通常没有文本数据，脚本会生成占位文本文件
7. **生理信号**：CREMA-D不包含生理信号，该目录会保持为空

**如果视频文件没有被整理**：

如果运行脚本后视频文件没有被整理到`data/video`目录，请检查：

1. **确认数据集路径**：
   ```bash
   find downloads/ -name "*.flv" | head -10
   ```

2. **检查脚本输出**：查看脚本是否找到了视频文件
   - 应该看到类似：`找到 XXX 个文件` 或 `只处理视频文件: XXX 个`

3. **手动指定路径**：如果数据集在其他位置，可以修改脚本中的 `DOWNLOAD_DIR` 变量

4. **检查文件权限**：确保对目标目录有写权限
   ```bash
   ls -ld data/train/video
   chmod 755 data/train/video  # 如果需要
   ```

**如果数据集结构不同**：

如果脚本无法正确识别数据集结构，可以：

1. **查看数据集实际结构**：
   ```bash
   cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
   find crema-d-emotional-multimodal-dataset -type f | head -20
   ls -la crema-d-emotional-multimodal-dataset/
   ```

2. **根据实际结构修改脚本**：将实际的文件结构信息提供给脚本，调整文件查找逻辑

---

## 七、（可选）生成 CSV 索引文件

`project/data/dataset.py` 支持两种数据组织方式：

1. **目录结构方式**（我们已经搭好 `data/train/...` 等目录即可直接使用）  
2. **CSV 文件方式**（`data/train.csv` / `val.csv` / `test.csv`）

如果你更喜欢 CSV 可视化管理样本列表，可以根据 `详细文档.md` 的示例编写或使用项目脚本（例如 `scripts/create_data_csv.py`，如果仓库中已有）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

python scripts/create_data_csv.py --data_dir data/
```

生成后的 `data/train.csv` 格式应类似：

```csv
sample_id,video_path,audio_path,physiological_path,text_path,label_path
sample_001,video/sample_001.mp4,audio/sample_001.wav,physiological/sample_001.npy,text/sample_001.txt,labels/sample_001.txt
...
```

> 如果脚本不存在或参数不匹配，运行时会报错；此时可以根据错误信息，对照 `dataset.py` 中 `_load_data_list` 的说明重新编写一个简单的 CSV 生成脚本。

---

## 八、验证数据集是否准备成功

### 8.1 目录与文件数量检查

在远程终端执行：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project

ls data/train/video | head
ls data/train/audio | head
ls data/train/physiological | head
ls data/train/text | head
ls data/train/labels | head
```

快速统计文件数量：

```bash
python - <<'PY'
import os
base = "data/train"
for sub in ["video","audio","physiological","text","labels"]:
    p = os.path.join(base, sub)
    if os.path.exists(p):
        print(sub, "EXIST, count =", len(os.listdir(p)))
    else:
        print(sub, "MISSING")
PY
```

### 8.2 使用 MultimodalDataset 试读一个样本

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

python - <<'PY'
import os, yaml, torch
from data.dataset import MultimodalDataset

project_root = "/home/lizhichun_24/sda1/code/multimodal/project"
config_path  = os.path.join(project_root, "config", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset = MultimodalDataset(
    data_dir=os.path.join(project_root, "data"),
    split="train",
    config=config,
)

print("样本数量：", len(dataset))

sample = dataset[0]
for k, v in sample.items():
    if torch.is_tensor(v):
        print(k, "shape:", tuple(v.shape))
    else:
        print(k, "=", v)
PY
```

如果：

- 正常输出样本数量，且 `video` / `audio` / `physiological` / `text_input_ids` / `emotion_label` 等字段都有合理的 `shape` 或数值，说明数据准备成功。  
- 出现 `FileNotFoundError` 或路径相关错误，说明文件名或目录结构与 `dataset.py` 预期不一致，需要回到第六节调整整理脚本。  
- 出现标签解析错误，检查 `labels/*.txt` 是否满足“第一行类别、第二行数值”的格式。  

---

## 九、从预训练到微调：如何使用这些数据

当数据准备就绪后，可以按照 `project/README.md` 和 `project/详细文档.md` 的训练流程进行实验。

### 9.1 预训练阶段

预训练使用开源多模态情感数据集（CREMA-D、MELD、CMU-MOSEI），配置在 `config/config.yaml` 中的：

```yaml
training:
  pretrain:
    enabled: true
    datasets: ["CREMA-D", "MELD", "CMU-MOSEI"]  # 使用开源数据集
```

**数据集适配特性**：

项目已实现自动数据集适配功能：

1. **自动数据集检测**：通过文件命名前缀自动识别数据集类型
   - `crema_train_0001.flv` → 自动识别为CREMA-D数据集
   - `meld_train_0001.mp4` → 自动识别为MELD数据集

2. **动态情感类别映射**：
   - CREMA-D：6种情感类别（自动映射）
   - MELD：7种情感类别（自动映射）
   - CMU-MOSEI：7种情感类别（标准映射）

3. **多格式支持**：
   - 视频：`.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`
   - 音频：`.wav`, `.mp3`, `.flac`, `.m4a`
   - 自动识别文件扩展名

4. **缺失模态处理**：
   - 自动处理缺失的生理信号（使用零填充）
   - 优雅处理缺失的音频或文本

**训练命令**：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

# 方式1：自动检测数据集（推荐）
python scripts/train.py --config config/config.yaml --mode pretrain

# 方式2：指定数据集
python scripts/train.py --config config/config.yaml --mode pretrain --dataset crema
python scripts/train.py --config config/config.yaml --mode pretrain --dataset meld
python scripts/train.py --config config/config.yaml --mode pretrain --dataset mosei
```

**配置文件说明**：

数据集配置在 `config/config.yaml` 的 `datasets` 节中：

```yaml
datasets:
  crema:
    emotion_classes: 6
    emotion_map: {...}
    video_format: ".flv"
    has_physiological: false
  
  meld:
    emotion_classes: 7
    emotion_map: {...}
    video_format: ".mp4"
    has_physiological: false
    has_text: true
```

如果需要使用其他数据集，可以在配置文件中添加相应的数据集配置。

### 9.2 微调阶段

微调使用驾驶员场景数据集（MPDB、DEFE 等）：

```yaml
training:
  finetune:
    enabled: true
    datasets: ["MPDB"]
    epochs: 30
    freeze_backbone: false
```

在预训练完成后，使用最佳预训练模型进行微调：

```bash
python scripts/train.py \
    --config config/config.yaml \
    --mode finetune \
    --resume checkpoints/pretrain_best_model.pth
```

### 9.3 推理与评估

详见 `project/详细文档.md` 第 9 章，基本命令为：

```bash
# 评估
python scripts/evaluate.py \
    --config config/config.yaml \
    --model_path checkpoints/best_model.pth \
    --test_data data/test/

# 推理
python scripts/inference.py \
    --config config/config.yaml \
    --model_path checkpoints/best_model.pth \
    --input_dir data/test/ \
    --output_path results/inference_results.json
```

---

## 十、数据集适配说明

### 10.1 数据集自动适配功能

项目已实现完整的数据集适配功能，支持CREMA-D、MELD、CMU-MOSEI等开源数据集：

**核心特性**：

1. **自动数据集检测**：
   - 通过文件命名前缀自动识别数据集类型（如 `crema_`, `meld_`）
   - 无需手动配置，自动应用对应的情感类别映射

2. **多格式支持**：
   - 视频格式：`.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`（CREMA-D使用.flv）
   - 音频格式：`.wav`, `.mp3`, `.flac`, `.m4a`
   - 自动识别文件扩展名，支持动态格式匹配

3. **数据集特定配置**：
   - CREMA-D：6种情感类别，FLV视频格式
   - MELD：7种情感类别，MP4视频格式，包含文本标注
   - CMU-MOSEI：7种情感类别，MP4视频格式，包含文本标注

4. **缺失模态处理**：
   - 自动处理缺失的生理信号（使用零填充）
   - 优雅处理缺失的音频或文本（返回None，模型自动处理）

### 10.2 代码修改说明

**主要修改的文件**：

1. **`data/dataset.py`**：
   - 扩展视频格式支持（添加`.flv`等格式）
   - 实现数据集特定的情感类别映射
   - 改进文件路径查找逻辑，支持动态扩展名识别
   - 添加数据集自动检测功能

2. **`data/preprocess.py`**：
   - 增强`.flv`格式支持（需要ffmpeg）
   - 改进错误处理，对不支持的格式给出明确提示
   - 支持动态文件扩展名识别

3. **`config/config.yaml`**：
   - 添加`datasets`配置节，定义各数据集的情感类别映射
   - 支持数据集特定的预处理参数

4. **`scripts/train.py`**：
   - 添加`--dataset`参数，支持指定数据集
   - 根据数据集配置动态调整模型输出类别数

### 10.3 使用新数据集的步骤

1. **下载和整理数据集**（参考前面的章节）：
   ```bash
   # CREMA-D
   python scripts/organize_crema_d.py
   
   # MELD
   python scripts/organize_meld.py
   ```

2. **验证数据格式**：
   ```bash
   # 检查文件是否存在
   ls data/train/video/ | head -10
   ls data/train/labels/ | head -10
   
   # 验证数据加载
   python -c "from data.dataset import MultimodalDataset; import yaml; config = yaml.safe_load(open('config/config.yaml')); ds = MultimodalDataset('data/', 'train', config); print(f'Samples: {len(ds)}'); print(ds[0])"
   ```

3. **开始训练**：
   ```bash
   # 自动检测数据集
   python scripts/train.py --config config/config.yaml --mode pretrain
   
   # 或指定数据集
   python scripts/train.py --config config/config.yaml --mode pretrain --dataset crema
   ```

### 10.4 注意事项

1. **FLV格式支持**：
   - CREMA-D使用`.flv`格式，需要系统安装ffmpeg
   - 如果OpenCV无法读取FLV文件，请安装ffmpeg：
     ```bash
     # Ubuntu/Debian
     sudo apt-get install ffmpeg
     
     # 验证安装
     ffmpeg -version
     ```

2. **情感类别映射**：
   - 不同数据集的情感类别会自动映射到项目的标准类别
   - MELD的`joy`映射到`happy`，`sadness`映射到`sad`等
   - 映射规则在`dataset.py`的`DATASET_EMOTION_MAPS`中定义

3. **缺失模态处理**：
   - 如果数据集缺少某个模态（如生理信号），模型会自动使用零填充
   - 确保数据目录结构正确，即使文件不存在也要有对应的目录

## 十一、遇到问题时的排查建议

1. **无法 wget 下载**：  
   - 检查链接是否需要登录/表单；  
   - 尝试在本地浏览器下载 + `scp` 上传到服务器。  
2. **解压后结构看不懂**：  
   - 在远程执行 `cd downloads/XXX && find . -maxdepth 3 -type f | head -n 50` 查看结构；  
   - 再根据文件名/子目录编写或修改第六节的整理脚本。  
3. **训练时报"文件找不到"**：  
   - 检查 `config.yaml` 中 `data.root_dir` 是否为 `"data/"`；  
   - 检查 `data/train/...` 目录下是否真的有文件；  
   - 检查 `labels/*.txt` 是否存在并格式正确。
4. **FLV格式无法读取**：
   - 检查ffmpeg是否安装：`ffmpeg -version`
   - 如果未安装，请安装ffmpeg：`sudo apt-get install ffmpeg`
   - 检查OpenCV是否支持FLV：`python -c "import cv2; print(cv2.getBuildInformation())"`
5. **数据集无法自动检测**：
   - 检查文件命名格式是否为 `{dataset}_{split}_{idx}.{ext}`
   - 确保数据集名称在`DATASET_EMOTION_MAPS`中定义
   - 可以手动指定数据集：`--dataset crema`  

如果你在某一个具体数据集（例如 MAFW 或 MPDB）上，已经完成了解压，但不确定如何写整理脚本，可以在远程执行：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
find downloads/MAFW -maxdepth 3 -type f | head -n 50
```

将输出复制出来，让智能助手根据实际结构为你定制一个**可以直接复制执行**的整理脚本。

---

## 十一、混合数据集训练优化方案

### 11.1 概述

当使用多个开源数据集（CREMA-D、MELD、CMU-MOSEI）进行混合训练时，由于数据集之间存在异构性（域偏移、类别不平衡、标注不一致等），需要采用专门的优化策略。本项目已实现了以下优化方案：

1. **数据集平衡采样**：确保每个batch包含来自不同数据集的样本
2. **类别平衡损失**：处理类别不平衡问题
3. **域适应机制**：学习域不变特征，提高跨域泛化能力
4. **数据集特定归一化**：减少域间特征分布差异
5. **混合训练策略**：优化训练流程（交替训练、渐进式训练、课程学习）

### 11.2 配置混合数据集训练

在 `config/config.yaml` 中配置混合数据集训练优化：

```yaml
training:
  # 数据集平衡采样
  sampling:
    enabled: true  # 启用平衡采样
    mode: "proportional"  # "proportional"（按比例）或 "uniform"（均匀）
    shuffle: true
    seed: null
  
  # 损失函数配置
  loss:
    # 类别平衡损失
    use_class_balanced: true
    class_balance_beta: 0.9999
    
    # Focal Loss（替代类别平衡损失）
    use_focal_loss: false
    focal_alpha: 1.0
    focal_gamma: 2.0
    
    # 域适应损失
    use_domain_adaptation: true
    domain_loss_weight: 0.1

model:
  # 域适应配置
  domain_adaptation:
    enabled: true
    num_domains: 3  # CREMA-D, MELD, CMU-MOSEI
    hidden_dim: 256
    lambda_param: 1.0
    adaptive_lambda: true  # 自适应lambda
  
  # 数据集特定归一化
  dataset_normalization:
    enabled: true
    num_datasets: 3
    momentum: 0.1
```

### 11.3 运行混合数据集训练

使用训练脚本进行混合数据集训练：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/train.py --config config/config.yaml --mode pretrain
```

训练脚本会自动：
1. 检测数据集（从文件名中提取dataset_id）
2. 应用平衡采样（如果启用）
3. 计算类别平衡损失和域适应损失
4. 应用数据集特定归一化

### 11.4 训练策略选择

项目支持三种混合训练策略（在 `utils/training_strategies.py` 中实现）：

1. **交替训练（Alternating Training）**：
   - 每个epoch交替使用不同数据集
   - 确保模型充分学习每个数据集的特征

2. **渐进式训练（Progressive Training）**：
   - 先单数据集训练，再混合训练
   - 逐步增加数据集的多样性

3. **课程学习（Curriculum Learning）**：
   - 从简单数据集到复杂数据集
   - 逐步增加训练难度

### 11.5 监控训练过程

训练过程中会输出以下信息：

```
Epoch 0/50
Train Loss: 2.3456
Train Loss Breakdown: {'classification': 1.2345, 'regression': 0.5678, 'domain': 0.1234}
Validation Loss: 2.1234, Metrics: {'accuracy': 0.65, 'f1': 0.62}
```

- `classification`: 情感分类损失
- `regression`: 情绪维度回归损失
- `domain`: 域适应损失（如果启用）

### 11.6 预期效果

使用混合数据集训练优化方案后，预期能够：

- **跨域准确率提升**：在跨数据集测试中，准确率提升5-10%
- **类别平衡改善**：少数类别的F1分数提升10-15%
- **泛化能力增强**：在新数据集上的表现提升8-12%

### 11.7 详细文档

更多关于混合数据集训练优化的详细信息，请参考：

- `docs/MIXED_DATASET_TRAINING_ANALYSIS.md`：问题分析与优化方案详细说明
- `data/balanced_sampler.py`：平衡采样器实现
- `models/balanced_loss.py`：类别平衡损失实现
- `models/domain_adaptation.py`：域适应模块实现
- `utils/training_strategies.py`：混合训练策略实现

### 11.8 故障排查

如果遇到混合数据集训练问题：

1. **数据集ID未正确识别**：
   - 检查文件名格式是否为 `{dataset}_{split}_{idx}.{ext}`
   - 确保数据集名称在 `dataset.py` 的 `DATASET_EMOTION_MAPS` 中定义

2. **平衡采样不工作**：
   - 检查 `config.yaml` 中 `sampling.enabled` 是否为 `true`
   - 查看训练日志中的采样统计信息

3. **域适应损失为0或NaN**：
   - 检查 `dataset_ids` 是否正确传递到模型
   - 确保 `domain_adaptation.enabled` 为 `true`

4. **类别不平衡仍然严重**：
   - 尝试调整 `class_balance_beta` 参数（增大beta值）
   - 或使用 Focal Loss（设置 `use_focal_loss: true`）