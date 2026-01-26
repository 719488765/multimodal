# CREMA-D数据集下载执行指南

## 快速开始

### 在远程服务器上执行以下命令：

```bash
# 1. 连接到远程服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录
cd /home/lizhichun_24/sda1/code/multimodal/project

# 3. 激活虚拟环境
conda activate myenv310

# 4. 安装kagglehub（如果还没安装）
pip install kagglehub

# 5. 确保kaggle.json已配置（如果还没配置，参考use_data.md步骤2-3）
ls -la ~/.kaggle/kaggle.json

# 6. 运行下载脚本
python scripts/download_crema_d.py
```

## 详细步骤

### 步骤1：准备环境

确保已安装kagglehub：

```bash
conda activate myenv310
pip install kagglehub
python -c "import kagglehub; print('kagglehub安装成功')"
```

### 步骤2：配置Kaggle认证

确保`~/.kaggle/kaggle.json`文件存在且权限正确：

```bash
# 检查文件是否存在
ls -la ~/.kaggle/kaggle.json

# 如果不存在，需要上传（参考use_data.md）
# 如果存在，确保权限正确
chmod 600 ~/.kaggle/kaggle.json
```

### 步骤3：执行下载脚本

**方法一：使用默认路径（推荐）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python scripts/download_crema_d.py
```

数据集将下载到：`/home/lizhichun_24/sda1/code/multimodal/project/downloads`

**方法二：指定自定义路径**

```bash
python scripts/download_crema_d.py /path/to/custom/downloads
```

### 步骤4：验证下载结果

```bash
# 检查下载的文件
ls -lh /home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 查看数据集结构
find /home/lizhichun_24/sda1/code/multimodal/project/downloads -type f | head -20

# 检查数据集大小
du -sh /home/lizhichun_24/sda1/code/multimodal/project/downloads/*
```

## 脚本功能说明

脚本会自动尝试三种下载方法：

1. **方法1**：使用`path`参数直接指定下载位置（如果kagglehub版本支持）
2. **方法2**：使用环境变量`KAGGLEHUB_HOME`指定缓存目录
3. **方法3**：下载到默认位置后移动到目标路径

脚本会自动选择可用的方法，无需手动干预。

## 常见问题

### 问题1：找不到kagglehub模块

**错误信息**：`ModuleNotFoundError: No module named 'kagglehub'`

**解决方案**：
```bash
conda activate myenv310
pip install kagglehub
```

### 问题2：认证失败

**错误信息**：`401 Unauthorized` 或 `Authentication failed`

**解决方案**：
```bash
# 检查kaggle.json文件
cat ~/.kaggle/kaggle.json

# 检查文件权限
chmod 600 ~/.kaggle/kaggle.json

# 验证认证
python -c "import kagglehub; kagglehub.dataset_download('orvile/crema-d-emotional-multimodal-dataset')"
```

### 问题3：存储空间不足

**错误信息**：`No space left on device`

**解决方案**：
```bash
# 检查可用空间
df -h /home/lizhichun_24/sda1

# 如果空间不足，可以指定其他路径
python scripts/download_crema_d.py /path/to/larger/disk/downloads
```

### 问题4：下载中断

**解决方案**：kagglehub支持断点续传，直接重新运行脚本即可：
```bash
python scripts/download_crema_d.py
```

## 下载后的处理

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

## 完整执行示例

```bash
# 1. SSH连接到服务器
ssh -p 1022 lizhichun_24@49.233.89.203

# 2. 进入项目目录并激活环境
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

# 3. 检查环境
python -c "import kagglehub; print('kagglehub版本:', kagglehub.__version__)"
ls -la ~/.kaggle/kaggle.json

# 4. 运行下载脚本
python scripts/download_crema_d.py

# 5. 等待下载完成（可能需要几分钟到几十分钟，取决于网络速度）

# 6. 验证下载结果
ls -lh downloads/
du -sh downloads/*
```

## 注意事项

1. **网络连接**：确保服务器可以访问Kaggle（可能需要配置代理）
2. **存储空间**：CREMA-D数据集可能较大，确保有足够空间（建议至少10GB）
3. **下载时间**：根据网络速度，下载可能需要较长时间，建议使用`screen`或`tmux`保持会话
4. **断点续传**：如果下载中断，重新运行脚本即可继续下载

## 使用screen保持会话（推荐）

如果担心SSH连接断开导致下载中断，可以使用screen：

```bash
# 安装screen（如果未安装）
# sudo apt install screen  # Ubuntu/Debian
# sudo yum install screen   # CentOS/RHEL

# 创建新的screen会话
screen -S download_crema_d

# 在screen中执行下载
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310
python scripts/download_crema_d.py

# 如果需要暂时离开，按 Ctrl+A 然后按 D（detach）
# 重新连接：screen -r download_crema_d
# 查看所有会话：screen -ls
```

