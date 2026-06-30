# CMU-MOSEI 数据集完整安装与整合指南（完整版 31GB）

本指南将详细指导你完成 CMU-MOSEI **完整版（31GB）**数据集的下载、上传、整合，以及后续的训练和微调配置。

## ⚡ 快速参考（31GB 大文件）

### 关键信息
- **文件大小**：约 31GB
- **总磁盘空间需求**：至少 70GB（32GB zip + 40GB 解压后）
- **预计总时间**：下载 1-8 小时 + 上传 1-8 小时 + 解压 10-60 分钟

### 推荐工具
- **下载**：浏览器直接下载（支持断点续传）
- **上传**：FileZilla 或 `rsync`（支持断点续传）
- **解压**：使用 `nohup` 后台解压（避免 SSH 断开中断）

### 关键命令速查

```bash
# 1. 检查服务器磁盘空间
df -h /home/lizhichun_24/sda1

# 2. 上传文件（使用 rsync，支持断点续传）
rsync -avz --progress ~/Downloads/cmu-mosei.zip lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 3. 后台解压（推荐）
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
nohup unzip -q cmu-mosei.zip -d CMU_MOSEI_raw > unzip.log 2>&1 &

# 4. 监控解压进度
tail -f unzip.log
```

---

## 📋 目录

1. [步骤一：本地浏览器下载数据集](#步骤一本地浏览器下载数据集)
2. [步骤二：上传到服务器](#步骤二上传到服务器)
3. [步骤三：在服务器上解压](#步骤三在服务器上解压)
4. [步骤四：运行整理脚本](#步骤四运行整理脚本)
5. [步骤五：验证数据整合结果](#步骤五验证数据整合结果)
6. [步骤六：配置训练脚本](#步骤六配置训练脚本)
7. [步骤七：开始训练和微调](#步骤七开始训练和微调)

---

## 步骤一：本地浏览器下载数据集（完整版 31GB）

### 1.1 访问 Kaggle 数据集页面

1. 在你的**本地电脑**（Windows/Mac）上，打开浏览器
2. 登录你的 Kaggle 账号
3. 访问**完整版数据集页面**：
   - **完整版（约 31GB）**：https://www.kaggle.com/datasets/samarwarsi/cmu-mosei

### 1.2 下载数据集（31GB 大文件）

#### ⚠️ 重要提示

- **文件大小**：约 31GB（31,201,301,641 bytes）
- **下载时间估算**：
  - 100 Mbps 网络：约 40-50 分钟
  - 50 Mbps 网络：约 1.5-2 小时
  - 10 Mbps 网络：约 6-8 小时
- **磁盘空间需求**：
  - 下载 zip 文件：至少 32GB 可用空间
  - 解压后：至少 35-40GB 可用空间
  - **总计需要：至少 70GB 可用空间**

#### 下载步骤

1. **检查本地磁盘空间**：
   ```powershell
   # Windows PowerShell
   Get-PSDrive C | Select-Object Used,Free
   ```
   ```bash
   # Mac/Linux
   df -h ~/Downloads
   ```

2. **在数据集页面，点击右上角的 "Download" 按钮**

3. **浏览器会开始下载 `cmu-mosei.zip` 文件**

4. **下载注意事项**：
   - ✅ 保持网络连接稳定
   - ✅ 不要关闭浏览器标签页
   - ✅ 如果下载中断，大多数浏览器支持断点续传（重新点击下载即可）
   - ✅ 建议使用下载管理器（如 IDM、Free Download Manager）以获得更好的断点续传支持
   - ⚠️ 下载过程中不要休眠或关闭电脑

5. **等待下载完成**（可能需要数小时，请耐心等待）

### 1.3 如果下载速度太慢（6.3 KB/s 或更慢）的解决方案

如果你的下载速度只有几 KB/s，预计需要数天才能完成，请尝试以下方案：

#### 方案 A：直接在服务器上使用 Kaggle API 下载（强烈推荐）

**优势**：
- ✅ 服务器网络通常比本地网络更稳定
- ✅ 可以后台下载，不占用本地资源
- ✅ 支持断点续传
- ✅ 下载速度通常更快

**步骤**：

1. **在服务器上配置 Kaggle API**（如果还没配置）：
   ```bash
   ssh lizhichun_24@49.233.89.203
   mkdir -p ~/.kaggle
   # 将你的 kaggle.json 上传到服务器（参考步骤二的方法）
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **在服务器上直接下载**：
   ```bash
   cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
   
   # 使用 nohup 后台下载，支持断点续传
   nohup kaggle datasets download -d samarwarsi/cmu-mosei -p . > kaggle_download.log 2>&1 &
   
   # 监控下载进度
   tail -f kaggle_download.log
   ```

3. **检查下载进度**：
   ```bash
   # 查看文件大小变化
   watch -n 10 'ls -lh cmu-mosei.zip 2>/dev/null || echo "文件还未创建"'
   
   # 或查看日志
   tail -20 kaggle_download.log
   ```

**如果下载中断**：
- 直接重新运行 `kaggle datasets download` 命令，Kaggle CLI 支持断点续传

#### 方案 B：使用专业下载管理器（本地电脑）

**Windows**：
- **IDM (Internet Download Manager)**：https://www.internetdownloadmanager.com/
  - 支持多线程下载，速度提升明显
  - 支持断点续传
  - 可以设置代理

- **Free Download Manager**：https://www.freedownloadmanager.org/
  - 免费开源
  - 支持多线程下载
  - 支持断点续传

**Mac**：
- **Folx**：https://mac.eltima.com/download-manager.html
- **Downie**：https://software.charliemonroe.net/downie/

**使用方法**：
1. 安装下载管理器
2. 在 Kaggle 数据集页面，右键点击 "Download" 按钮
3. 选择 "复制链接地址"
4. 在下载管理器中添加下载任务
5. 配置多线程下载（通常设置为 8-16 线程）

#### 方案 C：检查网络问题并优化

1. **检查网络速度**：
   ```bash
   # 在本地电脑测试网络速度
   # Windows: 访问 https://www.speedtest.net/
   # Mac/Linux: 安装 speedtest-cli
   pip install speedtest-cli
   speedtest-cli
   ```

2. **尝试更换网络**：
   - 如果使用 WiFi，尝试使用有线网络
   - 如果使用移动网络，尝试使用固定宽带
   - 尝试使用手机热点（如果手机网络更好）

3. **使用 VPN 或代理**（如果 Kaggle 在你的地区被限速）：
   - 配置系统代理或浏览器代理
   - 使用 VPN 服务

#### 方案 D：使用其他数据源（备选方案）

如果以上方案都不行，可以考虑：

1. **使用较小的数据集版本**（3GB）：
   ```bash
   # 在服务器上直接下载较小的版本
   kaggle datasets download -d gnurtqh/cmu-mosei -p .
   ```
   这个版本虽然数据量较小，但通常也足够使用。

2. **联系数据集作者**：
   - 在 Kaggle 数据集页面留言，询问是否有其他下载方式
   - 或查看是否有镜像站点

#### 方案 E：分批下载（如果数据集支持）

某些 Kaggle 数据集支持分批下载，可以：
1. 查看数据集页面是否有 "Download All" 和单独文件下载选项
2. 如果有单独文件，可以分批下载，然后合并

---

### 推荐方案优先级

1. **首选**：方案 A（服务器上直接下载）- 通常最快最稳定
2. **次选**：方案 B（使用下载管理器）- 如果必须在本地下载
3. **备选**：方案 C（优化网络）- 如果网络确实有问题
4. **最后**：方案 D（使用小版本或其他数据源）

---

6. **验证下载文件**：
   ```powershell
   # Windows PowerShell
   (Get-Item "C:\Users\你的用户名\Downloads\cmu-mosei.zip").Length / 1GB
   ```
   ```bash
   # Mac/Linux
   ls -lh ~/Downloads/cmu-mosei.zip
   ```
   文件大小应该接近 31GB（约 29-31 GB）

7. **记住下载文件的保存位置**（通常是 `Downloads` 文件夹）

---

## 步骤二：上传到服务器（31GB 大文件）

### ⚠️ 上传前准备

1. **检查服务器磁盘空间**（在服务器上执行）：
   ```bash
   ssh lizhichun_24@49.233.89.203
   df -h /home/lizhichun_24/sda1
   ```
   确保至少有 **70GB** 可用空间（32GB zip + 35-40GB 解压后）

2. **检查本地网络稳定性**：
   - 确保网络连接稳定
   - 建议使用有线网络而非 WiFi
   - 上传过程中不要关闭终端或断开连接

### 方法 A：使用 scp 命令（推荐，支持断点续传）

#### Windows（PowerShell 或 CMD）

```powershell
# 方法 1：普通上传（如果中断需要重新开始）
scp "C:\Users\你的用户名\Downloads\cmu-mosei.zip" lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 方法 2：使用 rsync（推荐，支持断点续传，需要先安装）
# 如果安装了 Git Bash 或 WSL，可以使用：
rsync -avz --progress "C:\Users\你的用户名\Downloads\cmu-mosei.zip" lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/
```

#### Mac/Linux（本地终端）

```bash
# 方法 1：使用 scp（如果中断需要重新开始）
scp ~/Downloads/cmu-mosei.zip lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/

# 方法 2：使用 rsync（强烈推荐，支持断点续传）
rsync -avz --progress ~/Downloads/cmu-mosei.zip lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/
```

**rsync 的优势**：
- ✅ 支持断点续传（如果上传中断，重新运行命令会从中断处继续）
- ✅ 显示上传进度
- ✅ 更稳定，适合大文件传输

**上传时间估算**：
- 100 Mbps 上传：约 40-50 分钟
- 50 Mbps 上传：约 1.5-2 小时
- 10 Mbps 上传：约 6-8 小时

**如果上传中断**：
- 使用 `rsync`：直接重新运行命令，会自动续传
- 使用 `scp`：需要重新开始（建议改用 `rsync`）

### 方法 B：使用 SFTP 客户端（如 FileZilla、WinSCP）

**推荐用于大文件上传**，因为：
- ✅ 图形界面，操作直观
- ✅ 支持断点续传
- ✅ 可以暂停/恢复上传
- ✅ 显示详细进度

#### 使用 FileZilla（推荐）

1. **下载并安装 FileZilla**：https://filezilla-project.org/

2. **连接到服务器**：
   - **主机**：`sftp://49.233.89.203` 或 `49.233.89.203`
   - **用户名**：`lizhichun_24`
   - **密码**：你的服务器密码
   - **端口**：22

3. **导航到目标目录**：
   - 左侧：本地文件（找到 `cmu-mosei.zip`）
   - 右侧：服务器目录 `/home/lizhichun_24/sda1/code/multimodal/project/downloads/`

4. **开始上传**：
   - 右键点击 `cmu-mosei.zip` → 选择 "Upload"
   - 或直接拖拽文件到右侧窗口

5. **监控上传进度**：
   - 底部窗口会显示上传速度和剩余时间
   - 如果网络中断，FileZilla 会自动重试

6. **如果上传中断**：
   - FileZilla 会自动尝试续传
   - 如果失败，右键点击文件 → "Resume" 继续上传

### 方法 C：使用 VSCode Remote（不推荐用于 31GB 文件）

⚠️ **注意**：VSCode 的文件上传对大文件支持不佳，可能不稳定，建议使用方法 A 或 B。

### 方法 B：使用 SFTP 客户端（如 FileZilla、WinSCP）

1. 打开 SFTP 客户端
2. 连接到服务器：
   - **主机**：`49.233.89.203`
   - **用户名**：`lizhichun_24`
   - **端口**：22（或你的 SSH 端口）
3. 导航到目标目录：`/home/lizhichun_24/sda1/code/multimodal/project/downloads/`
4. 将本地的 `cmu-mosei.zip` 拖拽上传到该目录

### 方法 C：使用 VSCode Remote（如果你用 VSCode）

1. 在 VSCode 中连接到远程服务器
2. 在文件浏览器中，右键点击 `project/downloads/` 目录
3. 选择 "Upload..." 或直接拖拽文件

---

## 步骤三：在服务器上解压（31GB 大文件）

### 3.1 登录服务器

```bash
ssh lizhichun_24@49.233.89.203
```

### 3.2 检查上传的文件

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads
ls -lh | grep -i cmu-mosei
```

应该能看到 `cmu-mosei.zip` 文件，大小应该接近 31GB。

**验证文件完整性**（可选，但推荐）：
```bash
# 检查文件大小
du -h cmu-mosei.zip

# 如果 zip 文件支持，可以测试完整性（可能需要较长时间）
unzip -t cmu-mosei.zip | head -20
```

### 3.3 检查磁盘空间

```bash
# 检查目标磁盘的可用空间
df -h /home/lizhichun_24/sda1

# 确保至少有 40GB 可用空间用于解压
```

**如果空间不足**：
- 清理不需要的文件
- 或使用其他有足够空间的目录

### 3.4 解压数据集（推荐使用后台任务）

#### 方法 1：后台解压（推荐，不会因 SSH 断开而中断）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 使用 nohup 在后台解压，输出保存到日志文件
nohup unzip -q cmu-mosei.zip -d CMU_MOSEI_raw > unzip.log 2>&1 &

# 记录进程 ID（用于后续检查）
echo $! > unzip.pid

# 查看解压进度（实时）
tail -f unzip.log
```

**解压过程中可以**：
- 按 `Ctrl+C` 退出 `tail`（不会影响解压）
- 随时重新连接服务器检查进度：
  ```bash
  tail -f /home/lizhichun_24/sda1/code/multimodal/project/downloads/unzip.log
  ```

**检查解压是否完成**：
```bash
# 方法 1：检查进程是否还在运行
ps aux | grep unzip

# 方法 2：检查日志最后几行
tail -20 unzip.log

# 方法 3：检查目标目录是否在增长
du -sh CMU_MOSEI_raw
```

#### 方法 2：前台解压（可以看到实时进度）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads

# 使用 screen 或 tmux 保持会话（推荐）
# 如果没安装，先安装：
# sudo apt-get install screen  # Ubuntu/Debian
# 或
# sudo yum install screen      # CentOS/RHEL

# 启动 screen 会话
screen -S unzip_mosei

# 在 screen 中解压
unzip cmu-mosei.zip -d CMU_MOSEI_raw

# 如果需要暂时离开（按 Ctrl+A 然后按 D）
# 重新连接：screen -r unzip_mosei
```

**解压时间估算**：
- 普通 HDD：约 30-60 分钟
- SSD：约 10-20 分钟
- 网络存储：取决于网络速度

### 3.5 验证解压结果

```bash
# 检查解压后的目录大小
du -sh CMU_MOSEI_raw

# 应该接近 35-40GB

# 查看目录结构
ls -lh CMU_MOSEI_raw | head -20

# 统计文件数量
find CMU_MOSEI_raw -type f | wc -l
```

### 3.6 如果解压失败

**常见问题**：

1. **磁盘空间不足**：
   ```bash
   # 清理空间或使用其他目录
   df -h
   ```

2. **文件损坏**：
   ```bash
   # 重新上传 zip 文件
   # 或使用修复选项
   unzip -FF cmu-mosei.zip -d CMU_MOSEI_raw
   ```

3. **权限问题**：
   ```bash
   # 检查并修复权限
   chmod 755 /home/lizhichun_24/sda1/code/multimodal/project/downloads
   ```

### 3.4 查看解压后的目录结构

```bash
ls -R /home/lizhichun_24/sda1/code/multimodal/project/downloads/CMU_MOSEI_raw | head -50
```

**请把这一步的输出发给我**，我可以根据实际的目录结构优化整理脚本。

---

## 步骤四：运行整理脚本

### 4.1 激活 conda 环境

```bash
conda activate myenv310
```

### 4.2 运行整理脚本

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/scripts
python organize_cmu_mosei_raw.py
```

脚本会自动：
- 扫描 `downloads/CMU_MOSEI_raw` 目录
- 识别视频、音频、文本文件
- 按照项目格式整理到 `data/` 目录
- 生成 train/val/test 划分
- 创建标签文件

### 4.3 如果脚本报错

如果脚本报错（比如找不到文件、目录结构不匹配等），请：

1. **把错误信息完整复制给我**
2. **运行以下命令，把目录结构发给我**：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/downloads/CMU_MOSEI_raw
find . -type f -name "*.mp4" | head -10
find . -type f -name "*.csv" | head -10
find . -type f -name "*.json" | head -10
ls -R . | head -100
```

我会根据实际情况调整脚本。

---

## 步骤五：验证数据整合结果

### 5.1 检查整理后的目录结构

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project/data

# 查看各 split 的文件数量
for split in train val test; do
    echo "=== $split ==="
    echo "视频: $(ls $split/video 2>/dev/null | wc -l)"
    echo "音频: $(ls $split/audio 2>/dev/null | wc -l)"
    echo "文本: $(ls $split/text 2>/dev/null | wc -l)"
    echo "标签: $(ls $split/labels 2>/dev/null | wc -l)"
done
```

### 5.2 查看样本示例

```bash
# 查看一个训练样本
ls data/train/video/ | head -1
ls data/train/text/ | head -1
ls data/train/labels/ | head -1

# 查看标签文件内容
cat data/train/labels/$(ls data/train/labels/ | head -1)
```

应该看到类似：

```
neutral
0.0,0.0
```

---

## 步骤六：配置训练脚本

### 6.1 检查配置文件

找到你的训练配置文件（通常是 `config.yaml` 或类似文件），确保数据路径正确：

```yaml
data:
  root_dir: /home/lizhichun_24/sda1/code/multimodal/project/data

datasets:
  mosei:
    enabled: true
    emotion_classes: 7
    emotion_map:
      happy: 0
      sad: 1
      angry: 2
      fear: 3
      neutral: 4
      surprise: 5
      disgust: 6
```

### 6.2 验证数据集配置

确保 `MultimodalDataset` 能正确识别 MOSEI 数据集：

```python
# 在 Python 中测试
from data.dataset import MultimodalDataset

data_dir = "/home/lizhichun_24/sda1/code/multimodal/project/data"
dataset = MultimodalDataset(data_dir, split='train', config=your_config)

print(f"数据集大小: {len(dataset)}")
sample = dataset[0]
print(f"样本ID: {sample['sample_id']}")
print(f"数据集ID: {sample['dataset_id']}")  # MOSEI 应该是 2
```

---

## 步骤七：开始训练和微调

### 7.1 预训练（使用所有数据集）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310

python scripts/train.py \
    --config configs/your_config.yaml \
    --mode pretrain
```

### 7.2 微调（仅使用 MOSEI）

```bash
python scripts/train.py \
    --config configs/your_config.yaml \
    --mode finetune \
    --dataset mosei
```

### 7.3 监控训练过程

训练过程中，你可以：

1. **查看日志输出**：训练脚本会打印 loss、准确率等指标
2. **检查检查点**：模型会定期保存到 `checkpoints/` 目录
3. **使用 TensorBoard**（如果配置了）：

```bash
tensorboard --logdir logs/
```

---

## 🔧 常见问题排查（31GB 大文件特别注意事项）

### Q1: 解压失败，提示磁盘空间不足

**解决方案**：
```bash
# 检查磁盘空间
df -h

# 清理不需要的文件
# 删除旧的下载文件
rm -f /home/lizhichun_24/sda1/code/multimodal/project/downloads/*.zip

# 或使用其他有足够空间的目录
# 例如：解压到 /tmp 然后移动（如果 /tmp 有足够空间）
```

### Q1.1: 上传中断，如何续传？

**使用 rsync（推荐）**：
```bash
# 重新运行 rsync 命令，会自动从中断处继续
rsync -avz --progress ~/Downloads/cmu-mosei.zip lizhichun_24@49.233.89.203:/home/lizhichun_24/sda1/code/multimodal/project/downloads/
```

**使用 FileZilla**：
- 右键点击文件 → "Resume" 继续上传

### Q1.2: 解压时间太长，如何监控进度？

**解决方案**：
```bash
# 方法 1：查看解压日志
tail -f /home/lizhichun_24/sda1/code/multimodal/project/downloads/unzip.log

# 方法 2：监控目录大小变化
watch -n 5 'du -sh /home/lizhichun_24/sda1/code/multimodal/project/downloads/CMU_MOSEI_raw'

# 方法 3：查看进程状态
ps aux | grep unzip
```

### Q2: 整理脚本找不到文件

**解决方案**：
1. 确认解压目录名称正确：`CMU_MOSEI_raw`
2. 检查目录权限：`ls -la downloads/CMU_MOSEI_raw`
3. 查看实际目录结构，根据实际情况调整脚本

### Q3: 训练时提示找不到数据

**解决方案**：
1. 检查 `config.yaml` 中的 `data.root_dir` 路径是否正确
2. 确认数据目录下有 `train/val/test` 子目录
3. 检查文件命名格式是否符合 `mosei_{split}_{idx}.xxx`

### Q4: 标签文件格式不对

**解决方案**：
标签文件应该是：
- 第一行：情绪类别（happy/sad/angry/fear/neutral/surprise/disgust）
- 第二行：效价,唤醒度（例如：0.8,0.7）

如果格式不对，可以批量修复：

```bash
# 示例：批量修复标签文件
find data/train/labels -name "*.txt" -exec sed -i '1s/.*/neutral/' {} \;
```

---

## 📝 后续优化建议

1. **完善标签映射**：根据 MOSEI 的实际标签格式，更新 `organize_cmu_mosei_raw.py` 中的标签解析逻辑
2. **添加数据增强**：在训练配置中启用数据增强，提高模型泛化能力
3. **混合数据集训练**：将 MOSEI 与 CREMA-D、MELD 一起训练，提高模型鲁棒性
4. **域适应**：如果使用混合数据集，启用域适应模块，减少域偏移影响

---

## ✅ 完成检查清单

- [ ] 本地下载 `cmu-mosei.zip` 完成
- [ ] 上传到服务器完成
- [ ] 解压到 `CMU_MOSEI_raw` 完成
- [ ] 运行整理脚本成功
- [ ] 验证数据目录结构正确
- [ ] 配置文件路径正确
- [ ] 训练脚本能正常加载数据
- [ ] 开始训练/微调

---

## 📞 需要帮助？

如果在任何步骤遇到问题，请：

1. **复制完整的错误信息**
2. **提供相关目录结构**（`ls -R` 或 `tree` 输出）
3. **说明你当前执行到哪一步**

我会根据具体情况帮你解决问题。
