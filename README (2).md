

# MAFW

## 详细信息

**MAFW** 是一个用于**自然场景（in the wild）动态人脸表情识别**的大规模、多模态、复合情感数据库。数据库中的片段来自中国、日本、韩国、欧洲、美国和印度，覆盖多种主题（如综艺、家庭、科幻、悬疑、爱情、喜剧、访谈等），包含广泛的人类情绪表达。每个片段由 11 名训练有素的标注员**独立标注 11 次**。MAFW 数据库具有高度多样性、大规模数据量和丰富的标注信息，包括：

- 来自电影、电视剧与短视频的 **10,045** 个视频片段；
- 每个视频片段对应一个 **11 维**表情分布向量；
- **三种标注**：（1）单一表情标签；（2）多重表情标签；（3）双语情感描述文本；
- **两个子集**：单表情子集（包含 **11** 类单一情绪）；多表情子集（包含 **32** 类复合情绪）；
- **三种自动标注**：逐帧 68 个面部关键点、面部区域框、性别；
- **四个基准任务**：单模态单表情分类、多模态单表情分类、单模态复合表情分类、多模态复合表情分类。


## 视频-音频片段示例


### 1. MAFW 单表情示例

<table id="tfhover" class="tftable" border="1">
<tr><td width="30%"><image src="samples-gif/anger_07317_4s.gif" /></td><td width="15%"><b>愤怒（Anger）</b></td><td>English: A girl with tears in her eyes shouts at the person opposite her. The deep frown,a downward pull on the lip corners,the higher inner corners of eyebrows and the lower outer corners of eyebrows.<br />中文：一个女生眼含着泪水大声训斥着对面的人。眉头紧蹙，嘴角下拉，眉毛内高外低。</td></tr>
<tr><td><image src="samples-gif/disgust_07734.gif" /></td><td><b>厌恶（Disgust）</b></td><td>English: A woman looks nervously at her feet. The frown,the closed eyes and  the  open mouth.<br />中文：一个女人紧张的看着脚下的东西。皱眉，眼睛微闭，嘴巴张开。</td></tr>
<tr><td><image src="samples-gif/fear_09246.gif" /></td><td><b>恐惧（Fear）</b></td><td>English: A girl gasps in the dark. The wide eyes and the open mouth.<br />中文：一个女孩在昏暗的环境中急促的喘息。瞪眼，嘴巴张大。</td></tr>
<tr><td><image src="samples-gif/happy_01440.gif" /></td><td><b>快乐（Happiness）</b></td><td>English: A woman communicates with a man, talking about dinner. The slightly closed eyes, the open mouth and the raised lip corners.<br />中文：一个女人与男人交流，谈论着晚餐。眼睛微闭，嘴巴张开，嘴角上扬。</td></tr>
<tr><td><image src="samples-gif/sad_00467.gif" /></td><td><b>悲伤（Sadness）</b></td><td>English: A girl stands on the beach, tilting her head back and crying. The deep frown and the wide open mouth.<br />中文：一个女孩站在海边，仰着头哭泣。眉头紧蹙，嘴巴张大。</td></tr>
<tr><td><image src="samples-gif/surprise_15152.gif" /></td><td><b>惊讶（Surprise）</b></td><td>English: The woman sits with her eyes slowly widening, then suddenly shivers back and asks in a confused voice. The wide eyes.<br />中文：女人坐着眼睛慢慢睁大，然后突然向后颤动了一下，慌乱地询问道。瞪眼。</td></tr>
<tr><td><image src="samples-gif/contempt_08668.gif" /></td><td><b>蔑视（Contempt）</b></td><td>English: A woman gives a dismissive laugh. A curl of the lips.<br />中文：一个女人不屑地笑了一声。撇嘴。</td></tr>
<tr><td><image src="samples-gif/anxiety_07499.gif" /></td><td><b>焦虑（Anxiety）</b></td><td>English: A woman can't get through to the phone and grumbles impatiently. The  frown and the compressed lips.<br />中文：一个女人打不通电话，不耐烦地埋怨了一声。皱眉，抿嘴。</td></tr>
<tr><td><image src="samples-gif/helplessness_08581.gif" /></td><td><b>无助（Helplessness）</b></td><td>English: A maid stands thoughtfully behind her mistress. The wide eyes and the compressed lips.<br />中文：一个侍女若有所思地站在女主人身后。瞪眼，抿嘴。</td></tr>
<tr><td><image src="samples-gif/disappointment_09683.gif" /></td><td><b>失望（Disappointment）</b></td><td>English: A man lowers his head and takes a deep breath. The open mouth and the trembling lips.<br />中文：一个男人低下头，深呼吸。嘴巴半张，嘴唇颤抖。</td></tr>
<tr><td><image src="samples-gif/neutral_00120_3s.gif" /></td><td><b>中性（Neutral）</b></td><td>-</td></tr>
</table>

### 2. MAFW 多表情示例

<table id="tfhover" class="tftable" border="1">
<tr><td width="30%"><image src="samples-gif/anger-disgust_08827.gif" /></td><td width="15%"><b>愤怒<br/>厌恶</b></td><td>English: A man glances his head to the side at the words of the man in front of him. The frown and a downward pull on the lip corners.<br />中文：一个男人听到面前的男人的话语后将头瞥向一边。皱眉，嘴角下拉。</td></tr>
<tr><td><image src="samples-gif/fear_sad_07213.gif" /></td><td><b>恐惧<br/>悲伤</b></td><td>English: A woman squats on the ground and talks to a man tearfully. The marked frown and a downward pull on the lip corners.<br />中文：一个女人蹲坐在地上满含泪水的在和男人说话。皱眉，嘴角下拉。</td></tr>
<tr><td><image src="samples-gif/fear-sad-anxiety_09878.gif" /></td><td><b>恐惧<br/>悲伤<br/>焦虑</b></td><td>English: A woman begs someone. The marked frown, the wide eyes and the slightly open mouth.<br />中文：一个女人哀求着。皱眉，瞪眼，嘴巴微张。</td></tr>
<tr><td><image src="samples-gif/anger-disgust-contempt_15120.gif" /></td><td><b>愤怒<br/>厌恶<br/>蔑视</b></td><td>English: A woman makes a condescending sarcasm at the other person. The raised chin, the raised eyebrows, the closed eyes and the frown.<br />中文：一个女人居高临下地挖苦对方。下巴上扬，挑眉，闭眼，皱眉。</td></tr>
</table>


## 条款与条件（Terms & Conditions）

- MAFW 数据库仅可用于 **非商业研究目的**。
- 你同意**不得**以商业目的复制、复刻、拷贝、出售、交易、转售或利用任何片段内容及其衍生数据。
- 你同意**不得**进一步复制、发布或分发 MAFW 数据库的任何部分。除同一机构内单一站点的内部使用外，允许制作数据集副本。



## 如何获取 MAFW 数据集

该数据库对**隶属于大学的教授与科研人员**公开且免费。若你是学生并希望获取数据集，请注意申请通常需要你所在机构教师的正式背书。

仅在正确完成以下步骤后，才会授予你使用（但不得复制或分发）MAFW 数据库的许可：

1. 下载 [MAFW-academics -final.pdf](/academics/mafw-academics-final.pdf) 文档，该文档作为最终用户许可协议（EULA）。
2. 仔细阅读条款并确认接受。需要在文档末尾填写并签署所需信息——**对学生申请者而言，该签名必须由其所属大学的教授签署**以验证请求。
3. 将填写完整并签署后的文档发送至：1202411179@cug.edu.cn。
4. 审核通过后，你将通过邮件收到下载链接，提供两种下载方式：**百度网盘** 与 **Google Drive**。

<!--
## 内容预览（Content Preview）
下面是数据集内容的预览。

```
MAFW Dataset{
	clips: {
	       .....
	       00151.mp4,
	       00152.mp4,
	       00153.mp4,
	       .....
		}
	caption-label: {
		single_set : single_label.txt
		multi_Set : multi_label.txt		 
		}
	readme.txt
}
```

`single_label.txt` 示例：<br>
03352.mp4  &emsp;&emsp;happiness	 &emsp; &emsp;&emsp;    一位男子手握一杯饮料，向别人介绍着这杯饮料。嘴角上扬。	 &emsp;A man holds a drink in his hand and introduces it to others. The raised lip corners.
<br>
<br>
`multi_label.txt` 示例：<br>
09932.mp4 &emsp;sadness_surprise &emsp;	一个女人听见一个噩耗，手机从耳边滑下。眉头紧蹙，瞪眼。	 &emsp;A woman hears a bad news and her mobile phone slips down her ear. The deep frown and the wide eyes.
<br>
<br>
更多数据集细节请参考论文：["MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild"](/academics/MAFW.pdf)。
-->

## 引用（Citation）

如果你觉得我们的工作对你的研究有帮助，请引用以下论文：

- Yuanyuan Liu, Wei Dai, Chuanxu Feng, Wenbin Wang, Guanghao Yin, Jiabei Zeng, and Shiguang Shan. 2022. MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild. In Proceedings of the 30th ACM International Conference on Multimedia (MM ’22), October 10–14, 2022, Lisboa, Portugal. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3503161.3548190

```
@inbook{liu_mafw_2022,
	author = {Liu, Yuanyuan and Dai, Wei and Feng, Chuanxu and Wang, Wenbin and Yin, Guanghao and Zeng, Jiabei and Shan, Shiguang},
	title = {MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild},
	year = {2022}
	isbn = {978-1-4503-9203-7},
	publisher = {ACM},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3503161.3548190},
	booktitle = {Proceedings of the 30th ACM International Conference on Multimedia (MM’22)},
	numpages = {9}
}
```

## 内容预览（Content Preview）

- 数据（Data）

<image src="samples-gif/data.png" height="130" />

- 标签（Labels）

<image src="samples-gif/labels.png" height="130" />

- 标签（自动标注，Labels auto）

<image src="samples-gif/auto_labels.png" height="130" />

- 训练集与测试集（Train & Test Set）

<image src="samples-gif/train&test.png" height="130" />

更多数据集细节请参考论文：[MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild](/academics/MAFW_final.pdf)。

关于情感描述文本的更多信息，请参考 MAFW 的[补充材料](/academics/MAFW_supp.pdf)。


## 常见问题（FAQ）

### 1. 百度网盘与 Google Drive 下载链接有什么区别？
申请通过后，你会收到两种下载方式：

- **百度网盘**：包含完整数据集，包括帧数据（frames）。
- **Google Drive**：不包含帧数据（仅含视频片段 clips 与标签文件）。

### 2. 如果无法访问百度网盘但又需要帧数据（frames），该怎么办？
如果你需要帧数据但无法访问百度网盘，你可以从 Google Drive 获取的视频片段中自行提取并处理帧图像，步骤如下（与论文中的预处理流程一致）：

1. 使用 OpenCV 等工具从视频片段中**提取帧图像**。
2. **人脸检测与关键点提取**：使用人脸工具（例如论文中引用的 [face-alignment-master](https://github.com/1adrianb/face-alignment)，或其它方便的人脸检测库）定位人脸区域并提取 68 个关键点。
3. **人脸对齐与缩放**：使用仿射变换与旋转（OpenCV 或类似库）进行对齐，然后将对齐的人脸区域缩放到 **224×224**（与数据集标准格式一致）。

你可以使用任何你熟悉的工具完成“检测/关键点/对齐”，关键是确保最终输出为 **224×224 的对齐人脸帧**，以保证与基准设置一致。

### 3. 如何处理分卷压缩文件（例如 clips 和 frames）？
由于 `clips` 与 `frames` 目录体积较大，通常会被拆分为多个分卷压缩文件：

- Clips：`clips.7z.001`, `clips.7z.002`, ...（按序号递增）
- Frames：`frames.7z.001`, `frames.7z.002`, ..., `frames.7z.010`（最多 10 个分卷）

**解压说明：**

- **Windows**：使用 7-Zip 或 WinRAR。右键第一个文件（如 `clips.7z.001` 或 `frames.7z.001`）选择“解压到当前目录”，软件会自动将所有分卷合并并解压为一个文件夹。
- **Linux/macOS**：在终端使用 `7z` 命令，运行 `7z x clips.7z.001`（或 `frames.7z.001`），工具会自动识别并按顺序处理所有相关分卷。

**关键注意事项：**

1. 确保所有分卷文件都在**同一个目录**中（不要分散到子目录）。
2. 不要**重命名**任何分卷文件（例如不要把 `clips.7z.001` 改成 `clips_part1.7z`），否则会破坏分卷序列导致解压失败。
3. 确认所有分卷文件都已完整下载（没有损坏或缺失）——分卷不完整会导致解压失败。
4. 最终解压出来的目录名通常为 `clips` 或 `frames`（无需手动合并文件夹）。

## 代码（Code）

我们提出的 T-ESFL 模型源码可以在此下载：[https://github.com/MAFW-database/MAFW](https://github.com/MAFW-database/MAFW)。

## 联系方式（Contact）

如对 MAFW 有任何问题，请联系我们。

<table id="tfhover" class="tftable" border="1">
<tr><td width="20%">Yuanyuan Liu</td><td width="65%">Professor, China University of Geosciences</td><td width="15%"><a href="mailto:liuyy@cug.edu.cn">liuyy@cug.edu.cn</a></td></tr>
<tr><td>Shuyang Liu</td><td>Master, China University of Geosciences</td><td><a href="mailto:20171003670@cug.edu.cn">20171003670@cug.edu.cn</a></td></tr>
<tr><td>Ying Qian</td><td>Master, China University of Geosciences</td><td><a href="mailto:1202411179@cug.edu.cn">1202411179@cug.edu.cn</a></td></tr>
</table>

更多信息欢迎访问团队主页：[https://cvlab-liuyuanyuan.github.io/](https://cvlab-liuyuanyuan.github.io/)


