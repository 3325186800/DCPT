# DCPT：面向C-V2X通信的鲁棒物理层硬件指纹识别

Robust Physical-Layer Identification for C-V2X Communications via Differential Projection of Hardware Fingerprints

核心模块：DCPT特征构建、ViT训练与跨域测试

本仓库实现面向LTE-V2X/C-V2X侧链通信的射频指纹识别（RFFI）流程。方法以PSBCH子帧中的DMRS为稳定观测对象，经同步、载波频偏（CFO）校正和信道均衡后，通过统计投影融合（SPF）构建差分星座投影特征（DCPT），并使用Vision Transformer（ViT）进行设备类别识别与跨域测试。

本README以两个核心代码入口为主线：DCPT特征构建代码和ViT模型训练/测试代码。DCPT（代码说明书中的标注）负责生成供ViT使用的投影融合特征。

## 研究目标与主要贡献

C-V2X场景中的多径衰落、多普勒效应及同步残差，容易掩盖发射端微弱而稳定的硬件非理想特征。该工作通过以下设计提升跨场景识别的鲁棒性：

1. 使用PSBCH-DMRS，并通过时间同步、CFO去除及LMMSE辅助均衡抑制宏观链路影响；

1. 以多个差分间隔K构建DCPT特征，保留不同时间尺度下的设备相关变化；

1. 对I、45°对角线和Q三个方向做投影直方图统计，生成紧凑的DCPT特征；

1. 以互信息（MI）筛选信息量高的差分间隔，并将融合后的DCPT特征送入ViT分类器。

## 数据集与输入格式

### 数据来源与实验数据

论文实验采集自10台ESP32开发板。测试场景包括直连静态场景D1、室内LOS/NLOS场景I1-I3，以及车速30-50 km/h的室外移动场景O1-O2。接收端也使用ESP32，在5.915 GHz、20 MHz采样带宽和30.72 MSamples/s条件下采集。

### 原始MATLAB输入

DCPT/SPF脚本会扫描dataDir中的*.mat文件。每个文件必须包含：

```text
expendDMrs    % 行为样本、列为复数 DMRS 时域序列的二维数组
```

```text
expendDMrs应为完成PSBCH/DMRS提取及前端预处理后的复数序列。DCPT脚本还依赖下列函数：
```

```text
F_Data_IQ_Offset_Process
F_Differential_Process
```

这两个函数在所提供的代码说明书中被调用但未给出函数体。运行前请将其对应实现置于MATLAB路径中，并确认它们分别完成I/Q偏移处理和差分处理；缺失时脚本会发出警告并停止当前文件的处理。

### DCPT特征与训练输入

DCPT/SPF脚本会为每个原始文件输出一个.mat文件，其中包括：

```text
Device_Matrix_Data    % [样本数, 差分间隔数, 特征宽度]
K_Values              % 参与构建的差分间隔
```

在当前参数下，K_Values含30个间隔、Projection_Bins = 128、三个投影方向，因此单个间隔的特征宽度为3 × 128 = 384，典型输入形状为[N, 30, 384]。

Python训练代码以文件名前缀作为类别标签：设备类别_任意描述.mat中下划线前的部分即类别名。例如，device01_D1.mat会被标注为device01。每个输入.mat中应包含名称含Device或Data的数组；脚本优先选择名称含Device的变量。

## 代码组成

建议将代码说明书中的两段核心程序分别保存为以下文件。若仓库中已有不同文件名，只需相应修改下文命令即可。

| 文件 | 语言 | 作用 | 主要输入/输出 |
| --- | --- | --- | --- |
| matlab/dcpt.m | MATLAB | 核心代码一（DCPT）：对多K的差分结果执行SPF，生成投影融合训练张量 | expendDMrs -> Device_Data、K_Values |
| python/vit.py | Python | 核心代码二（ViT）：加载DCPT .mat文件，训练/加载ViT，并在目标域进行平滑推理 | .mat -> best_asymmetric_model.pth、控制台指标 |

## 环境与依赖

### MATLAB

- MATLAB（需支持histcounts、histcounts2和imresize）；

- Image Processing Toolbox（imresize所需）；

- 本项目的DMRS/IQ预处理函数：F_Data_IQ_Offset_Process.m与F_Differential_Process.m。

### Python

- Python 3；

- PyTorch与Torchvision；

- NumPy；

- SciPy。

- 可创建独立环境并安装Python依赖：

```text
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy scipy
```

如使用NVIDIA GPU，请根据本机CUDA驱动和PyTorch官方安装说明选择匹配的PyTorch安装包。训练脚本会自动选择cuda；不可用时自动回退至cpu。首次训练会请求Torchvision的ViT-B/16预训练权重；无网络环境请预先缓存该权重，或将代码中的weights=models.ViT_B_16_Weights.DEFAULT改为weights=None。

#### 实测运行环境（ai）

以下版本和硬件信息均在截图所示的ai Conda环境中实测。

| 项目 | 实测版本/配置 |
| --- | --- |
| Conda环境 | ai（C:\Users\33251\.conda\envs\ai） |
| Python | 3.10.19（64位） |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| 操作系统 | Windows 10 Home China 25H2（内部版本26200，64位） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8188MiB显存） |
| NVIDIA驱动 | 566.24 |
| CUDA | PyTorch编译CUDA 12.1；NVIDIA驱动报告CUDA 12.7 |
| cuDNN | 9.1.0（版本号90100） |

## 快速开始

### 1. 配置路径和参数

在DCPT脚本中设置数据和输出目录。建议使用相对路径，避免保留代码说明书中的本地绝对路径：

```text
dataDir = fullfile('data', 'raw_mat', 'D1');
saveDir = fullfile('data', 'dcpt_mat');
```

训练脚本中设置训练域与目标测试域：

```text
TRAIN_ROOT_1 = r"data/dcpt_mat/source_a"
TRAIN_ROOT_2 = r"data/dcpt_mat/source_b"
TEST_ROOT    = r"data/dcpt_mat/target"
```

请确认三处目录中均为正确命名的.mat特征文件，且训练集和目标域没有重叠采样帧。

### 2. 构建DCPT/SPF特征（核心代码一）

在MATLAB中运行：

```text
run('matlab/dcpt.m')
```

脚本默认对30个候选差分间隔进行处理，分别在I、45°和Q方向统计128个投影bin，保存Device_Matrix_Data与K_Values。请在运行前核对K_Values是否为你的MI筛选结果；若重新计算了MI，须以新的最优间隔列表替换该变量。

### 3. ViT训练和跨域测试（核心代码二）

```text
python python/vit.py
```

脚本默认配置为批大小50、学习率1e-5、训练20个epoch、AdamW优化器、StepLR（每10个epoch衰减为原来的0.5）、加权交叉熵和label_smoothing=0.1。最佳权重写入当前工作目录的best_asymmetric_model.pth。

若该文件已存在，代码会默认加载权重并跳过训练。若需强制重训，将入口改为：

run_advanced_cross_domain(force_train=True)

推理阶段使用长度为5的softmax概率滑动平均，并输出总体目标域准确率、逐设备识别统计及平均推理耗时。

### 4. 完整一键运行命令

将以下内容保存为项目根目录的run_all.bat后双击即可；脚本先构建DCPT特征，再调用ai环境中的ViT程序。

```bat
@echo off
setlocal
cd /d "%~dp0"
"D:\bin\matlab.exe" -batch "run('matlab/dcpt.m')"
if errorlevel 1 exit /b %errorlevel%
"D:\1\Scripts\conda.exe" run -p C:\Users\33251\.conda\envs\ai python python\vit.py
endlocal
```

## 完整复现实验流程

PSBCH原始采样
        │
        ▼
DMRS提取 -> 时间同步 / CFO去除 / LMMSE均衡
        │                         （预处理函数需另行提供）
        ▼
expendDMrs(.mat)
        │
        ▼
dcpt.m ──> Device_Data(.mat)
        │
        ▼
vit.py ──> ViT权重、跨域识别结果

建议先以一个.mat文件完成DCPT特征构建，确认expendDMrs的维度、复数类型及差分函数输出正确，再批量构建特征并启动训练。

## 方法说明

### 1. SPF与DCPT：紧凑投影融合

DCPT脚本直接将复数差分样本投影到三个方向：0（I）、π/4（对角线）和π/2（Q）。每个方向用归一化直方图表示，三个直方图拼接为单个K的384维特征。

基于互信息对候选K排序后，拼接最有区分力的间隔特征形成DCPT表示。论文报告的设置使用Top-30个间隔；应在训练和测试阶段固定同一组K_Values。

### 2. ViT分类器

训练脚本将单通道DCPT矩阵复制为3通道、缩放至224 × 224，再输入ViT-B/16。分类头为：

```text
Linear(768, 512) -> ReLU -> Dropout(0.6) -> Linear(512, num_classes)
```

## 实验设置与结果说明

论文报告的实验采用10台设备，并以D1为跨域训练源、I1-I3和O1-O2为测试目标。论文中为避免连续帧泄漏，推荐每个设备按时间顺序划分前70% 训练、中间10% 验证、后20% 测试；论文报告的结果为约99% 的域内准确率，以及约85%-96.8% 的跨域准确率（视场景而定）。这些数值是论文结果，不代表任意数据集或运行环境下的保证性能。

```text
重要：代码与论文划分协议的差异。所提供的Python训练脚本当前使用random_split(..., generator=torch.Generator().manual_seed(42))将源域数据按80%/20% 随机划分为训练/验证集；这与论文中的时间顺序70%/10%/20% 划分并不等价。若要严格复现实验结论，请按采集时间或帧序号先行生成不重叠的train/validation/test文件清单，并相应替换random_split的数据加载逻辑。不要随机打散连续采样帧后再声称获得了论文的无泄漏结果。
```

为便于比较不同运行，建议至少记录：数据域与设备数、每类帧数、K_Values、数据划分策略、随机种子、Python/MATLAB/PyTorch版本、GPU型号、训练耗时、每域准确率与混淆矩阵。

## 可复现性与注意事项

- 标签一致性：目标域的类别集合应与源域一致。脚本通过forced_classes=standard_classes固定目标域标签映射，不属于源域类别的文件会被跳过。

- 数据隔离：同一连续采集会话的相邻帧高度相关。请按时间块或会话划分数据，避免训练/测试泄漏。

- K_Values固定： MI筛选只能在训练数据上完成；将目标域样本用于选择K会造成评估泄漏。

- 路径可移植性：请勿提交个人电脑绝对路径、原始设备编号、敏感采样位置或私钥。推荐通过配置文件或命令行参数传入路径。

- 权重可复用：已有best_asymmetric_model.pth时脚本会跳过训练；变更类别数、ViT结构或K_Values后应删除/另存旧权重并强制重训。

## 引用

如使用本代码或方法，请引用相关论文《Robust Physical-Layer Identification for C-V2X Communications via Differential Projection of Hardware Fingerprints》。

## 贡献与致谢

### 贡献与问题反馈

欢迎提交可复现性修正、路径参数化、数据加载校验、结果记录与文档改进建议。贡献前请：

1. 不提交原始C-V2X采样数据、个人信息或未经授权的设备信息；

1. 说明MATLAB/Python版本、数据划分和复现实验命令；

1. 为功能修改提供最小可复现实例或测试结果；

1. 不对论文结果作超出实验设置的性能承诺。

### 致谢

本实现使用MATLAB、PyTorch、Torchvision、NumPy和SciPy。
