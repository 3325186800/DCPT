# DCPT: Robust Physical-Layer Hardware Fingerprint Identification for C-V2X Communications

**Robust Physical-Layer Identification for C-V2X Communications via Differential Projection of Hardware Fingerprints**

Core modules: DCPT feature construction, ViT training, and cross-domain testing.

This repository implements a radio frequency fingerprint identification (RFFI) pipeline for LTE-V2X/C-V2X sidelink communications. The method uses the Demodulation Reference Signal (DMRS) in Physical Sidelink Broadcast Channel (PSBCH) subframes as a stable observation source. After synchronization, carrier frequency offset (CFO) correction, and channel equalization, it constructs Differential Constellation Projection Trace (DCPT) features through Statistical Projection Fusion (SPF) and uses a Vision Transformer (ViT) for device classification and cross-domain testing.

This README is organized around two main code entry points: DCPT feature construction and ViT model training/testing. DCPT, as labeled in the code documentation, generates the projection-fusion features used by the ViT classifier.

## Research Objectives and Main Contributions

Multipath fading, Doppler effects, and synchronization residuals in C-V2X environments can obscure weak but stable transmitter hardware imperfections. This work improves cross-scenario identification robustness through the following designs:

1. It uses PSBCH-DMRS and suppresses macroscopic link effects through time synchronization, CFO removal, and LMMSE-assisted equalization.

2. It constructs DCPT features at multiple differential intervals, denoted by $K$, to retain device-dependent variations at different temporal scales.

3. It computes projection histograms along the I axis, the 45-degree diagonal, and the Q axis to generate compact DCPT features.

4. It uses mutual information (MI) to select informative differential intervals and feeds the fused DCPT features into a ViT classifier.

## Dataset and Input Format

### Data Sources and Experimental Data

The experiments reported in the paper were conducted using ten ESP32 development boards. An ESP32 was also used at the receiver. Data were collected at a carrier frequency of 5.915 GHz, a sampling bandwidth of 20 MHz, and a sampling rate of 30.72 MSamples/s.

The dataset contains six independent subsets covering environments ranging from an ideal direct link to complex indoor and outdoor conditions. Each subset contains 5,000 radio frequency samples: 500 samples from each of the ten devices.

| Subset | Scenario | Channel conditions | Motion state | Collection date |
| :---: | :---: | :---: | :---: | :---: |
| **D1** | Direct link | - | Stationary | 2025-10-02 |
| **I1** | Indoor | LOS | Stationary | 2025-10-04 |
| **I2** | Indoor | LOS | Stationary | 2025-11-12 |
| **I3** | Indoor | NLOS | Stationary | 2025-12-24 |
| **O1** | Outdoor | Mixed | 30-50 km/h | 2025-12-05 |
| **O2** | Outdoor | Mixed | 30-50 km/h | 2025-12-18 |

> **Usage notes**
>
> - **D1** is the ideal baseline dataset and is recommended as the source domain for initial model training in cross-domain experiments.
> - **I1-I3** and **O1-O2** include line-of-sight (LOS) and non-line-of-sight (NLOS) variations, temporal shifts, and Doppler shifts caused by mobility. They are recommended as target domains for evaluating cross-domain generalization.

### Raw MATLAB Input

The DCPT/SPF script scans all `*.mat` files in `dataDir`. Each file must contain:

```text
expendDMrs    % A two-dimensional array: rows are samples and columns are complex DMRS time-domain sequences
```

`expendDMrs` should contain complex-valued sequences obtained after PSBCH/DMRS extraction and front-end preprocessing. The DCPT script also depends on the following functions:

```text
F_Data_IQ_Offset_Process
F_Differential_Process
```

These functions are called in the provided code documentation, but their implementations are not included. Before running the script, place the corresponding implementations on the MATLAB path and verify that they perform I/Q offset processing and differential processing, respectively. If either function is missing, the script issues a warning and stops processing the current file.

### DCPT Features and Training Input

The DCPT/SPF script creates one output `.mat` file for each raw input file. Each output file includes:

```text
Device_Matrix_Data    % [number of samples, number of differential intervals, feature width]
K_Values              % Differential intervals used for feature construction
```

With the current parameters, `K_Values` contains 30 intervals, `Projection_Bins = 128`, and three projection directions are used. The feature width for each interval is therefore $3 \times 128 = 384$, and a typical input shape is `[N, 30, 384]`.

The Python training code uses the filename prefix as the class label. For a file named `device_class_arbitrary_description.mat`, the portion before the first underscore is treated as the class name. For example, `device01_D1.mat` is labeled as `device01`. Each input `.mat` file should contain an array whose name includes `Device` or `Data`; the script gives priority to variable names containing `Device`.

## Code Structure

The two main programs described in the code documentation should be saved as the following files. If the repository uses different filenames, update the commands below accordingly.

| File | Language | Purpose | Main input/output |
| --- | --- | --- | --- |
| `code/dcpt.m` | MATLAB | Core module 1 (DCPT): applies SPF to differential results for multiple values of $K$ and generates projection-fusion training tensors | `expendDMrs` -> `Device_Data`, `K_Values` |
| `code/vit.py` | Python | Core module 2 (ViT): loads DCPT `.mat` files, trains or loads the ViT model, and performs smoothed inference in the target domain | `.mat` -> `best_asymmetric_model.pth`, console metrics |

## Environment and Dependencies

### MATLAB

- MATLAB with support for `histcounts`, `histcounts2`, and `imresize`.

- Image Processing Toolbox, required for `imresize`.

- Project-specific DMRS/IQ preprocessing functions: `F_Data_IQ_Offset_Process.m` and `F_Differential_Process.m`.

### Python

- Python 3.

- PyTorch and Torchvision.

- NumPy.

- SciPy.

Create an isolated environment and install the Python dependencies as follows:

```text
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy scipy
```

When using an NVIDIA GPU, select a PyTorch installation compatible with the local CUDA driver by following the official PyTorch installation instructions. The training script automatically selects `cuda` when available and otherwise falls back to `cpu`. During the first training run, Torchvision downloads the pretrained ViT-B/16 weights. For an offline environment, cache the weights in advance or replace `weights=models.ViT_B_16_Weights.DEFAULT` with `weights=None` in the code.

#### Tested Runtime Environment (`ai`)

The following software versions and hardware configuration were tested in the `ai` Conda environment.

| Item | Tested version/configuration |
| --- | --- |
| Conda environment | `ai` (`C:\Users\33251\.conda\envs\ai`) |
| Python | 3.10.19 (64-bit) |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Operating system | Windows 10 Home China 25H2 (build 26200, 64-bit) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM) |
| NVIDIA driver | 566.24 |
| CUDA | PyTorch built with CUDA 12.1; NVIDIA driver reports CUDA 12.7 |
| cuDNN | 9.1.0 (version number 90100) |

## Quick Start

### 1. Configure Paths and Parameters

Set the data and output directories in the DCPT script. Relative paths are recommended so that local absolute paths from the code documentation are not retained:

```text
dataDir = fullfile('data', 'raw_mat', 'D1');
saveDir = fullfile('data', 'dcpt_mat');
```

Set the training domains and target test domain in the training script:

```text
TRAIN_ROOT_1 = r"data/dcpt_mat/source_a"
TRAIN_ROOT_2 = r"data/dcpt_mat/source_b"
TEST_ROOT    = r"data/dcpt_mat/target"
```

Verify that all three directories contain correctly named `.mat` feature files and that the training and target domains do not contain overlapping sampled frames.

### 2. Construct DCPT/SPF Features (Core Module 1)

Run the following command in MATLAB:

```text
run('code/dcpt.m')
```

By default, the script processes 30 candidate differential intervals, computes 128 projection bins along the I, 45-degree, and Q directions, and saves `Device_Matrix_Data` and `K_Values`. Before execution, verify that `K_Values` contains the intervals selected by MI. If MI is recalculated, replace this variable with the new list of optimal intervals.

### 3. Train the ViT and Perform Cross-Domain Testing (Core Module 2)

```text
python code/vit.py
```

The default configuration uses a batch size of 50, a learning rate of `1e-5`, 20 training epochs, the AdamW optimizer, a StepLR scheduler that multiplies the learning rate by 0.5 every 10 epochs, weighted cross-entropy, and `label_smoothing=0.1`. The best model weights are written to `best_asymmetric_model.pth` in the current working directory.

If this file already exists, the script loads the weights and skips training by default. To force retraining, change the entry point to:

```python
run_advanced_cross_domain(force_train=True)
```

During inference, the script applies a moving average of softmax probabilities over a window of length 5 and reports the overall target-domain accuracy, per-device identification statistics, and average inference time.

### 4. Complete One-Click Command

Save the following content as `run_all.bat` in the project root and double-click it to execute the complete workflow. The script first constructs the DCPT features and then runs the ViT program in the `ai` environment.

```bat
@echo off
setlocal
cd /d "%~dp0"
"D:\bin\matlab.exe" -batch "run('code/dcpt.m')"
if errorlevel 1 exit /b %errorlevel%
"D:\1\Scripts\conda.exe" run -p C:\Users\33251\.conda\envs\ai python python\vit.py
endlocal
```

## Complete Reproduction Workflow

```text
Raw PSBCH samples
        |
        v
DMRS extraction -> Time synchronization / CFO removal / LMMSE equalization
        |                               (preprocessing functions must be provided separately)
        v
expendDMrs (.mat)
        |
        v
dcpt.m -> Device_Data (.mat)
        |
        v
vit.py -> ViT weights and cross-domain identification results
```

It is recommended to construct DCPT features from a single `.mat` file first. Confirm that the dimensions and complex-valued data type of `expendDMrs`, as well as the output of the differential function, are correct before constructing features in batch and starting model training.

## Methodology

### 1. SPF and DCPT: Compact Projection Fusion

The DCPT script projects complex-valued differential samples onto three directions: 0 (I), $\pi/4$ (diagonal), and $\pi/2$ (Q). Each projection is represented by a normalized histogram, and the three histograms are concatenated to form a 384-dimensional feature for each value of $K$.

After candidate values of $K$ are ranked using mutual information, the features associated with the most discriminative intervals are concatenated to form the DCPT representation. The configuration reported in the paper uses the top 30 intervals. The same fixed `K_Values` must be used during training and testing.

### 2. ViT Classifier

The training script replicates the single-channel DCPT matrix across three channels, resizes it to $224 \times 224$, and feeds it into a ViT-B/16 model. The classification head is:

```text
Linear(768, 512) -> ReLU -> Dropout(0.6) -> Linear(512, num_classes)
```

## Experimental Settings and Reported Results

The experiments reported in the paper use ten devices, with D1 serving as the cross-domain training source and I1-I3 and O1-O2 serving as target test domains. To avoid leakage between consecutive frames, the paper recommends splitting each device's samples chronologically into the first 70% for training, the next 10% for validation, and the final 20% for testing. The paper reports approximately 99% intra-domain accuracy and approximately 85%-96.8% cross-domain accuracy, depending on the scenario. These values are results reported in the paper and do not guarantee performance on arbitrary datasets or computing environments.

> **Important difference between the code and the protocol described in the paper:** The provided Python training script currently uses `random_split(..., generator=torch.Generator().manual_seed(42))` to divide the source-domain data randomly into 80% training and 20% validation subsets. This is not equivalent to the chronological 70%/10%/20% split described in the paper. To reproduce the experimental conclusions strictly, generate non-overlapping train/validation/test file lists according to acquisition time or frame index, and update the data-loading logic that currently uses `random_split`. Do not randomly shuffle consecutive sampled frames and then claim that the resulting evaluation is free from data leakage.

For meaningful comparisons across runs, record at least the following information: data domains and number of devices, number of frames per class, `K_Values`, data-splitting strategy, random seed, Python/MATLAB/PyTorch versions, GPU model, training time, per-domain accuracy, and confusion matrices.

## Reproducibility Notes and Precautions

- **Label consistency:** The target-domain class set must match the source-domain class set. The script fixes the target-domain label mapping through `forced_classes=standard_classes`; files belonging to classes absent from the source domain are skipped.

- **Data separation:** Adjacent frames from the same continuous acquisition session are highly correlated. Split the data by temporal block or acquisition session to avoid training/test leakage.

- **Fixed `K_Values`:** MI-based selection must be performed using training data only. Using target-domain samples to select $K$ causes evaluation leakage.

- **Portable paths:** Do not commit local absolute paths, raw device identifiers, sensitive acquisition locations, or private keys. Paths should be supplied through a configuration file or command-line arguments.

- **Weight reuse:** If `best_asymmetric_model.pth` already exists, the script skips training. After changing the number of classes, the ViT architecture, or `K_Values`, delete or rename the previous weight file and force retraining.

## Citation

If you use this code or method, please cite the associated paper:

> *Robust Physical-Layer Identification for C-V2X Communications via Differential Projection of Hardware Fingerprints*.

## Contributions and Acknowledgments

### Contributions and Issue Reports

Contributions related to reproducibility fixes, path parameterization, data-loading validation, result logging, and documentation improvements are welcome. Before contributing:

1. Do not submit raw C-V2X acquisition data, personal information, or unauthorized device information.

2. State the MATLAB/Python versions, data split, and reproduction commands.

3. Provide a minimal reproducible example or test results for functional changes.

4. Do not make performance claims beyond the experimental settings reported in the paper.

### Acknowledgments

This implementation uses MATLAB, PyTorch, Torchvision, NumPy, and SciPy.
