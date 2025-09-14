# ContextFusion: Multimodal Image Fusion with Contextual Guidance and Shape Preserving Decoding

[Muchen Xu](https://github.com/stream2005), [Haowen Guo](https://github.com/hww9), [Shimin Shu](), [Botao Shen](https://github.com/shenbt), [Yuexin Song](https://github.com/songyuexin666-wq), Jun Yang∗

School of Artificial Intelligence, China University of Mining and Technology-Beijing, No. 11 Ding, Xueyuan Road, Beijing, 100083, China

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

## Abstract

Infrared and visible image fusion plays a significant role in night vision and visual enhancement applications. However, existing methods suffer from two major problems: the local feature extraction process often lacks semantic contextual awareness, and insufficient explicit modeling of thermal target shapes. To address these issues, we propose the ContextFusion framework, which comprises three core modules: the Multi-Scale Sensitive Local Feature Extraction Module (MSSLFEM), Hybrid Attention-Convolution Fusion Module (HACFM), and Multi-Scale Fusion Decoder (MSF Decoder). To fully leverage broad contextual information for local feature extraction while suppressing irrelevant noise and preserving richer detail features, we introduce the MSSLFEM, which innovatively combines Large-Small Convolution and adopts a parallel design of dynamic and static feature extraction branches. For joint modeling of local dependencies and global correlations to enhance feature representation, we design the HACFM. To better model thermal target morphology, establish effective long-range dependencies, and retain critical scene details, we incorporate deformable sliding window attention mechanisms and integrate them with HACFM to construct the MSF decoder. ContextFusion has been evaluated on multiple benchmark datasets, demonstrating superior performance in both visual quality and quantitative metrics. Furthermore, the model exhibits strong generalization capabilities, a relatively lightweight architecture, and high computational efficiency.

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/stream2005/ContextFusion.git
cd ContextFusion

# Create virtual environment
conda create -n contextfusion python=3.8
conda activate contextfusion

# Install dependencies
pip install -r requirements.txt
```

### Additional Dependencies

For optimal performance, ensure the following are properly installed:

```bash
# For Triton acceleration (optional)
pip install triton

# For deformable attention mechanisms
pip install natten
```

## Dataset Preparation

### MSRS Dataset

1. [MSRS dataset](https://github.com/Linfeng-Tang/MSRS)
2. [RoadScene dataset](https://github.com/hanna-xu/RoadScene)
3. [TNO dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
2. Organize the dataset structure as follows:

```
dataprocessing/
├── Data/
│   ├── MSRS_train_128_200.h5    # Training data in HDF5 format
│   ├── test/
│   │   ├── ir/                  # Infrared test images
│   │   └── vi/                  # Visible test images
```

### Data Processing

Convert your dataset to HDF5 format using the provided preprocessing script:

```bash
cd dataprocessing
python MSRS_train.py
```

This script:
- Processes images into 128×128 patches
- Converts RGB to grayscale for visible images
- Saves data in efficient HDF5 format for training

## Usage

### Training

To train the ContextFusion model:

```bash
python train.py
```

**Training Configuration:**
- Epochs: 120
- Batch size: 8
- Learning rate: 1e-4 with MultiStepLR scheduler
- Loss weights: [1, 1, 10, 100] for IR, VI, SSIM, and gradient losses
- Optimizer: Adam with weight decay

### Testing

For inference on new image pairs:

```bash
python test.py
```


### Evaluation Metrics
To evaluate the fusion results, see: [MMIF-CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)


## Network Specifications

```
Input: IR (1×H×W) + VI (1×H×W)
├── Dual-Branch Encoders
│   ├── Level 1: 1 → 8 channels
│   ├── Level 2: 8 → 16 channels  
│   ├── Level 3: 16 → 32 channels
│   └── Level 4: 32 → 32 channels
├── Multi-Scale Fusion Modules
│   
└── Shape-Preserving Decoder
    ├── Progressive Upsampling (PixelShuffle)
    ├── Skip Connections
    └── Final Reconstruction (Sigmoid)
Output: Fused Image (1×H×W)
```

## Project Structure

```
ContextFusion/
├── nets/
│   └── ContextFusion.py           # Main model architecture
├── losses/
│   └── __init__.py               # Loss function implementations
├── dataprocessing/
│   ├── MSRS_train.py            # Dataset preprocessing
│   └── Data/                    # Training and test data
├── DSwinIR/                     # Deformable attention modules
├── model/                       # Saved model checkpoints
├── test_result/                 # Output fusion results
├── runs/                        # TensorBoard logs
├── train.py                     # Training script
├── test.py                      # Testing/inference script
├── utils.py                     # Utility functions
├── lsnet.py                     # Feature extraction blocks
├── ska.py                       # Selective kernel attention
└── requirements.txt             # Dependencies
```


## Acknowledgments

- Thanks to the [MSRS dataset](https://github.com/Linfeng-Tang/MSRS), [RoadScene dataset](https://github.com/hanna-xu/RoadScene) and [TNO dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) contributors for providing high-quality fusion benchmarks.
- Thanks to the open source implementations that inspired this work: [MMIF-EMMA](https://github.com/Zhaozixiang1228/MMIF-EMMA), [MMIF-CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse), [DSwinIR](https://github.com/Aitical/DSwinIR), [lsnet](https://github.com/THU-MIG/lsnet)
- Special recognition to the PyTorch and computer vision research communities

