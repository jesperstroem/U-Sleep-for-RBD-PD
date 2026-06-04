# Automatic Sleep Staging for RBD and PD Populations

This repository demonstrates how to perform automatic sleep staging with a U-Sleep model fine-tuned for subjects with REM Sleep Behavior Disorder (RBD) and Parkinson’s Disease (PD).

The training and validation of the models are described in our paper, *"Fully Automated Sleep Staging: Multicenter Validation of a Generalizable Deep Neural Network for Parkinson’s Disease and Isolated REM Sleep Behavior Disorder"*, which is currently available as an open-access preprint on arXiv: https://arxiv.org/abs/2602.09793

Questions regarding the paper should be directed to the corresponding author at cas@clin.au.dk\
Questions regarding this repository can be directed to js@ece.au.dk

This repository and its model weights (`Pretrained_Model.ckpt`, `Generalized_Model.ckpt`) are intended for non-commercial research purposes only.

---

## Repository Overview

The repository currently provides two models:

1. **Pretrained Model**  
   Designed for use with the general healthy population.

2. **Generalized Model**  
   Designed for use with both the general healthy population and individuals with RBD and/or PD.

At present, the provided Python notebook (`demo.ipynb`) supports the following MNE-compatible file formats:

- `.set`
- `.vhdr`
- `.edf`

The code has been tested on both:

- A local Windows machine (CPU)
- A high-performance computing cluster (NVIDIA GPU, CUDA 11.8)

---

## Installation Guide

To get started, follow the steps below.

### 1. Install Software Required for Setup

Make sure the following software is installed:

- Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- Git: https://git-scm.com/install/

---

### 2. Clone the Repository

```console
git clone https://github.com/jesperstroem/U-Sleep-for-RBD-PD
cd U-Sleep-for-RBD-PD
```

---

### 3. Create and Activate a Conda Environment

Replace `<your_environment_name>` with a name of your choice.

```console
conda create -n <your_environment_name> python=3.10
conda activate <your_environment_name>
```

---

### 4. Install PyTorch (Version 2.0.1)

The PyTorch installation procedure depends on whether your system has GPU support available. If you are unsure, install the CPU version.

Run **only one** of the following commands.

#### 4A. GPU Version (CUDA 11.8)

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### 4B. CPU Version

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

---

### 5. Install PyTorch Lightning

```console
python -m pip install lightning==2.1.3 torch==2.0.1
```

---

### 6. Install the Repository Package

```console
python -m pip install .
```

---

## Demo

You can now begin performing automatic sleep staging on your MNE-compatible files.

See the provided Python notebook (`demos/demo.ipynb`) for an example demonstrating how to use the supplied model weights.
