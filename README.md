# Automatic Sleep Staging for RBD and PD Populations

This repository will show you how to install and use the U-Sleep model which has been finetuned towards individuals with REM Sleep Behavior Disorder (RBD) and Parkinson's Disease (PD).

Two different models exist:
1. The Pretrained Model, which is suitable for the general healthy population.
2. The Generalized Model, which is suitable for the general healthy population, as well as for individuals with RBD and/or PD.

At the moment, it works with MNE compatible files (.set, .vhdr, .edf).

The code has been tested on a local Windows machine (CPU) and a larger computing cluster (NVIDIA GPU, CUDA 11.8).

If you encounter any issues or have questions, feel free to write me: js@ece.au.dk

## Installation guide:
To get started, follow these steps:

**1) Make sure that you have the following installed:**

Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html\
Git: https://git-scm.com/install/

**2) In a terminal, clone this repository and enter the code directory:**
```console
git clone https://github.com/jesperstroem/U-Sleep-for-RBD-PD
cd U-Sleep-for-RBD-PD
```

**3) Create a new conda environment and activate it**

```console
conda create -n <your_environment_name> python=3.10
conda activate <your_environment_name>
```

**4) Install PyTorch version 2.0.1:**

Now you need to install PyTorch - the installation differs depending if you have a GPU or CPU available. If in doubt - just install the CPU version.

Run only **one** of the following commands:

**4A) GPU**:

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**4B) CPU**:

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

**5) Install PyTorch Lightning**

```console
python -m pip install lightning==2.1.3 torch==2.0.1
```

**6) Run the following command:**

```console
python -m pip install .
```

## Demo
Now you can start sleep staging your MNE compatible files.

Check the demo notebook (demo.ipynb) for an example of how to use the model and the different weights.