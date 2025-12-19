# 
This repository will show you how to install and use the U-Sleep model.
At the moment, it works with MNE compatible files (.set, .vhdr, .edf)
The code has been tested on a local Windows machine (CPU) and a larger computing cluster (NVIDIA GPU, CUDA 11.8).

If you encounter any issues or have questions, feel free to write me: js@ece.au.dk

## Installation guide:
To get started, follow these steps:

1) Make sure that you have the following installed:
Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
Git: https://git-scm.com/install/

2) In a terminal, clone this repository and enter the code directory:
```console
git clone XXXXXXXXXXXXXXXXX
cd XXXXXXXXXXXXXXXXXXXXXXX
```

3) Create a new conda environment and activate it
```console
conda create -n <your_environment_name> python=3.10
conda activate <your_environment_name>
```

4) Install PyTorch version 2.0.1:
If you have a CUDA enabled GPU, run the following:

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

If you do not have a CUDA enabled GPU (or simply wish to use your CPU), run the following instead:

```console
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

5) Install PyTorch Lightning
```console
python -m pip install lightning==2.1.3 torch==2.0.1
```

6) Run the following command:
```console
python -m pip install .
```

## Demo
Now that you have followed the installation instructions, you can start sleep staging your MNE compatible files.

Check the demoscript (demo.py) for an example of how to use the model.
The repository includes the Pretrained Model and the Generalized Model, as described in XXX.