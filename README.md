# CSDP - Common Sleep Data Pipeline

This repository contains functions and classes for preprocessing and data-loading PSG data, which can then be used for training/predicting with the U-Sleep model.

There are three submodules:
- "csdp_datastore": Preprocessing of PSG datasets. If you already have access to compatible HDF5 files (for example from ERDA), you can ignore this submodule.
- "csdp_pipeline": PyTorch based dataloading of the preprocessed HDF5 files from the datastore submodule.
- "csdp_training": PyTorch Lightning based module of U-Sleep to be used for training, validation, test and simple predictions.

## Installation guide
To install the pipeline, follow these steps:

1) Make sure that you have Conda installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2) Create a new conda environment and activate it:

```console
conda create -n <your_environment_name> python=3.10
activate <your_environment_name>
```

You CAN use an existing Conda environment, but keep in mind that various dependencies will be changed, so only do so with caution.

3) Install PyTorch version 2.0.1 (I will update this dependency in the near future, but for now it works with 2.0.1).
If you have a CUDA enabled GPU, run the following:

```console
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

If not, then you will be running everything on the CPU and you should therefore run the following instead:

```console
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
```

4) Install PyTorch Lightning
```console
python -m pip install lightning==2.1.3 torch==2.0.1
```

5) Install our internal PyTorch package "ML Architectures"
```console
python -m pip install git+https://gitlab.au.dk/tech_ear-eeg/ml_architectures.git@main
```

6) Install this repository as a package:
```console
python -m pip install git+https://gitlab.au.dk/tech_ear-eeg/sleep-code/common-sleep-data-pipeline.git@main
```


## Preprocessing
This submodule contains classes that can take known datasets and transform them into HDF5 files, which are used by the csdp_pipeline submodule.
There is an existing datastore available at ERDA: NTData/Big_Sleep_Set/pt_processed/v2, so that you don't need to run this step yourself.

If you wish to run the preprocessing yourself, you need to download all the raw data. Note that to download the data from https://sleepdata.org/, you need a personal download token from their website, and you need the NSRR ruby gem installed: https://github.com/nsrr/nsrr-gem. When downloaded, point to the location of the raw data. See example below.

```python

from csdp_datastore import ABC

raw_data_path = "/path/to/raw/data/location"
output_data_path = "/path/to/output/data/file"

a = ABC(dataset_path = raw_data_path,
        output_path = output_data_path,
        output_sample_rate = 128)

# This call starts the preprocessing
a.port_data()
```

## Using the dataloading pipeline without PyTorch Lightning
This submodule contains the dataloading logic for the HDF5 files created by csdp_datastore.
These are compatible with the V2 dataset from ERDA: NTData/Big_Sleep_Set/pt_processed/v2

The following example shows how to use the dataloaders in a pure PyTorch manner (no PyTorch Lightning involved).
The example specifies the path to the datafiles and dataloader details.
It also shows how to feed the loaded data through U-Sleep to get predictions.

```python
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.models import Split
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, Determ_sampler, SamplerConfiguration
from ml_architectures.usleep.usleep import USleep
import torch

# Input your directory path here
base_path = ""

batch_size = 64

base_hdf5_path = f"{base_path}/data" # Base-path to the HDF5 files. Should contain at least one HDF5 file
split_file_path = base_path # Where to save the generated dataloading split file
sleep_epochs_per_sample = 35 # Number of sleep epochs per sample
num_batches = 10 # Number of batches per training epoch
training_iterations = batch_size*num_batches # The resulting number of sampling iterations in the data
dataloader_workers = 1 # Number of dataloader workers. Set to number of CPU cores available.

# Create a random subject-based split in the data. Can be dumped to a file, which can later be used with Split.File(....)
split = Split.random(base_hdf5_path=base_hdf5_path,
                     split_name="demo")
split.dump_file(path=split_file_path)

# Define the samplers and build the dataloaders
train_sampler = Random_Sampler(split,
                               num_epochs=sleep_epochs_per_sample,
                               num_iterations=training_iterations)

val_sampler = Determ_sampler(split,
                             split_type="val")

test_sampler = Determ_sampler(split,
                              split_type="test")

samplers = SamplerConfiguration(train_sampler,
                                val_sampler,
                                test_sampler)

factory = Dataloader_Factory(training_batch_size=batch_size,
                             samplers=samplers)

train_loader = factory.training_loader(num_workers=dataloader_workers)
val_loader = factory.validation_loader(num_workers=dataloader_workers)
test_loader = factory.testing_loader(num_workers=dataloader_workers)

# Load the data
iterator = iter(train_loader)
batch = next(iterator)

eeg_signals = batch["eeg"]
eog_signals = batch["eog"]
labels = batch["labels"]
meta_data = batch["tag"]

# Feed the sampled data through U-Sleep
batched_data = torch.cat([eeg_signals, eog_signals], dim=1)

usleep_instance = USleep()

predictions = usleep_instance(batched_data.float())
```

## Dataloading and training of U-Sleep with PyTorch Lightning
This submodule adds the use of a PyTorch Lightning module to train, finetune or predict with U-Sleep.

The following example shows how to start training, validating and testing.
It is entirely plausible to remove the training&validation step or the test step, depending on your needs.

After training, model weights are saved into the "lighting_logs" folder.

```python
import torch
import pytorch_lightning as pl
from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, Determ_sampler, SamplerConfiguration
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory
from csdp_pipeline.pipeline_elements.models import Split

# Input your directory path here
base_path = ""

batch_size = 64
learning_rate = 0.0001
complexity_factor = 1.67
depth = 12
max_training_epochs = 2

accelerator = "cuda" if torch.cuda.is_available() else "cpu"

pretrained_path = None
results_folder = f"{base_path}/results"

base_hdf5_path = f"{base_path}/data" # Base-path to the HDF5 files. Should contain at least one HDF5 file
split_file_path = base_path # Where to save the generated dataloading split file
sleep_epochs_per_sample = 35 # Number of sleep epochs per sample
num_batches = 3 # Number of batches per training epoch
training_iterations = batch_size*num_batches # The resulting number of sampling iterations in the data
dataloader_workers = 1 # Number of dataloader workers. Set to number of CPU cores available.

# Create a random subject-based split in the data. Can be dumped to a file, which can later be used with Split.File(....)
split = Split.random(base_hdf5_path=base_hdf5_path,
                     split_name="demo")
split.dump_file(path=split_file_path)

train_sampler = Random_Sampler(split,
                               num_epochs=sleep_epochs_per_sample,
                               num_iterations=training_iterations)

val_sampler = Determ_sampler(split,
                             split_type="val")

test_sampler = Determ_sampler(split,
                              split_type="test")

samplers = SamplerConfiguration(train_sampler,
                                val_sampler,
                                test_sampler)

data_fac = Dataloader_Factory(training_batch_size=batch_size,
                              samplers=samplers)

model_fac = USleep_Factory(lr=learning_rate,
                           batch_size=batch_size,
                           complexity_factor=complexity_factor,
                           depth=depth)

# Load a clean model or from a checkpoint
if pretrained_path == None:
    net: USleep_Lightning = model_fac.create_new_net()
else:
    net: USleep_Lightning = model_fac.create_pretrained_net(pretrained_path)

trainer = pl.Trainer(max_epochs=max_training_epochs,
                     accelerator=accelerator,
                     devices=1,
                     num_nodes=1)

train_loader = data_fac.training_loader(num_workers=dataloader_workers)
val_loader = data_fac.validation_loader(num_workers=dataloader_workers)

# Start training
trainer.fit(net, train_loader, val_loader)

test_loader = data_fac.testing_loader(num_workers=dataloader_workers)

# Start test and output results to a specific folder
net.run_test(trainer,
             test_loader,
             output_folder_prefix=results_folder)
```

## Using the pipeline for cross-validation
A cross-validation functionality has been implemented.
See the below example if you want to run a cross-validation experiment across one or multiple HDF5 datasets.

```python
from csdp_training.experiments.cv import CV_Experiment
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_pipeline.pipeline_elements.pipeline import PipelineConfiguration

# Input your working directory
base_path = ""

logging_folder = f"{base_path}"

# Path to the HDF5 files
data_path = f"{base_path}/data"

# Names of the HDF5 files in the data directory to use
datasets = ["abc.hdf5", "homepap.hdf5"]
split_filepath = None

batch_size = 64
learning_rate = 0.0001
max_training_epohcs = 1
num_folds = 2
early_stopping_patience = 1
batches_per_training_epoch = 2
num_validation_subjects = 2

pipeline_configuration = PipelineConfiguration()

net = USleep_Lightning(lr=learning_rate, batch_size=batch_size)

cv = CV_Experiment(base_net=net,
                     base_data_path=data_path,
                     datasets=datasets,
                     training_epochs=max_training_epohcs,
                     batch_size=batch_size,
                     num_folds=num_folds,
                     earlystopping_patience=early_stopping_patience,
                     batches_per_epoch=batches_per_training_epoch,
                     logging_folder=logging_folder,
                     num_validation_subjects=num_validation_subjects,
                     pipeline_configuration=pipeline_configuration,
                     split_filepath=split_filepath)

cv.run()
```

## Predicting on a single MNE compatible file
If you have one or a few MNE compatible files (.edf, .set, etc.) that you need sleep-staged by U-Sleep, the following example code will do so:


