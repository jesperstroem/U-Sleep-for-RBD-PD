# CSDP - Common Sleep Data pipeline

This repository contains a pipeline for preprocessing and loading of PSG data to be used for the training of U-Sleep in automatic sleep staging.

The repository has three submodules:
- "csdp_datastore": Preprocessing of PSG datasets
- "csdp_pipeline": Dataloading of the preprocessed data
- "csdp_training": Pytorch Lightning module of U-Sleep to be used for automatic sleep scoring with the dataloaders.

Every submodule can be used independently and should thus serve to satisfy most needs.

## Installation guide
Before installing the pipeline here are some prerequisites:

1) Make sure that you have Conda installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2) Create a new conda environment and activate it:

```console
conda create -n <your_environment_name> python=3.10
activate <your_environment_name>
```

3) Install PyTorch version 2.0.1 (I will update this dependency in the near future, but for now it works with 2.0.1).
If you are working on a machine with a CUDA enabled GPU, run the following:

```console
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

If not, then you will be running on CPU and you should type the following:

```console
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
```

4) Install PyTorch Lightning
python -m pip install lightning==2.1.3 torch==2.0.1

5) Install the dependency PyTorch package ML Architectures
```console
python -m pip install git+https://gitlab.au.dk/tech_ear-eeg/ml_architectures.git@main
```

6) Install this repository as a package:
```console
python -m pip install git+https://gitlab.au.dk/tech_ear-eeg/sleep-code/common-sleep-data-pipeline.git@main
```


## csdp_datastore

Before you can use the dataloaders and lightning modules, you need to download and preprocess the raw data. Note that to download the data from https://sleepdata.org/, you need a personal download token from their website, and you need the NSRR ruby gem installed: https://github.com/nsrr/nsrr-gem. When downloaded, point to the location of the raw data. See example below.

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

## csdp_pipeline

To use the implemented pytorch dataloaders, look at the following example

"hdf5_base_path" is the root path to your preprocessed HDF5 files from the common datastore.
train/val/test-sets is a list of datasets in the root path, that you want to use in the dataloading. It should be the name of the file without the hdf5 extension.
"data_split_path" is a json file containing the configuration of train/validation/test subjects. Look in the csdp_pipeline/splits if you want examples. If you leave it as "None", all subjects will be used from the listed datasets. If you specify the parameter "create_random_split" as True, then a random split json file will be created and used.


```python

from csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory

dataloader_factory = USleep_Dataloader_Factory(gradient_steps=100,
                                               batch_size=64,
                                               hdf5_base_path="/root/path/to/datasets",
                                               trainsets=["abc", "cfs"],
                                               valsets=["abc", "cfs"],
                                               testsets=["chat"],
                                               data_split_path="/path/to/split/file"
                                               create_random_split=False)

train_loader = dataloader_factory.create_training_loader(num_workers=1)
val_loader = dataloader_factory.create_validation_loader(num_workers=1)
test_loader = dataloader_factory.create_testing_loader(num_workers=1)

```

## csdp_training

To also use the implemented pytorch lightning versions of U-Sleep, see the following example.

If you want a pretrained model, you need to specify a checkpoint. A checkpoint for u-sleep is available in the checkpoints folder.

```python

from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory

model_factory = USleep_Factory(lr = 0.0001,
                               batch_size = 64,
                               initial_filters = 5,
                               complexity_factor = 1.67,
                               progression_factor = 2)

usleep = model_factory.create_new_net()
usleep_pretrained = model_factory.create_pretrained_net("/path/to/usleep/checkpoint/file")


```

## Making additions to the csdp_datastore submodule

To make an addition, you can either create a new class inheriting from the baseclass "BaseDataset" or you can inherit from a subclass. See the following examples of a "base" EESM class and a specialization inheriting from the EESM class. Check the baseclass documentation or the existing classes if in doubt on how to design the functions.

```python

class EESM(BaseDataset):
    def label_mapping(self):
        return {
            1: self.Labels.Wake,
            2: self.Labels.REM,
            3: self.Labels.N1,
            4: self.Labels.N2,
            5: self.Labels.N3,
            # Any other additions here
        }
        
    def dataset_name(self):
        return "eesm"

    def channel_mapping(self):
        return {
            "ELA": self.Mapping(self.EarEEGRef.ELA, self.EarEEGRef.REF),
            "ELB": self.Mapping(self.EarEEGRef.ELB, self.EarEEGRef.REF),
            # More channels can go here
        }    

    def list_records(self, basepath):

        paths_dict = dict()

        # Return value must be a dictionary
        # Every key is a subject ID, and the values are on the form (data_path, label_path), which must be absolute paths to the files containing data and labels.

        return paths_dict

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()
        y = []

        # Return value must be on the form: (Dictionary, Labels)
        # The dictionary has a key for every channel, and the values are (channel_data, sampling_rate)
        
        return x, y

class EESM_Raw(EESM):
    
    # Overriding the name of the dataset
    def dataset_name(self):
        return "eesm_raw"


    # Overriding the logic for how each record is read from disk, but keeping all other logic
    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()
        y = []

        return x, y

```
