import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from neptune.utils import stringify_unsupported
import h5py
from sklearn.model_selection import train_test_split
from csdp_pipeline.pipeline_elements.sampler import Random_Sampler
from csdp_pipeline.pipeline_elements.determ_sampler import Determ_sampler
from csdp_pipeline.pipeline_elements.pipe import PipelineConfiguration, SamplerConfiguration
from csdp_pipeline.factories.dataloader_factory import Dataloader_Wrapper
from csdp_training.lightning_models.usleep import USleep_Lightning
from copy import deepcopy
from pytorch_lightning.loggers import NeptuneLogger
import neptune
from csdp_pipeline.pipeline_elements.pipe import Split, Dataset_Split
from sklearn.model_selection import KFold
import os
import json

def save_split_data(split_data: list[Split], exp_name):
    cwd = os.getcwd()

    os.makedirs(f"{cwd}/splits/{exp_name}")

    for i, split in enumerate(split_data):
        dic = split.get_dict()
        
        with open(f"{cwd}/splits/{exp_name}/{i}.json", 'w') as fp:
            json.dump(dic, fp)

def create_global_split(dataset_filepaths: list[str]):
    split_data = Split()
    
    for dataset_filepath in dataset_filepaths:
        with h5py.File(dataset_filepath, "r") as hdf5:
            subs = list(hdf5.keys())

        split_data.dataset_splits.append(Dataset_Split(dataset_filepath,
                                                       [],
                                                       [],
                                                       subs))
    
    return split_data

def create_loso_split(dataset_filepaths: list[str],
                      num_folds,
                      num_validation_subjects = 1) -> [Split]:
    all_subs = []
    all_split_data: list[Split] = []

    for file in dataset_filepaths:
        with h5py.File(file, "r") as hdf5:
            subs = list(hdf5.keys())

            subs = [(file, sub) for sub in subs]
            all_subs.extend(subs)
    
    kf = KFold(n_splits=num_folds,
               shuffle=True)

    for _, (train_index, test_index) in enumerate(kf.split(all_subs)):
        split_data = Split()
        
        train_records = [all_subs[i] for i in train_index]

        train_records, val_records = train_test_split(train_records,
                                                      test_size=num_validation_subjects)

        test_records = [all_subs[i] for i in test_index]
        
        for file in dataset_filepaths:
            dataset_split = Dataset_Split(file)
            dataset_split.train = [x[1] for x in list(filter(lambda x: file == x[0], train_records))]
            dataset_split.val = [x[1] for x in list(filter(lambda x: file == x[0], val_records))]
            dataset_split.test = [x[1] for x in list(filter(lambda x: file == x[0], test_records))]

            split_data.dataset_splits.append(dataset_split)
        
        all_split_data.append(split_data)

    return all_split_data

class CV_Experiment:
    def __init__(self,
                 base_net: USleep_Lightning,
                 dataset_paths: list[str],
                 training_epochs: int,
                 batch_size: int,
                 num_folds: int,
                 earlystopping_patience: int,
                 num_validation_subjects: int = 1,
                 batches_per_epoch: int = 100,
                 pick_all_channels: [bool] = [False, False, False],
                 test_first: bool = False,
                 pipeline_configuration: PipelineConfiguration = PipelineConfiguration(),
                 experiment_name: str = "LOSO",
                 neptune_run: neptune.Run | None = None):
        """_summary_

        Args:
            base_net (USleep_Lightning): A U-Sleep lightning model
            dataset_paths (list[str]): A list of exact filepaths to the datasets used for the experiment
            training_epochs (int): Number of training epochs per fold
            batch_size (int): Batchsize during training
            num_val_subjects (int, optional): The number of validation subjects per fold. Defaults to 1.
            batches_per_epoch (int, optional): Number of mini-batches per epoch. Defaults to 100.
            pick_all_channels (bool, optional): If True, the sampled data will contain all available channels. If False, one random EEG and EOG is picked. Defaults to False.
            test_first (bool, optional): If True, the base model will be tested first on all records. Defaults to False.
            pipeline_configuration (PipelineConfiguration, optional): A desired pipeline configuration. The default parameter has no pipes. Defaults to PipelineConfiguration().
            experiment_name (str, optional): The name of the experiment and the name of the test output folder. Defaults to "LOSO".
            neptune_run (neptune.Run | None, optional): An initialized neptune logging run. Defaults to None.
        """

        self.batches_per_epoch = batches_per_epoch
        self.pick_all_channels = pick_all_channels
        self.dataset_paths = dataset_paths
        self.pipeline_configuration = pipeline_configuration
        self.experiment_name = experiment_name
        self.training_epochs = training_epochs
        self.neptune_run = neptune_run
        self.base_net = base_net
        self.batch_size = batch_size
        self.earlystopping_patience = earlystopping_patience
        self.num_folds = num_folds
        self.num_validation_subjects = num_validation_subjects
        self.split_data = create_loso_split(dataset_paths,
                                            num_folds=num_folds,
                                            num_validation_subjects=num_validation_subjects)
        
        save_split_data(self.split_data, experiment_name)

        self.test_first = test_first
        self.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    def run_training(self):
        if self.test_first == True:
            global_split = create_global_split(self.dataset_paths)
            wrapper = self.__create_wrapper(global_split, self.batch_size)

            trainer = self.__init_trainer(max_epochs=self.training_epochs,
                                          split_data = global_split,
                                          split_name="Global Test")
            
            base_net = deepcopy(self.base_net)
            
            self.__test(trainer, wrapper, base_net, split_name="Global Test", load_best_model=False)

        for i, split in enumerate(self.split_data):
            # f = list(filter(lambda x: len(x.test) > 0, split.dataset_splits))
            # split_name = f[0].test[0]
            split_name = f"Split_{i}"

            wrapper = self.__create_wrapper(split, self.batch_size)

            trainer = self.__init_trainer(max_epochs=self.training_epochs,
                                          split_data=split,
                                          split_name=split_name)

            base_net = deepcopy(self.base_net)

            net, trainer = self.__train(base_net,
                                         wrapper,
                                         trainer)

            self.__test(trainer,
                        wrapper,
                        net=net,
                        split_name=split_name)

    def __train(self,
                net: USleep_Lightning,
                wrapper: Dataloader_Wrapper,
                trainer: pl.Trainer):
        
        train_loader = wrapper.training_loader(num_workers=8)
        val_loader = wrapper.validation_loader(num_workers=1)

        trainer.fit(net, train_loader, val_loader)

        return net, trainer


    def __test(self,
               trainer: pl.Trainer,
               wrapper: Dataloader_Wrapper,
               net: USleep_Lightning,
               split_name: str,
               load_best_model = True):        
        loader = wrapper.testing_loader(num_workers=1)

        net.run_test(trainer, 
                     net, 
                     loader, 
                     output_folder_prefix=f"{self.experiment_name}/split_{split_name}",
                     load_best_model=load_best_model)

    def __init_trainer(self,
                       max_epochs: int,
                       split_data: Split,
                       split_name: str) -> pl.Trainer:
        
        checkpoint_callback = ModelCheckpoint(filename=f"best-{split_name}", monitor="valKap", mode="max")
        
        early_stopping = EarlyStopping(
            monitor="valKap",
            min_delta=0.00,
            patience=self.earlystopping_patience,
            verbose=True,
            mode="max"
        )

        callbacks = [checkpoint_callback, early_stopping]

        if self.neptune_run != None:
            self.neptune_run[f"{split_name}/split_data"] = stringify_unsupported(split_data.get_dict())

            logger = NeptuneLogger(run=self.neptune_run,
                                   prefix=split_name)
        else:
            logger = None
        
        trainer = pl.Trainer(logger=logger,
                             max_epochs=max_epochs,
                             callbacks=callbacks,
                             accelerator=self.accelerator,
                             devices=1,
                             num_nodes=1)
        
        return trainer
    
    def __create_wrapper(self,
                         split: Split,
                         batch_size):
        
        train_sampler = Random_Sampler(split,
                                    split_type="train",
                                    num_epochs=35,
                                    get_all_channels=self.pick_all_channels[0],
                                    num_iterations=batch_size*self.batches_per_epoch)
        
        val_sampler = Determ_sampler(split,
                                     get_all_channels=self.pick_all_channels[1],
                                    split_type="val")
        
        test_sampler = Determ_sampler(split,
                                      get_all_channels=self.pick_all_channels[2],
                                      split_type="test")
        
        samplers = SamplerConfiguration(train_sampler,
                                        val_sampler,
                                        test_sampler)
        
        pipes = self.pipeline_configuration

        wrapper = Dataloader_Wrapper(batch_size,
                                     samplers,
                                     pipes)
    
        return wrapper
