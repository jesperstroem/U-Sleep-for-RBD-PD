import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
                      num_validation_subjects = 1) -> [Split]:
    all_subs = []
    all_split_data: list[Split] = []

    for file in dataset_filepaths:
        with h5py.File(file, "r") as hdf5:
            subs = list(hdf5.keys())

            subs = [(file, sub) for sub in subs]
            all_subs.extend(subs)
    
    for test_sub in all_subs:
        rest_of_subs = list(filter(lambda x: x != test_sub, all_subs))
        
        train, val = train_test_split(rest_of_subs,
                                      test_size=num_validation_subjects)
        
        split_data = Split()

        for file in dataset_filepaths:
            dataset_split = Dataset_Split(file)
            dataset_split.train = [x[1] for x in list(filter(lambda x: file == x[0], train))]
            dataset_split.val = [x[1] for x in list(filter(lambda x: file == x[0], val))]
            dataset_split.test = [test_sub[1]] if test_sub[0] == file else []

            split_data.dataset_splits.append(dataset_split)
        
        all_split_data.append(split_data)

    return all_split_data

class LOSO_Experiment:
    def __init__(self,
                 base_net: USleep_Lightning,
                 dataset_paths: list[str],
                 training_epochs: int,
                 batch_size: int,
                 num_val_subjects: int = 1,
                 test_first: bool = False,
                 sampler_configuration: SamplerConfiguration = None,
                 pipeline_configuration: PipelineConfiguration = PipelineConfiguration(),
                 experiment_name: str = "LOSO",
                 neptune_run: neptune.Run | None = None,
                 pretrained_model: str = None):
        self.dataset_paths = dataset_paths
        self.pipeline_configuration = pipeline_configuration
        self.sampler_configuration = sampler_configuration
        self.experiment_name = experiment_name
        self.training_epochs = training_epochs
        self.neptune_run = neptune_run
        self.pretrained_model = pretrained_model
        self.base_net = base_net
        self.batch_size = batch_size
        self.split_data = create_loso_split(dataset_paths,
                                            num_validation_subjects=num_val_subjects)
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

        for split in self.split_data:
            f = list(filter(lambda x: len(x.test) > 0, split.dataset_splits))
            split_name = f[0].test[0]

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
       
        callbacks = [checkpoint_callback]

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
        
        if self.sampler_configuration != None:
            samplers = self.sampler_configuration
        else:                
            train_sampler = Random_Sampler(split,
                                        split_type="train",
                                        num_epochs=35,
                                        num_iterations=batch_size*1)
            
            val_sampler = Determ_sampler(split,
                                        split_type="val")
            
            test_sampler = Determ_sampler(split,
                                        split_type="test")
            
            samplers = SamplerConfiguration(train_sampler,
                                            val_sampler,
                                            test_sampler)
        
        pipes = self.pipeline_configuration

        wrapper = Dataloader_Wrapper(batch_size,
                                     samplers,
                                     pipes)
    
        return wrapper