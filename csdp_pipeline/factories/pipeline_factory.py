from abc import ABC, abstractmethod, abstractproperty
from csdp_pipeline.pipeline_elements.sampler import Sampler
from csdp_pipeline.pipeline_elements.resampler import Resampler
from csdp_pipeline.pipeline_elements.spectrogram import Spectrogram
from csdp_pipeline.pipeline_elements.determ_sampler import Determ_sampler
from csdp_pipeline.pipeline_elements.pipe import IPipe

class IPipeline_Factory(ABC):
    @abstractmethod
    def pipes_for_stage(self, stage: str) -> [IPipe]:
        pass

class USleep_Pipeline_Factory(IPipeline_Factory):
    def __init__(self,
                 hdf5_base_path, 
                 split_path, 
                 trainsets, 
                 valsets, 
                 testsets, 
                 sub_percentage=1.0):
        self.split_path = split_path
        self.hdf5_base_path = hdf5_base_path
        self.trainsets = trainsets
        self.valsets = valsets
        self.testsets = testsets
        self.sub_percentage = sub_percentage

        self.train_pipes = [
        ]

        self.val_pipes = [
        ]

        self.test_pipes = [
        ]

    def pipes_for_stage(self, stage: str) -> [IPipe]:
        assert stage == "train" or stage == "val" or stage == "test"

        if stage == "train":
            return self.train_pipes
        elif stage == "val":
            return self.val_pipes
        else:
            return self.test_pipes

class LSeqSleepNet_Pipeline_Factory(IPipeline_Factory):
    def __init__(
        self, hdf5_base_path, split_path, trainsets, valsets, testsets, num_epochs
    ):
        self.split_path = split_path
        self.hdf5_base_path = hdf5_base_path
        self.trainsets = trainsets
        self.valsets = valsets
        self.testsets = testsets
        self.num_epochs = num_epochs

    def create_training_pipeline(self):
        return [
            Sampler(
                self.hdf5_base_path,
                self.trainsets,
                split_type="train",
                num_epochs=self.num_epochs,
                split_file_path=self.split_path,
                subject_percentage=1.0,
            ),
            Resampler(source_sample=128, target_sample=100),
            Spectrogram(),  # default parameters match SeqSleepNet and derivatives
        ]

    def __evaluation_pipeline(self, split_type, datasets):
        return [
            Determ_sampler(
                self.hdf5_base_path,
                datasets,
                split_type=split_type,
                num_epochs=self.num_epochs,
                split_file=self.split_path,
                get_all_channels= True if split_type == "test" else False,
            ),
            Resampler(source_sample=128, target_sample=100),
            Spectrogram(),
        ]

    def create_validation_pipeline(self):
        return self.__evaluation_pipeline("val", self.valsets)

    def create_test_pipeline(self):
        return self.__evaluation_pipeline("test", self.testsets)
