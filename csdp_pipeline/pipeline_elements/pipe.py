from abc import ABC, abstractmethod, abstractproperty
import torch

class Dataset_Split():
    dataset_filepath: str
    train: list[str]
    val: list[str]
    test: list[str]

    def __init__(self,
                 dataset_filepath: str,
                 train: list[str] = [],
                 val: list[str] = [],
                 test: list[str] = []):
        self.dataset_filepath = dataset_filepath
        self.train = train
        self.val = val
        self.test = test

class Split():
    dataset_splits: list[Dataset_Split]

    def __init__(self,
                 dataset_splits: list[Dataset_Split] = []):
        self.dataset_splits = dataset_splits

class ITag:
    dataset: str
    subject: str
    record: str
    eeg: str
    eog: str
    start_idx: str
    end_idx: str

    def __init__(self,
                 dataset: str = "",
                 subject: str = "",
                 record : str = "",
                 eeg = "",
                 eog = "",
                 start_idx = -1,
                 end_idx = -1):
        self.dataset = dataset
        self.subject = subject
        self.record = record
        self.eeg = eeg
        self.eog = eog
        self.start_idx = start_idx
        self.end_idx = end_idx

class ISample:
    def __init__(self, index: int):
        self.index = index

    index: int
    eeg: torch.Tensor | None
    eog: torch.Tensor | None
    labels: torch.Tensor | None
    tag: ITag | None

class ISampler:
    @abstractmethod
    def get_sample(self, index) -> ISample:
        pass

    num_samples: int

class IBatch:
    eeg: torch.Tensor
    eog: torch.Tensor
    labels: torch.Tensor
    size: int
    tags: list[ITag]

    def __init__(self,
                 samples: [ISample]):
        eegs = [sample.eeg for sample in samples]
        eogs = [sample.eog for sample in samples]
        labels = [sample.labels for sample in samples]

        self.eeg = torch.stack(eegs)
        self.eog = torch.stack(eogs)

        self.size = len(samples)
        self.labels = torch.stack(labels)
        self.tags = [sample.tag for sample in samples]

class IPipe(ABC):
    @abstractmethod
    def process(x: IBatch) -> IBatch:
        pass

class Pipeline:
    def __init__(self,
                 pipes: list[IPipe]):
        self.pipes = pipes

    def preprocess(self, batch) -> ISample:
        for _, p in enumerate(self.pipes):
            batch = p.process(batch)
        
        return batch
    
class PipelineConfiguration:
    def __init__(self, 
                 train: [IPipe] = [],
                 val: [IPipe] = [], 
                 test: [IPipe] = []):
        self.train_pipes = train
        self.val_pipes = val
        self.test_pipes = test
    
    def get_pipe_by_stage(self, stage: str):
        assert stage == "train" or stage == "val" or stage == "test"

        if stage == "train":
            return self.train_pipes
        elif stage == "val":
            return self.val_pipes
        else:
            return self.test_pipes
        
class SamplerConfiguration:
    def __init__(self, 
                 train: ISampler,
                 val: ISampler, 
                 test: ISampler):
        self.train_sampler = train
        self.val_sampler = val
        self.test_sampler = test

    def get_sampler_by_stage(self, stage: str):
        assert stage == "train" or stage == "val" or stage == "test"

        if stage == "train":
            return self.train_sampler
        elif stage == "val":
            return self.val_sampler
        else:
            return self.test_sampler