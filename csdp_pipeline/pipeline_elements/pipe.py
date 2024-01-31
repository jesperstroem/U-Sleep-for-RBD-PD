from abc import ABC, abstractmethod, abstractproperty
import torch

class ITag:
    dataset: str
    subject: str
    record: str
    eeg: str
    eog: str
    start_idx: str
    end_idx: str

    def __init__(self,
                 dataset: str,
                 subject: str,
                 record : str,
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