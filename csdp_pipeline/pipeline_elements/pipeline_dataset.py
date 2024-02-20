
import torch
from .pipe import Pipeline
from csdp_pipeline.pipeline_elements.pipe import ISample, IPipe, ISampler

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 sampler: ISampler,
                 pipes: [IPipe]):
        self.sampler = sampler
        self.iterations = sampler.num_samples
        self.pipes = pipes
        
    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        sample = self.sampler.get_sample(idx)
        
        for pipe in self.pipes:
            sample = pipe.process(sample)

        d = dict()

        d["eeg"] = sample.eeg
        d["eog"] = sample.eog
        d["labels"] = sample.labels
        d["tag"] = {"dataset": sample.tag.dataset,
                    "subject": sample.tag.subject,
                    "record": sample.tag.record}
                    #"eeg": sample.tag.eeg,
                    #"eog": sample.tag.eog,
                    #"start_idx": sample.tag.start_idx,
                    #"end_idx": sample.tag.end_idx}

        return d
