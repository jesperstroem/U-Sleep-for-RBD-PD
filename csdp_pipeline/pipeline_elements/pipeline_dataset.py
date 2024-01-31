
import torch
from .pipe import Pipeline
from csdp_pipeline.pipeline_elements.pipe import ISample, IPipe, ISampler

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, sampler: ISampler, iterations: int):
        self.sampler = sampler
        self.iterations = iterations
        
    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        return self.sampler.get_sample(idx)