from csdp_training.experiments.loso import LOSO_Experiment
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_pipeline.pipeline_elements.pipe import IBatch, IPipe, PipelineConfiguration
import torch
import neptune

api_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzViZjJlYy00NDNhLTRhN2EtOGZmYy00NDEzODBmNTgxYzMifQ=="
project = "NTLAB/ear-eeg"
name = "test"

class Channel_Picker(IPipe):
    def process(self, x: IBatch) -> IBatch:

        x.eeg = torch.index_select(x.eeg, dim=1, index=torch.tensor([0]))
        x.eog = torch.index_select(x.eog, dim=1, index=torch.tensor([0]))
        
        return x 

def main():
    logging_run = neptune.init_run(project=project,
                                   api_token=api_key,
                                   name=name,
                                   mode="sync")

    picker = Channel_Picker()

    pipeline_configuration = PipelineConfiguration(train=[picker], val=[picker], test=[picker])

    net = USleep_Lightning.load_from_checkpoint(lr=0.0001,
                                                batch_size=64,
                                                checkpoint_path="C:/Users/au588953/Git Repos/CSDP/weights/best-usleep.ckpt")

    loso = LOSO_Experiment(base_net=net,
                           dataset_path="C:/Users/au588953/Big_Sleep_Set/sedf_st.hdf5",
                           training_epochs=1,
                           batch_size=64,
                           pipeline_configuration=pipeline_configuration,
                           neptune_run=logging_run,
                           test_first=True)

    loso.run_training()

if __name__=="__main__": 
    main()