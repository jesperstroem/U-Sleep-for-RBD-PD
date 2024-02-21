from csdp_training.experiments.loso import CV_Experiment, create_loso_split
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_pipeline.pipeline_elements.pipe import ISample, IPipe, PipelineConfiguration
import torch
import neptune

api_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzViZjJlYy00NDNhLTRhN2EtOGZmYy00NDEzODBmNTgxYzMifQ=="
project = "NTLAB/ear-eeg"
name = "test"

class EESM2_Channel_Combiner(IPipe):
    def process(self, x: ISample) -> ISample:
        eegs: torch.Tensor = x.eeg

        left = eegs[0:2, :]
        right = eegs[2:4, :]

        left = torch.mean(left, dim=0, keepdim=True)
        right = torch.mean(right, dim=0, keepdim=True)

        result = left - right
        x.eeg = result

        return x

class First_Channel_Picker(IPipe):
    def process(self, x: ISample) -> ISample:

        x.eeg = torch.index_select(x.eeg, dim=0, index=torch.tensor([0]))
        x.eog = torch.index_select(x.eog, dim=0, index=torch.tensor([0]))
 
        return x 

def main():
    # logging_run = neptune.init_run(project=project,
    #                                api_token=api_key,
    #                                name=name,
    #                                mode="sync")
    
    # EAR EEG SETUP
    #picker = EESM2_Channel_Combiner()
    #checkpoint_path = "C:/Users/au588953/Git Repos/USleep Pipeline/weights/onechannel.ckpt"
    #datasets = ["C:/Users/au588953/Big_Sleep_Set/sedf_st.hdf5"]

    # PSG SETUP
    #picker = First_Channel_Picker()
    checkpoint_path = "C:/Users/au588953/Git Repos/USleep Pipeline/weights/twochannel.ckpt"
    datasets = ["C:/Users/au588953/Big_Sleep_Set/sedf_st.hdf5"]

    pipeline_configuration = PipelineConfiguration(train=[], 
                                                   val=[], 
                                                   test=[])


    net = USleep_Lightning.load_from_checkpoint(lr=0.0001,
                                                batch_size=64,
                                                checkpoint_path=checkpoint_path)

    loso = CV_Experiment(base_net=net,
                           dataset_paths=datasets,
                           num_folds=10,
                           training_epochs=1,
                           batch_size=64,
                           batches_per_epoch=1,
                           pick_all_channels=[False, False, True],
                           pipeline_configuration=pipeline_configuration,
                           neptune_run=None,
                           test_first=True)

    loso.run_training()

if __name__=="__main__": 
    main()