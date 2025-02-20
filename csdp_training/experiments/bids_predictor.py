from csdp_pipeline.pipeline_elements.mne_sleep_dataset import sleep_dataset_from_paths
import mne_bids as mb
import torch
from ..lightning_models.factories.lightning_model_factory import USleep_Factory, USleep_Lightning

class BIDS_USleep_Predictor():
    def __init__(self,
                 data_dir, 
                 data_extension, 
                 data_task, 
                 subjects=None, 
                 sessions=None):
        self.data_dir = data_dir
        self.data_extension = data_extension
        self.data_task = data_task
        self.subjects = subjects
        self.sessions = sessions

        self.filepaths = mb.find_matching_paths(data_dir,extensions=data_extension,tasks=data_task,subjects=subjects, sessions=sessions)

    def get_available_channels(self, overlapping=False):
        ch_names = sleep_dataset_from_paths.get_available_channels(self.filepaths)

        if overlapping==True:
            return set.intersection(*[set(x) for x in ch_names])
        else:
            return ch_names
        
    def predict_all(self, lighting_checkpoint, dataset, eeg_indexes=None, eog_indexes=None):
        fac = USleep_Factory(lr=0.0001, batch_size=64)
        usleep: USleep_Lightning = fac.create_pretrained_net(lighting_checkpoint)
        num_channels = usleep.num_channels

        if num_channels==2:
            if (eeg_indexes==None) or (eog_indexes==None):
                raise ValueError("For a model with 2 channels EEG and EOG indexes must be specified")
            
        dataLoader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        all_preds = []

        for _, (x, _) in enumerate(dataLoader):
            if num_channels == 1:
                output = usleep.majority_vote_prediction(x_eegs=x)
            elif num_channels == 2:
                eegs = torch.index_select(x, dim=1, index=torch.tensor(eeg_indexes))
                eogs = torch.index_select(x, dim=1, index=torch.tensor(eeg_indexes))

                output = usleep.majority_vote_prediction(x_eegs=eegs, x_eogs=eogs)
            else:
                raise ValueError('Unknown number of channels for U-Sleep')
            
            num_epochs = output[list(output.keys())[0]].shape[2]
            num_classes = 5
                
            votes = torch.zeros(num_classes, num_epochs) # fordi vi summerer løbende

            num_channels = len(list(output.items()))

            for item in output.items():
                pred = item[1]
                votes = torch.add(votes, pred)

            preds = torch.argmax(votes, axis=1)
            preds = torch.squeeze(preds)

            all_preds.append(preds)

        return all_preds

    def build_dataset(self, channel_names: list):
        data_set = sleep_dataset_from_paths(self.filepaths,
                                            scoring_paths=[],
                                            ch_names=channel_names,
                                            scoring_preprocess=None,
                                            fullRecords=True)
        return data_set