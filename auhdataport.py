from csdp_datastore.base import BaseDataset
import os
from mne.io import read_raw_eeglab
import h5py

class PDRBD_PB(BaseDataset):
    def label_mapping(self):
        return {
            1: self.Labels.Wake,
            2: self.Labels.REM,
            3: self.Labels.N1,
            4: self.Labels.N2,
            5: self.Labels.N3,
            6: self.Labels.UNKNOWN,
            7: self.Labels.UNKNOWN,
            8: self.Labels.UNKNOWN,
            9: self.Labels.UNKNOWN
        }
    
    def dataset_name(self):
        return "PDRBD_PB"
    
    def list_records(self, basepath) -> dict[str, list[tuple]]:
        paths_dict = {}
        
        sub_paths = os.listdir(basepath)
        sub_paths = [sub for sub in sub_paths if sub.startswith("sub")]

        for sub_path in sub_paths:
            record_paths = os.listdir(f"{basepath}/{sub_path}")
            record_paths = [rec for rec in record_paths if rec.startswith("ses")]

            paths_dict[sub_path] = []

            for rec_path in record_paths:
                record_path = f"{basepath}/{sub_path}/{rec_path}/eeg"
                psg_path = f"{record_path}/{sub_path}_{rec_path}_task-sleep_acq-PSG_eeg.set"
                hyp_path = f"{record_path}/{sub_path}_{rec_path}_task-sleep_acq-scoring_events.tsv"
                paths_dict[sub_path].append((psg_path, hyp_path))
        
        return paths_dict
    
    def read_psg(self, record):
        psg_path, hyp_path = record
        
        x = dict()
        y = []

        try:
            data = read_raw_eeglab(psg_path)
        except Exception as error:
            try:
                with h5py.File(psg_path, 'r') as f:
                    data = f
            except:
                print(f"ERROR READING DATA FOR PATH: {psg_path}")
                return None

        channels = data.ch_names
        info = data.info
        sr = int(info["sfreq"])
        channels = [chan for chan in channels if chan in self.channel_mapping().keys()]
        #print(channels)

        for chan in channels:
            pass


        exit()
        
        # with File(psg_path, "r") as h5:
        #     h5channels = h5.get("channels")
            
        #     for channel in self.channel_mapping().keys():
        #         channel_data = h5channels[channel][:]
                
        #         x[channel] = (channel_data, self.sample_rate()) # We are assuming sample rate is same across channels
        
        # with open(hyp_path) as f:
        #     hypnogram = f.readlines()

        #     for element in hypnogram:
        #         prev_stages_time, stage_time, label = element.rstrip().split(",")
        #         stage_time = int(stage_time)

        #         n_epochs_in_stage = int(stage_time/30)

        #         for label_entry in range(n_epochs_in_stage):
        #             stg = label
        #             assert stg != None
                    
        #             y.append(stg)
                    
        return x, y

class AUH(PDRBD_PB):
    def channel_mapping(self):
        return {'EOG2:M2': self.Mapping(self.TTRef.ER, self.TTRef.RPA), 
         'M2': self.Mapping(self.TTRef.RPA, self.TTRef.Cz), 
         'C4': self.Mapping(self.TTRef.C4, self.TTRef.Cz), 
         'C4:M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA), 
         'F4:M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA), 
         'O1:M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA), 
         'C3:M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA), 
         'F3:M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA), 
         'O2': self.Mapping(self.TTRef.O2, self.TTRef.Cz),
         'M1': self.Mapping(self.TTRef.LPA, self.TTRef.Cz), 
         'F4': self.Mapping(self.TTRef.F4, self.TTRef.Cz), 
         'EOG1:M1': self.Mapping(self.TTRef.EL, self.TTRef.LPA), 
         'O2:M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA), 
         'EOG2': self.Mapping(self.TTRef.ER, self.TTRef.Cz), 
         'F3': self.Mapping(self.TTRef.F3, self.TTRef.Cz), 
         'C3': self.Mapping(self.TTRef.C3, self.TTRef.Cz), 
         'EOG1': self.Mapping(self.TTRef.EL, self.TTRef.Cz), 
         'EOG1:M2': self.Mapping(self.TTRef.EL, self.TTRef.RPA), 
         'O1': self.Mapping(self.TTRef.O1, self.TTRef.Cz), 
         'EOG2:M1': self.Mapping(self.TTRef.ER, self.TTRef.LPA), 
    }
    
def main():
    datapath = "O:/Tech_Neuro247/BIDS/AUH"

    d = AUH(datapath,
            "")
    
    d.port_data()

if __name__ == "__main__":
    main()