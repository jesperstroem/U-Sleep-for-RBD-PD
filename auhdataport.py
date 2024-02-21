from csdp_datastore.base import BaseDataset
import os
from mne.io import read_raw_eeglab
import h5py
import pandas
import numpy as np
import mat73

class PDRBD_PB_Base(BaseDataset):
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
    
    def __front_align_data(self, signal, y, sr):
        diff = len(signal)-(len(y)*sr*30)
        signal_second_diff = diff/sr
        
        if signal_second_diff < 0:
            num_epochs_to_keep = int(np.floor(len(signal)/sr/30))
            data_to_keep = num_epochs_to_keep*sr*30
        else:
            num_epochs_to_keep = len(y)
            data_to_keep = len(y)*sr*30

        if signal_second_diff != 0:
            self.log_warning(f"Signal had second diff {signal_second_diff}")
        
        return signal[:data_to_keep], y[:num_epochs_to_keep]


    def __load_mat73file(self, psg_path, x, y):
        data = mat73.loadmat(psg_path)
        signals = data["data"][()]

        channels = data["chanlocs"]["labels"]
        channels = [ch[0] for ch in channels]
        sr = int(data["srate"][()])

        chans = [(i, chan) for i, chan in enumerate(channels) if chan in self.channel_mapping().keys()]
 
        for chan in chans:
            ch_idx = chan[0]
            ch_name = chan[1]
            signal = signals[ch_idx]
            signal, y = self.__front_align_data(signal, y, sr)

            x[ch_name] = (signal, sr)

        return x,y

    def __load_regular_eeglab(self, psg_path, x, y):
        data = read_raw_eeglab(psg_path)
        channels = data.ch_names
        channels = [chan for chan in channels if chan in self.channel_mapping().keys()]
        info = data.info
        sr = int(info["sfreq"])

        for chan in channels:
            signal = data.get_data(chan)[0]

            signal, y = self.__front_align_data(signal, y, sr)
                
            x[chan] = (signal, sr)

        return x,y

    def read_psg(self, record):
        psg_path, hyp_path = record
        
        x = dict()
        
        hyp_data = pandas.read_csv(hyp_path,sep='\t')
        y = hyp_data["staging"].tolist()

        try:
            x, y = self.__load_regular_eeglab(psg_path, x, y)
        except Exception as error:
            try:
                print(f"Could not read with MNE due to error: {error}")
                x, y = self.__load_mat73file(psg_path, x, y)
            except:
                print(f"ERROR READING DATA FOR PATH: {psg_path}")
                return None

        return x, y

class AUH(PDRBD_PB_Base):
    
    def dataset_name(self):
        return "PB_AUH"

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

class COLOGNE(PDRBD_PB_Base):
    
    def dataset_name(self):
        return "PB_COLOGNE"

    def channel_mapping(self):
        return  {'C3:M2': self.Mapping(self.TTRef.C3, self.TTRef.RPA), 
                'O2:A1': self.Mapping(self.TTRef.O2, self.TTRef.LPA), 
                'E2': self.Mapping(self.TTRef.ER, self.TTRef.Cz), 
                'M2': self.Mapping(self.TTRef.RPA, self.TTRef.Cz), 
                'E1': self.Mapping(self.TTRef.EL, self.TTRef.Cz), 
                'E2:M1': self.Mapping(self.TTRef.ER, self.TTRef.LPA), 
                'EOG1:A2': self.Mapping(self.TTRef.EL, self.TTRef.RPA), 
                'O1:M2': self.Mapping(self.TTRef.O1, self.TTRef.RPA), 
                'O1': self.Mapping(self.TTRef.O1, self.TTRef.Cz), 
                'C4:A1': self.Mapping(self.TTRef.C4, self.TTRef.LPA), 
                'O1:A2': self.Mapping(self.TTRef.O1, self.TTRef.RPA), 
                'EOG1': self.Mapping(self.TTRef.EL, self.TTRef.Cz), 
                'C3': self.Mapping(self.TTRef.C3, self.TTRef.Cz), 
                'C4': self.Mapping(self.TTRef.C4, self.TTRef.Cz), 
                'F4': self.Mapping(self.TTRef.F4, self.TTRef.Cz), 
                'F4:M1': self.Mapping(self.TTRef.F4, self.TTRef.LPA), 
                'EOG1:A1': self.Mapping(self.TTRef.EL, self.TTRef.LPA), 
                'EOG2:A1': self.Mapping(self.TTRef.ER, self.TTRef.LPA), 
                'M1': self.Mapping(self.TTRef.LPA, self.TTRef.Cz), 
                'EOG2:A2': self.Mapping(self.TTRef.ER, self.TTRef.RPA), 
                'F3:M2': self.Mapping(self.TTRef.F3, self.TTRef.RPA), 
                'F3:A2': self.Mapping(self.TTRef.F3, self.TTRef.RPA), 
                'O2': self.Mapping(self.TTRef.O2, self.TTRef.Cz), 
                'E1:M2': self.Mapping(self.TTRef.EL, self.TTRef.RPA), 
                'E1:M1': self.Mapping(self.TTRef.EL, self.TTRef.LPA), 
                'A2': self.Mapping(self.TTRef.RPA, self.TTRef.Cz), 
                'C3:A2': self.Mapping(self.TTRef.C3, self.TTRef.RPA), 
                'C4:M1': self.Mapping(self.TTRef.C4, self.TTRef.LPA), 
                'EOG2': self.Mapping(self.TTRef.ER, self.TTRef.Cz), 
                'O2:M1': self.Mapping(self.TTRef.O2, self.TTRef.LPA), 
                'E2:M2': self.Mapping(self.TTRef.ER, self.TTRef.RPA), 
                'A1': self.Mapping(self.TTRef.LPA, self.TTRef.Cz), 
                'F3': self.Mapping(self.TTRef.F3, self.TTRef.Cz), 
                'F4:A1': self.Mapping(self.TTRef.F4, self.TTRef.LPA)
                }

class KIEL(PDRBD_PB_Base):
    
    def dataset_name(self):
        return "PB_KIEL"

    def channel_mapping(self):
        return {'F4:A1': self.Mapping(self.TTRef.F4, self.TTRef.LPA), 
                'EOG2:A1': self.Mapping(self.TTRef.ER, self.TTRef.LPA), 
                'C3': self.Mapping(self.TTRef.C3, self.TTRef.Cz), 
                'A1': self.Mapping(self.TTRef.LPA, self.TTRef.Cz), 
                'O2:A1': self.Mapping(self.TTRef.O2, self.TTRef.LPA), 
                'F4': self.Mapping(self.TTRef.F4, self.TTRef.Cz), 
                'EOG2': self.Mapping(self.TTRef.ER, self.TTRef.Cz), 
                'C4': self.Mapping(self.TTRef.C4, self.TTRef.Cz), 
                'EOG2:A2': self.Mapping(self.TTRef.ER, self.TTRef.RPA), 
                'C4:A1': self.Mapping(self.TTRef.C4, self.TTRef.LPA), 
                'C3:A2': self.Mapping(self.TTRef.C3, self.TTRef.RPA), 
                'O2': self.Mapping(self.TTRef.O2, self.TTRef.Cz), 
                'A2': self.Mapping(self.TTRef.RPA, self.TTRef.Cz), 
                'EOG1:A1': self.Mapping(self.TTRef.EL, self.TTRef.LPA), 
                'EOG1:A2': self.Mapping(self.TTRef.EL, self.TTRef.RPA), 
                'EOG1': self.Mapping(self.TTRef.EL, self.TTRef.Cz)}

def main():
    datapath = "O:/Tech_Neuro247/BIDS/COLOGNE"
    out = "C:/Users/au588953"

    d = COLOGNE(datapath,
                out,
                overwrite_existing=False)
    
    d.port_data()

if __name__ == "__main__":
    main()