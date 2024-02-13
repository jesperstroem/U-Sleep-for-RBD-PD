# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:25:31 2023

@author: Jesper Strøm
"""

import torch
import os
import h5py
import math
import numpy as np

from csdp_pipeline.pipeline_elements.pipe import IPipe, ISample, ITag, ISampler, Split, Dataset_Split

class Determ_sampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str,
                 subject_percentage: float = 1.0,
                 get_all_channels = False):
        self.split_type = split_type
        self.split_data = split_data
        self.subject_percentage = subject_percentage
        self.records = self.list_records()
        self.num_samples = len(self.records)
        self.get_all_channels = get_all_channels

    def get_sample(self, index: int):
        sample: ISample = self.__get_sample(index)

        # if any(dim == 0 for dim in sample.eog.shape):
        #     print("Found no EOG channel, duplicating EEG instead")
        #     sample.eog = sample.eeg

        return sample

    def list_records(self):
        list_of_records = []
        datasets = self.split_data.dataset_splits

        for f in datasets:
            with h5py.File(f.dataset_filepath, "r") as hdf5:

                subjects = f.get_subjects_from_string(self.split_type)
                
                num_subjects = len(subjects)
                num_subjects_to_use = math.ceil(num_subjects*self.subject_percentage)
                subjects = subjects[0:num_subjects_to_use]
                
                for s in subjects:
                    try:
                        records = list(hdf5[s])
                    except:
                        print(f"Did not find subject {s} in dataset {f} for splittype {self.split_type}")
                        continue

                    for r in records:
                        list_of_records.append((f.dataset_filepath,s,r))
        
        return list_of_records

    def __get_sample(self, index: int) -> ISample:
        r = self.records[index]

        dataset = r[0]
        subject = r[1]
        rec = r[2]

        with h5py.File(dataset, "r") as hdf5:
            y = hdf5[subject][rec]["hypnogram"][()]

            psg_channels = list(hdf5[subject][rec]["psg"].keys())

            eeg_data, eog_data, eeg_tag, eog_tag = self.__load_data(hdf5, subject, rec, psg_channels)

        sample = ISample(index)
        sample.eeg = eeg_data
        sample.eog = eog_data
        sample.labels = torch.tensor(y)
        sample.tag = ITag(os.path.basename(dataset),
                          subject,
                          rec,
                          eeg_tag,
                          eog_tag)

        return sample
    
    def determine_single_key(self, keys):
        if len(keys) > 0:
            key = keys[0]
            tag = key
            keys = [key]
        else:
            tag = "none"
        
        return keys, tag

    def __load_data(self, hdf5, subject, rec, psg_channels):
        eeg_data = []
        eog_data = []

        available_eeg_keys = [x for x in psg_channels if x.startswith("EEG")]
        available_eog_keys = [x for x in psg_channels if x.startswith("EOG")]

        if self.get_all_channels == False:
            eeg_keys, eeg_tag = self.determine_single_key(available_eeg_keys)
            eog_keys, eog_tag = self.determine_single_key(available_eog_keys)
        else:
            eeg_keys = available_eeg_keys
            eog_keys = available_eog_keys
            eeg_tag = "all"
            eog_tag = "all"

        for ch in eeg_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eeg_data.append(data)
            
        for ch in eog_keys:
            data = hdf5[subject][rec]["psg"][ch][:]
            eog_data.append(data)

        eog_data = np.array(eog_data)
        eog_data = torch.Tensor(eog_data)

        eeg_data = np.array(eeg_data)
        eeg_data = torch.Tensor(eeg_data)
        
        return eeg_data, eog_data, eeg_tag, eog_tag
    
