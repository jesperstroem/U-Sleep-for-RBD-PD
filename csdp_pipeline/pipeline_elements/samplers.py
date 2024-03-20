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
from abc import ABC, abstractmethod
from csdp_pipeline.pipeline_elements.models import ISample, ITag, Split, Dataset_Split

class ISampler:
    @abstractmethod
    def get_sample(self, index) -> ISample:
        pass

    num_samples: int

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

class Random_Sampler(ISampler):
    def __init__(self,
                 split_data: Split,
                 split_type: str, 
                 num_epochs: int, 
                 num_iterations: int,
                 get_all_channels: bool = False,
                 subject_percentage = 1):
        self.split_type = split_type
        self.split_data = split_data
        self.get_all_channels = get_all_channels
        self.subject_percentage = subject_percentage
        self.subjects, self.num_records = self.__list_files()

        print(f"Number of {split_type} subjects: {len(self.subjects)} records: {self.num_records} - subject percentage: {subject_percentage}")

        try:
            self.probs = self.calc_probs()
        except:
            self.probs = []
            
        self.epoch_length = num_epochs
        self.num_samples = num_iterations
        
    def get_sample(self, index: int) -> ISample:
        success = False
            
        while not success:
            sample: ISample = self.__get_sample()

            if sample != None:
                success = True

        return sample
    
    def calc_probs(self):
        total_num_datasets = len(self.split_data.dataset_splits)
        total_num_records = sum(self.num_records)
        
        probs = []
        
        for i, _ in enumerate(self.split_data.dataset_splits):
            num_records = self.num_records[i]
            
            strat_prob = num_records/total_num_records
            dis_prob = 1 / total_num_datasets
            
            prob_d = 0.5 * strat_prob + 0.5*dis_prob
            probs.append(prob_d)

        return probs
    
    def __get_sample(self) -> ISample:
        
        possible_sets = self.split_data.dataset_splits
        probs = self.probs
        
         # Choose random dataset
        r_dataset: Dataset_Split = np.random.choice(possible_sets, 1, p=probs)[0]

        subjects = r_dataset.get_subjects_from_string(self.split_type)

        r_subject = np.random.choice(subjects, 1)[0]

        if len(subjects) == 0:
            raise ValueError(f"No subjects in split type: {self.split_type} for dataset {r_dataset}")

        with h5py.File(r_dataset.dataset_filepath, "r") as hdf5:

            # Choose random subject
            records = list(hdf5[r_subject].keys())

            #choose Random record
            r_record = np.random.choice(records, 1)[0]

            hyp = hdf5[r_subject][r_record]["hypnogram"][()]
            psg = list(hdf5[r_subject][r_record]["psg"].keys())

            if self.get_all_channels == True:
                eegs = self.__pick_all(psg, "EEG")
                eogs = self.__pick_all(psg, "EOG")
            else:
                eegs = self.__pick_random_channel(psg, "EEG")
                eogs = self.__pick_random_channel(psg, "EOG")

            # Choose random index of a random label
            label_set = np.unique(hyp)

            r_label = np.random.choice(label_set, 1)[0]

            indexes = [i for i in range(len(hyp)) if hyp[i] == r_label]
            r_index = np.random.choice(indexes, 1)[0]
            
            # Randomly shift the position of the random label index
            r_shift = np.random.choice(list(range(0,self.epoch_length)), 1)[0]
            
            assert r_shift <= 200
            
            start_index = r_index-r_shift
            
            if start_index < 0:
                start_index = 0
            elif (start_index + self.epoch_length) >= len(hyp):
                start_index = len(hyp) - self.epoch_length
            
            y = hyp[start_index:start_index+self.epoch_length]
            
            y = torch.tensor(y)
            
            x_start_index = start_index*128*30

            eeg_segments = []
            eog_segments = []

            for eeg in eegs:
                try:
                    eeg_segment = hdf5[r_subject][r_record]["psg"][eeg][x_start_index:x_start_index+(self.epoch_length*30*128)]
                except:
                    eeg_segment = []
                eeg_segments.append(eeg_segment)

            for eog in eogs:
                try:
                    eog_segment = hdf5[r_subject][r_record]["psg"][eog][x_start_index:x_start_index+(self.epoch_length*30*128)]
                except:
                    eog_segment = []

                eog_segments.append(eog_segment)

        eeg_segments = np.array(eeg_segments)
        eog_segments = np.array(eog_segments)

        x_eeg = torch.tensor(eeg_segments)
        x_eog = torch.tensor(eog_segments)

        sample = ISample(-1)
        sample.eeg = x_eeg
        sample.eog = x_eog
        sample.labels = y
        sample.tag = ITag(os.path.basename(r_dataset.dataset_filepath),
                          r_subject,
                          r_record,
                          [],
                          [],
                          x_start_index,
                          x_start_index+(self.epoch_length*30*128))
        
        return sample

    def __pick_random_channel(self, channel_list, type):
        #Choose random eeg and eog
        channels = [x for x in channel_list if x.startswith(type)]

        r_channel = np.random.choice(channels, 1)
                
        return r_channel
    
    def __pick_all(self, channel_list, type):
        #Choose random eeg and eog
        channels = [x for x in channel_list if x.startswith(type)]
                
        return channels

    def __list_files(self):
        subjects = []
        num_records = []
        
        for f in self.split_data.dataset_splits:
            file_path = f.dataset_filepath

            with h5py.File(file_path, "r") as hdf5:
                subs = f.get_subjects_from_string(self.split_type)
                    
                num_subjects = len(subs)
                num_subjects_to_use = math.ceil(num_subjects*self.subject_percentage)
                subs = subs[0:num_subjects_to_use]

                tot_records = 0
                subjects_to_add = []
                
                for subj_key in subs:
                    try:
                        subj = hdf5[subj_key]
                    except:
                        print(f"Did not find subject {subj_key} in dataset {f} for splittype {self.split_type}")
                        continue
                        
                    records = len(subj.keys())
                    tot_records += records
                    subjects_to_add.append(subj_key)
                
                num_records.append(tot_records)
                subjects.append(subjects_to_add)

        return subjects, num_records


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

        if any(dim == 0 for dim in sample.eog.shape):
            print("Found no EOG channel, duplicating EEG instead")
            sample.eog = sample.eeg

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
            eeg_tag = available_eeg_keys
            eog_tag = available_eog_keys

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
    
