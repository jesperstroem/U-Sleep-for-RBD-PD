'''data set class for sending sleep data to usleep'''

#pylint: disable=invalid-name


#TODO: it's not ideal that all data is loaded before all labels. if there's an issue with a label loading, you get that feedback pretty late. better to go night-by-night


#%% set up dataset

import mne
import torch
import numpy as np
from csdp_pipeline.preprocessing.usleep_prep_steps import scale_channel, filter_channel, remove_dc, resample_channel, clip_channel, FilterSettings

class sleep_dataset_from_paths(torch.utils.data.Dataset):
    '''Dataset class for sending sleep data to U-Sleep. 
    Takes a list of file paths as input, and preprocesses the data according to the settings specified in the constructor. 
    The data is preprocessed to be in line with the preprocessing steps used in U-Sleep, and is returned as a torch tensor. 
    The dataset can be used for training, validation and testing.

    Args:
    EEG_paths: list of file paths to the EEG data. The files must be in a format that can be read by MNE (e.g. .vhdr, .set, .edf).
    ch_names: list of channel names to be used. If None, all channels are used. Defaults to None.
    L: number of epochs to be drawn at a time. Defaults to 1.
    fullRecords: whether to return full records instead of draws of L epochs. Defaults to False.
    fsets: FilterSettings object specifying the filter settings for the dataset. Defaults to a highpass filter with cutoff 0.1 Hz and order 2. If fsets is None, no filtering is applied.'''
    def __init__(self, EEG_paths, ch_names=None, L=1,fullRecords=False, fsets=FilterSettings(lcut=0.1,hcut=None,order=2)):
        self.__constructFromPaths(EEG_paths,L,ch_names,fullRecords, fsets)

    def get_available_channels(filePaths):
        '''Helper function to obtain channel information from the MNE compatible files specified in filePaths'''
        filePaths=[str(fp) for fp in filePaths]
        names = []

        for path in filePaths:
            raw = sleep_dataset_from_paths.open_eeg_file(path, preload=False)
            ch_names = raw.ch_names
            names.append(ch_names)

        return names

    def __constructFromPaths(self, EEG_paths,L, ch_names, fullRecords, fsets):
        self.fsets = fsets
        self.file_paths=[str(fp) for fp in EEG_paths]
        self.L=L
        self.epochLength=128*30 #30 seconds
        self.fullRecords=fullRecords

        #preload data files and preprocess:
        self.data_arrays=[]
        self.nansamples=[]
        for path in self.file_paths:
            tempRaw = sleep_dataset_from_paths.open_eeg_file(path)

            if ch_names is not None:
                data=tempRaw.get_data(picks=ch_names)
            else:
                data=tempRaw.get_data()

            data=self.__preprocess_data(data,sfreq= tempRaw.info['sfreq'])

            self.data_arrays.append(data)

        #just make sure the data has length equal to integer number of epochs:
        for idx,data in enumerate(self.data_arrays):
            nSamples=data.shape[1]
            nSamples=(nSamples//self.epochLength)*self.epochLength
            self.data_arrays[idx]=data[:,:nSamples]

        #extract nansamples again:
        #it has to be done after preprocess_scoring, because that might remove some samples
        self.__extract_nansamples()

        #create data draws - assumes we will draw L epochs at a time:
        self.__create_data_draws()

        #send everything to torch tensors:
        self.data_arrays=[torch.tensor(data, dtype=torch.float32)
                          for data in self.data_arrays]

    def __extract_nansamples(self):
        '''Extracts 'nansamples' from the data arrays to bring them back
        to correct size, and keep track of all-nan epochs. If data is not integer
        number of epochs, the trailing samples are ignored.'''
        nansamples_list=[]
        for idx,data in enumerate(self.data_arrays):
            nDeriv=data.shape[0]//2
            assert nDeriv*2==data.shape[0]
            self.data_arrays[idx]=data[0:nDeriv,:]
            nansamples_list.append(data[nDeriv:,:])

        #determine all-nan epochs based on nanvals:
        self.nanEpochs=[]
        for nansamples in nansamples_list:
            nansamples=nansamples>0
            nEpochs=nansamples.shape[1]//self.epochLength
            nanEpochs=np.zeros((nansamples.shape[0],nEpochs),dtype=bool)
            for iChannel in range(nansamples.shape[0]):

                nanEpochs[iChannel,:]=np.all(nansamples[iChannel,:(nEpochs*self.epochLength)].reshape(self.epochLength,-1),axis=0).reshape(1,-1)

            self.nanEpochs.append(nanEpochs)


    def __create_data_draws(self):
        '''Creates a list of indices for drawing data from the dataset.
        Allows __getitem__ to ignore epochlength and number of files'''
        self.dataDraws=[]
        for file_idx,_ in enumerate(self.data_arrays):
            nSamples=self.data_arrays[file_idx].shape[1]
            for i in range(0,nSamples-self.L*self.epochLength,self.epochLength):
                self.dataDraws.append([file_idx,i])

        self.dataDraws=np.array(self.dataDraws)

    def __preprocess_data(self,data,sfreq):
        '''Preprocess data to be in line with usleep'''

        output_data=[]
        output_nansamples=[]

        for i in range(data.shape[0]):
            channel_data=data[i,:]

            nansamples=np.isnan(channel_data)
            channel_data[nansamples]=0

            channel_data = remove_dc(channel_data)

            #resampling both data and nansamples:
            channel_data = resample_channel(channel_data,
                                    output_rate=128,
                                    source_sample_rate=sfreq)
            nansamples=resample_channel(nansamples.astype(float),
                                        output_rate=128,
                                    source_sample_rate=sfreq)>.5

            if self.fsets != None:
                channel_data = filter_channel(channel_data, 128, self.fsets)

            channel_data = scale_channel(channel_data)
            channel_data = clip_channel(channel_data)

            output_data.append(channel_data)
            output_nansamples.append(nansamples)

        output_data=np.array(output_data)
        output_nansamples=np.array(output_nansamples)

        #return data with nansamples. nansamples are removed again later:
        return np.vstack((output_data,output_nansamples))

    def open_eeg_file(filename, preload=True):
        '''Opens data files. Supports .set, .edf and .vhdr files. Returns an MNE Raw object.'''
        if filename.endswith('.set'):
            return mne.io.read_raw_eeglab(filename,preload=preload,verbose=False)
        elif filename.endswith('.edf'):
            return mne.io.read_raw_edf(filename,preload=preload,verbose=False)
        elif filename.endswith(".vhdr"):
            return mne.io.read_raw_brainvision(filename, preload=preload, verbose=False)
        else:
            print('Unknown file type for file ' + filename)
            raise ValueError('Unknown file type')

    def __len__(self):
        if self.fullRecords:
            return len(self.data_arrays)
        else:
            return self.dataDraws.shape[0]

    def get_minibatch(self, idx):
        '''Returns a minibatch of data. Used for training.'''
        fileIdx=self.dataDraws[idx,0]
        sampleIdx=self.dataDraws[idx,1]
        epochIdx=sampleIdx//self.epochLength

        x = self.data_arrays[fileIdx][:,sampleIdx:sampleIdx+self.epochLength*self.L]

        assert x.shape[1]==self.epochLength*self.L

        return x,{'dataSet':0,'file':fileIdx,'sample':sampleIdx}

    def get_full_record(self, fileIdx):
        '''Returns a full record of data. Used for validation and testing.'''
        x = self.data_arrays[fileIdx]
        
        return x,fileIdx

    def __getitem__(self, idx):
        if self.fullRecords:
            return self.get_full_record(idx)
        else:
            return self.get_minibatch(idx)