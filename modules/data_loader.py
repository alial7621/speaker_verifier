import os 
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from torchaudio import transforms

class TrainDataset(Dataset):
    def __init__(self, root_path, speakers_file_path, samples_per_epoch=30000, loss_type='constrative',
                 sample_rate=16000, duration=3, vad=False):
        """
        Constructor for TrainDataset.

        Parameters
        ----------
        root_path : str
            root directory where speaker folders are located
        speakers_file_path : str
            path to a text file containing speaker IDs
        samples_per_epoch : int, optional
            number of samples to draw from the dataset per epoch
        loss_type : str, optional
            the type of loss to use, either 'constrative' or 'triplet'
        sample_rate : int, optional
            sample rate of audio files
        duration : int, optional
            duration of audio files in seconds
        vad : bool, optional
            whether to use voice activity detection on audio files

        Returns
        -------
        None
        """
        self.root_path = root_path
        self.samples_per_epoch = samples_per_epoch
        self.pos_speaker_to_files, self.all_speakers_to_files = self._read_speaker_dict(root_path, speakers_file_path)
        self.loss_type = loss_type
        self.sample_rate = sample_rate
        self.fixed_length = int(sample_rate * duration)
        self.vad = vad
        
        assert self.loss_type in ['constrative', 'triplet']

    def _read_speaker_dict(self,root_path,file_path):
        """
        Reads a text file containing speaker IDs and creates two dictionaries of speaker IDs to their respective file paths.
        The first dictionary only contains speakers with at least 2 files, used for positive pairs.
        The second dictionary contains all speakers, used for negative pairs.

        Parameters
        ----------
        root_path : str
            root directory where speaker folders are located
        file_path : str
            path to a text file containing speaker IDs

        Returns
        -------
        tuple
            two dictionaries, one with speakers having at least 2 files, the other with all speakers
        """
        with open(file_path, 'r') as f:
            speaker_ids = [line.strip() for line in f]
        
        # one for just positive pairs, one for negative pairs
        pos_speaker_to_files = {}
        all_speakers_to_files = {}

        # Collect audio file paths for each speaker
        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(root_path, speaker_id)
            if os.path.isdir(speaker_dir):
                files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith('.mp3')]
                if len(files) >= 2:
                    pos_speaker_to_files[speaker_id] = files
                all_speakers_to_files[speaker_id] = files

        return pos_speaker_to_files, all_speakers_to_files
    
    def __len__(self):
        return self.samples_per_epoch
    
    def _preprocess(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.vad:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, [['vad', '-t', '3'], ['rate', str(self.sample_rate)]])
        else:
            waveform = waveform / waveform.abs().max()

        if waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            pad_len = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform
    
    def __getitem__(self, index):
        if self.loss_type == 'constrative':
            # random boolean to choose positive pair or negative pair
            pos_or_neg = random.choice([True, False])
            file1, file2, label = self._get_constrative_pair(pos_or_neg)
            
            return self._preprocess(file1), self._preprocess(file2), label
        
        elif self.loss_type == 'triplet':
            anchor, positive, negative = self._get_triplet_pair()
            
            return self._preprocess(anchor), self._preprocess(positive), self._preprocess(negative)
        
    def _get_constrative_pair(self, pos_or_neg):
        if pos_or_neg:
            speaker_id = random.choice(list(self.pos_speaker_to_files.keys()))
            file1, file2 = random.sample(self.pos_speaker_to_files[speaker_id], 2)
            label = torch.tensor(1, dtype=torch.float)
        else:
            speakers_id = random.sample(list(self.all_speakers_to_files.keys()), 2)
            file1 = random.choice(self.all_speakers_to_files[speakers_id[0]])
            file2 = random.choice(self.all_speakers_to_files[speakers_id[1]])
            label = torch.tensor(0, dtype=torch.float)
        return file1, file2, label
    
    def _get_triplet_pair(self):
        anchor_id = random.choice(list(self.pos_speaker_to_files.keys()))
        temp_speakers_list = list(self.all_speakers_to_files.keys()).remove(anchor_id)
        negative_id = random.choice(temp_speakers_list)
        anchor, positive = random.sample(self.pos_speaker_to_files[anchor_id], 2)
        negative = random.choice(self.all_speakers_to_files[negative_id])
        return anchor, positive, negative
    

class ValidDataset(Dataset):
    def __init__(self, dataset_path, sample_rate=16000, duration=3, vad=False):
        """   
        Parameters
        ----------
        dataset_path : str
            path to a CSV file containing columns 'audio_path_1', 'audio_path_2', 'label'
        sample_rate : int, optional
            sample rate of audio files
        duration : int, optional
            duration of audio files in seconds
        vad : bool, optional
            whether to use voice activity detection on audio files
        """
        self.dataset_df = pd.read_csv(dataset_path)
        self.sample_rate = sample_rate
        self.fixed_length = int(sample_rate * duration)
        self.vad = vad

    def __len__(self):
        return len(self.dataset_df)
    
    def _preprocess(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.vad:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, [['vad', '-t', '3'], ['rate', str(self.sample_rate)]])
        else:
            waveform = waveform / waveform.abs().max()

        if waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            pad_len = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform
    
    def __getitem__(self, index):
        row = self.dataset_df.iloc[index]
        waveform1 = self._preprocess(row['audio_path_1'])
        waveform2 = self._preprocess(row['audio_path_2'])
        label = torch.tensor(row['label'], dtype=torch.float)

        return waveform1, waveform2, label
    
                

