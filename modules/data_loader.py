import os 
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchaudio import transforms

class TrainDataset(Dataset):
    def __init__(self, root_path, speakers_file_path, samples_per_epoch=30000, loss_type='constrative'):
        self.root_path = root_path
        self.samples_per_epoch = samples_per_epoch
        self.pos_speaker_to_files, self.all_speakers_to_files = self._read_speaker_dict(root_path, speakers_file_path)
        self.loss_type = loss_type
        
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
    
    def __getitem__(self, index):
        if self.loss_type == 'constrative':
            # random boolean to choose positive pair or negative pair
            pos_or_neg = random.choice([True, False])
            file1, file2, label = self._get_constrative_pair(pos_or_neg)
        elif self.loss_type == 'triplet':
            anchor, positive, negative = self._get_triplet_pair()
        
    def _get_constrative_pair(self, pos_or_neg):
        if pos_or_neg:
            speaker_id = random.choice(list(self.pos_speaker_to_files.keys()))
            file1, file2 = random.sample(self.pos_speaker_to_files[speaker_id], 2)
            label = torch.tensor(1, dtype=torch.long)
        else:
            speakers_id = random.sample(list(self.all_speakers_to_files.keys()), 2)
            file1 = random.choice(self.all_speakers_to_files[speakers_id[0]])
            file2 = random.choice(self.all_speakers_to_files[speakers_id[1]])
            label = torch.tensor(0, dtype=torch.long)
        return file1, file2, label
    
    def _get_triplet_pair(self):
        anchor_id = random.choice(list(self.pos_speaker_to_files.keys()))
        temp_speakers_list = list(self.all_speakers_to_files.keys()).remove(anchor_id)
        negative_id = random.choice(temp_speakers_list)
        anchor, positive = random.sample(self.pos_speaker_to_files[anchor_id], 2)
        negative = random.choice(self.all_speakers_to_files[negative_id])
        return anchor, positive, negative
    
                

