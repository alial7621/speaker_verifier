import os
import random

import pandas as pd
import torch
import torchaudio
from audiomentations import AddGaussianNoise, Compose, Gain
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms


class TrainDataset(Dataset):
    def __init__(self, root_path, speakers_file_path, samples_per_epoch=30000, loss_type='contrastive',
                 sample_rate=16000, duration=3, vad=False, augmentations=False, mfcc_feat_dim=80):
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
            the type of loss to use, including 'contrastive', 'triplet', or 'cosineemb'
        sample_rate : int, optional
            sample rate of audio files
        duration : int, optional
            duration of audio files in seconds
        vad : bool, optional
            whether to use voice activity detection on audio files
        augmentations : bool, optional
            whether to use augmentations on audio files
        mfcc_feat_dim : int, optional
            number of mel frequency cepstral coefficients to use

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
        if augmentations:
            self.augmentations = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                Gain(p=0.5)
            ])
            self.spec_aug = torch.nn.Sequential(
                transforms.FrequencyMasking(freq_mask_param=10),
                transforms.TimeMasking(time_mask_param=5)
            )
        else:
            self.augmentations = None

        self.mfcc_transform = transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=mfcc_feat_dim,
            melkwargs={
                'n_fft': 512,
                'win_length': int(self.sample_rate * 0.025),
                'hop_length': int(self.sample_rate * 0.010),
                'n_mels': mfcc_feat_dim
            }
        )
        
        assert self.loss_type in ['contrastive', 'triplet', 'cosineemb']

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
        data_path = os.path.join(root_path, 'data')

        # Collect audio file paths for each speaker
        for speaker_id in speaker_ids:
            speaker_dir = os.path.join(data_path, speaker_id)
            if os.path.isdir(speaker_dir):
                files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith('.mp3')]
                if len(files) >= 2:
                    pos_speaker_to_files[speaker_id] = files
                all_speakers_to_files[speaker_id] = files

        return pos_speaker_to_files, all_speakers_to_files
    
    def __len__(self):
        return self.samples_per_epoch
    
    def _preprocess(self, path):
        """
        Preprocess the audio file.

        Parameters
        ----------
        path : str
            path to the audio file

        Returns
        -------
        torch.Tensor
            preprocessed audio file
        """
        waveform, sr = torchaudio.load(path)
        # resample if necessary
        if sr != self.sample_rate:
            resampler = transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # apply voice activity detection if necessary
        if self.vad:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, [['vad', '-t', '3'], ['rate', str(self.sample_rate)]])
        else:
            waveform = waveform / waveform.abs().max()

        # pad or truncate if necessary
        if waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            pad_len = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # apply augmentations if necessary
        if self.augmentations is not None:
            waveform = self.augmentations(samples=waveform.numpy(), sample_rate=self.sample_rate)
            waveform = torch.tensor(waveform)
        
        # compute MFCC
        mfcc = self.mfcc_transform(waveform)

        # Cepstral Mean Normalization (per feature channel)
        mfcc = mfcc - mfcc.mean(dim=-1, keepdim=True)

        # apply spec augmentations if necessary
        if self.augmentations is not None:
            mfcc = self.spec_aug(mfcc)

        return mfcc
    
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters
        ----------
        index : int
            index of the item to get (Do not use this parameter)

        Returns
        -------
        tuple
            audio file 1, audio file 2, label
        """
        if self.loss_type == 'contrastive' or self.loss_type == 'cosineemb':
            # random boolean to choose positive pair or negative pair
            pos_or_neg = random.choice([True, False])
            file1, file2, label = self._get_contrastive_pair(pos_or_neg)
            
            return self._preprocess(file1), self._preprocess(file2), label
        
        elif self.loss_type == 'triplet':
            anchor, positive, negative = self._get_triplet_pair()
            anchor = self._preprocess(anchor)
            positive = self._preprocess(positive)
            negative = self._preprocess(negative)
            return anchor, positive, negative
        
    def _get_contrastive_pair(self, pos_or_neg):
        """
        Get a contrastive pair.

        Parameters
        ----------
        pos_or_neg : bool
            whether to get a positive or negative pair

        Returns
        -------
        tuple
            audio file 1, audio file 2, label
        """
        if pos_or_neg:
            # choose a speaker for positive pair
            speaker_id = random.choice(list(self.pos_speaker_to_files.keys()))
            file1, file2 = random.sample(self.pos_speaker_to_files[speaker_id], 2)
            label = torch.tensor(1, dtype=torch.float)
        else:
            # choose two speakers for negative pair
            speakers_id = random.sample(list(self.all_speakers_to_files.keys()), 2)
            file1 = random.choice(self.all_speakers_to_files[speakers_id[0]])
            file2 = random.choice(self.all_speakers_to_files[speakers_id[1]])
            label = torch.tensor(0, dtype=torch.float)
        return file1, file2, label
    
    def _get_triplet_pair(self):
        """
        Get a triplet pair.

        Returns
        -------
        tuple
            anchor, positive, negative
        """
        # choose anchor speaker
        anchor_id = random.choice(list(self.pos_speaker_to_files.keys()))

        # choose negative speaker
        temp_speakers_list = list(self.all_speakers_to_files.keys()).remove(anchor_id)
        negative_id = random.choice(temp_speakers_list)
        
        # choose positive and negative files
        anchor, positive = random.sample(self.pos_speaker_to_files[anchor_id], 2)
        negative = random.choice(self.all_speakers_to_files[negative_id])
        return anchor, positive, negative
    

class ValidDataset(Dataset):
    def __init__(self, dataset_path, sample_rate=16000, duration=3, vad=False, mfcc_feat_dim=80):
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
        mfcc_feat_dim : int, optional
            number of mel frequency cepstral coefficients to use
        """
        self.dataset_df = pd.read_csv(dataset_path)
        self.sample_rate = sample_rate
        self.fixed_length = int(sample_rate * duration)
        self.vad = vad

        self.mfcc_transform = transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=mfcc_feat_dim,
            melkwargs={
                'n_fft': 512,
                'win_length': int(self.sample_rate * 0.025),
                'hop_length': int(self.sample_rate * 0.010),
                'n_mels': mfcc_feat_dim
            }
        )

    def __len__(self):
        return len(self.dataset_df)
    
    def _preprocess(self, path):
        """
        Preprocess the audio file.

        Parameters
        ----------
        path : str
            path to the audio file

        Returns
        -------
        torch.Tensor
            preprocessed audio file
        """
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

        mfcc = self.mfcc_transform(waveform)

        # Cepstral Mean Normalization (per feature channel)
        mfcc = mfcc - mfcc.mean(dim=-1, keepdim=True)

        return mfcc
    
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters
        ----------
        index : int
            index of the item to get

        Returns
        -------
        tuple
            audio file 1, audio file 2, label
        """
        row = self.dataset_df.iloc[index]
        waveform1 = self._preprocess(row['audio_path_1'])
        waveform2 = self._preprocess(row['audio_path_2'])
        label = torch.tensor(row['label'], dtype=torch.int)

        return waveform1, waveform2, label
    
def get_data_loaders(root_path, samples_per_epoch=30000, loss_type='contrastive',
                     sample_rate=16000, duration=3, vad=False, augmentations=False,
                     batch_size=32, num_workers=4, mfcc_feat_dim=80):
    """
    Create data loaders for training and validation.

    Parameters
    ----------
    root_path : str
        path to the root directory
    samples_per_epoch : int, optional
        number of samples to draw from the dataset per epoch
    loss_type : str, optional
        the type of loss to use, including 'contrastive', 'triplet', or 'cosineemb'
    sample_rate : int, optional
        sample rate of audio files
    duration : int, optional
        duration of audio files in seconds
    vad : bool, optional
        whether to use voice activity detection on audio files
    augmentations : bool, optional
        whether to use augmentations on audio files
    mfcc_feat_dim : int, optional
        number of mel frequency cepstral coefficients to use
    batch_size : int, optional
        batch size for training and validation
    num_workers : int, optional
        number of workers for training and validation

    Returns
    -------
    tuple
        train_loader and valid_loader   
    """
    # Create datasets
    train_dataset = TrainDataset(
        root_path=root_path,
        speakers_file_path=os.path.join(root_path, "train_speakers.txt"),
        samples_per_epoch=samples_per_epoch,
        loss_type=loss_type,
        sample_rate=sample_rate,
        duration=duration,
        vad=vad, 
        augmentations=augmentations,
        mfcc_feat_dim=mfcc_feat_dim
    )
        
    valid_dataset = ValidDataset(
        dataset_path=os.path.join(root_path, "validation.csv"),
        sample_rate=sample_rate, 
        duration=duration, 
        vad=vad,
        mfcc_feat_dim=mfcc_feat_dim
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader                

