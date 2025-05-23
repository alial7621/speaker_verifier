import os

import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
import torch
import torchaudio
import torch.nn as nn
from torchaudio import transforms
from torchmetrics.classification import BinaryEER

from modules.model import ECAPA_TDNN

class SpeakerVerification():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.fixed_length = int(config.sample_rate * config.duration)

        # Load the model
        self.model_load()
        
        self.mfcc_transform = transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.mfcc_feat_dim,
            melkwargs={
                'n_fft': 512,
                'win_length': int(config.sample_rate * 0.025),
                'hop_length': int(config.sample_rate * 0.010),
                'n_mels': config.mfcc_feat_dim
            }
        )

    def model_load(self):
        # Initialize Model
        self.model = ECAPA_TDNN(
            in_channels=self.config.mfcc_feat_dim,
            channels=self.config.channels,
            embd_dim=self.config.embd_dim
        ).to(self.device)

        # Load the checkpoint
        if os.path.isfile(self.config.checkpoint):
            self.model.load_state_dict(torch.load(self.config.checkpoint, map_location=self.device))
        else:
            raise ValueError("Trained model should be provided")
        
        self.model.eval()

    def get_spk_emb(self, audio_file):
        audio_feat = self._preprocess(audio_file)
        with torch.no_grad():
            return self.model(audio_feat)

    def verify(self, audio_file1, audio_file2):
        audio_emb1 = self.get_spk_emb(audio_file1)
        audio_emb2 = self.get_spk_emb(audio_file2)
        
        # Calculate similarity
        similarity = nn.functional.cosine_similarity(audio_emb1, audio_emb2)

        if self.config.eer_thresh is None:
            print("Threshold has not been specified, using default: 0.7")
            self.config.eer_thresh = 0.7

        if similarity > self.config.eer_thresh:
            print("Verification Confirmed")
        else:
            print("Verification Denied")

    def _preprocess(self, audio_file):
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
        waveform, sr = torchaudio.load(audio_file)
        if sr != self.config.sample_rate:
            resampler = transforms.Resample(orig_freq=sr, new_freq=self.config.sample_rate)
            waveform = resampler(waveform)

        if self.config.vad:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.config.sample_rate, [['vad', '-t', '3'], ['rate', str(self.config.sample_rate)]])
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


class VerifierTester:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.optimal_threshold = None

        # Initialize Model
        self.model = ECAPA_TDNN(
            in_channels=config.mfcc_feat_dim,
            channels=config.channels,
            embd_dim=config.embd_dim
        ).to(device)

        # Load the checkpoint
        if os.path.isfile(config.checkpoint):
            self.model.load_state_dict(torch.load(config.checkpoint, map_location=device))
        else:
            raise ValueError("Trained model should be provided")
        
        self.model.eval()

    def validate(self, val_loader):

        all_similarities = []
        all_labels = []
        
        for input_data in tqdm(val_loader):
            # Get data
            # input_data shape: [batch_size, 1, feat_dim, seq_len]
            # model input shape: [batch_size, seq_len, feat_dim]
            audio_batch1 = (input_data[0].squeeze(1).permute(0, 2, 1)).to(self.device)
            audio_batch2 = (input_data[1].squeeze(1).permute(0, 2, 1)).to(self.device)
            labels = input_data[2].to(self.device)

            with torch.no_grad():
                # Get model output
                audio_outputs1 = self.model(audio_batch1)
                audio_outputs2 = self.model(audio_batch2)

            # Calculate similarity
            similarity = nn.functional.cosine_similarity(audio_outputs1, audio_outputs2)
            
            # Collect all similarities and labels for EER calculation
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # Calculate EER and find optimal threshold
        self._calculate_eer_with_threshold(all_similarities, all_labels)        

    def _calculate_eer_with_threshold(self, similarities, labels):
        """
        Calculate Equal Error Rate and find the optimal threshold.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of true labels (1 for same speaker, 0 for different speakers)
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        
        # Calculate False Negative Rate (FNR = 1 - TPR)
        fnr = 1 - tpr
        
        # Find the point where FPR and FNR are closest (EER point)
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        optimal_threshold = thresholds[eer_index]
        
        self.optimal_threshold = optimal_threshold
        print(f"Validation set EER: {eer}")

    def test(self, test_loader):
        """
        Validate the speaker verifier model with the testset.
        
        Args:
            test_loader: DataLoader for validation data
        """

        test_eer = 0
        
        # Initialize metric
        binary_eer = BinaryEER(thresholds=self.optimal_threshold)
        
        for input_data in tqdm(test_loader):

            # Get data
            # input_data shape: [batch_size, 1, feat_dim, seq_len]
            # model input shape: [batch_size, seq_len, feat_dim]
            audio_batch1 = (input_data[0].squeeze(1).permute(0, 2, 1)).to(self.device)
            audio_batch2 = (input_data[1].squeeze(1).permute(0, 2, 1)).to(self.device)
            labels = input_data[2].to(self.device)

            # Get model output
            with torch.no_grad():
                audio_outputs1 = self.model(audio_batch1)
                audio_outputs2 = self.model(audio_batch2)

            # calculate similarity
            similarity = nn.functional.cosine_similarity(audio_outputs1, audio_outputs2)

            # calculate eer
            test_eer += binary_eer(similarity, labels)

        # Record metrics and save the model in case of the best result
        test_eer = test_eer / len(test_loader)

        print(f"Testset EER: {test_eer}")
