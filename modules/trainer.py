import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryEER
from tqdm import tqdm

from modules.model import ECAPA_TDNN


class VerifierTrainer:
    """
    Class for training a speaker verification model.
    Handles the training process, checkpointing, and logging.
    """
    def __init__(self, config, device):
        """
        Initialize the VerifierTrainer.

        Args:
            config: configuration (input arguments)
            device: device (cpu or gpu)
        """
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.epochs = config.epochs
        self.best_eer = 1.01

        # Initialize metrics
        self.metrics = {
            "eer": [],
            "loss": []
        }

        # Initialize Model
        self.model = ECAPA_TDNN(
            in_channels=config.mfcc_feat_dim,
            channels=config.channels,
            embd_dim=config.embd_dim
        ).to(device)
        self._num_parameters()

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.momentum1, self.config.momentum2)
        )

        # Initialize Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", 
                                                              factor=config.weight_decay_factor,
                                                              patience=config.lr_wait, 
                                                              threshold=config.lr_thresh,
                                                              threshold_mode="abs",
                                                              min_lr=config.final_lr,
                                                              verbose=True)  
        
        # Initialize metric
        self.eer = BinaryEER(thresholds=config.eer_thresh)

        # Initialize loss function
        if config.loss_type == 'triplet':
            self.loss_func = nn.TripletMarginLoss(margin=config.margin_loss)
        elif config.loss_type == 'cosineemb':
            self.loss_func = nn.CosineEmbeddingLoss(margin=config.margin_loss)
        elif config.loss_type == 'contrastive':
            self.loss_func = self._contrastive_loss
        else:
            raise ValueError("Invalid input provided")
        
    def train(self, train_loader, val_loader=None): 
        """
        Train the speaker verifier model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
        """
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            self._train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                self._validate(val_loader)
            
            # Update learning rates
            self.scheduler.step(self.best_eer)
            
            # Save checkpoint
            self._save_checkpoint(mode='last')
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Time: {elapsed:.2f}s - "
                  f"Loss: {self.metrics['loss'][-1]:.4f} - "
                  f"EER: {self.metrics['eer'][-1]:.4f}\n")
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def _train_epoch(self, train_loader):
        """
        Train the speaker verifier model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        """
        
        self.model.train()
        
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.epochs}")
        for input_data in pbar:
            self.optimizer.zero_grad()
            if self.config.loss_type == 'triplet':
                # Get data
                # input_data shape: [batch_size, 1, feat_dim, seq_len]
                # model input shape: [batch_size, seq_len, feat_dim]
                anchor_batch = (input_data[0].squeeze(1).permute(0, 2, 1)).to(self.device)
                positive_batch = (input_data[1].squeeze(1).permute(0, 2, 1)).to(self.device)
                negative_batch = (input_data[2].squeeze(1).permute(0, 2, 1)).to(self.device)

                # Get model output
                anchor_output = self.model(anchor_batch)
                positive_output = self.model(positive_batch)
                negative_output = self.model(negative_batch)

                # Calcualte loss
                loss = self.loss_func(anchor_output, positive_output, negative_output)
            
            else:
                # Get data
                # input_data shape: [batch_size, 1, feat_dim, seq_len]
                # model input shape: [batch_size, seq_len, feat_dim]
                audio_batch1 = (input_data[0].squeeze(1).permute(0, 2, 1)).to(self.device)
                audio_batch2 = (input_data[1].squeeze(1).permute(0, 2, 1)).to(self.device)
                labels = input_data[2].to(self.device)

                # Get model output
                audio_outputs1 = self.model(audio_batch1)
                audio_outputs2 = self.model(audio_batch2)

                # Calcualte loss
                loss = self.loss_func(audio_outputs1, audio_outputs2, labels)

            
            # train the generator model
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item()
            })

        # Record metrics
        self.metrics['loss'].append(epoch_loss / len(train_loader))

    def _validate(self, val_loader):        
        """
        Validate the speaker verifier model.
        
        Args:
            val_loader: DataLoader for validation data
        """

        self.model.eval()
        epoch_eer = 0
        
        for input_data in tqdm(val_loader):

            # Get data
            # input_data shape: [batch_size, 1, feat_dim, seq_len]
            # model input shape: [batch_size, seq_len, feat_dim]
            audio_batch1 = (input_data[0].squeeze(1).permute(0, 2, 1)).to(self.device)
            audio_batch2 = (input_data[1].squeeze(1).permute(0, 2, 1)).to(self.device)
            labels = input_data[2].to(self.device)

            # Get model output
            audio_outputs1 = self.model(audio_batch1)
            audio_outputs2 = self.model(audio_batch2)

            # calculate similarity
            similarity = nn.functional.cosine_similarity(audio_outputs1, audio_outputs2)

            # calculate eer
            epoch_eer += self.eer(similarity, labels)

        # Record metrics and save the model in case of the best result
        epoch_eer = epoch_eer / len(val_loader)
        self.metrics['eer'].append(epoch_eer)
        if self.best_eer > epoch_eer:
            self._save_checkpoint(mode='best')

    def _save_checkpoint(self, mode='last'):
        """ 
        Save a checkpoint of the models.

        Args:
            mode: last or best
        """

        checkpoint_path = os.path.join(self.config.checkpoint_dir, "models")
        if mode == 'last':
            state = {
                "epoch": self.current_epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "metrics": self.metrics,
            }
            torch.save(state, os.path.join(checkpoint_path, "last.pt"))
        elif mode == 'best':
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "best.pt"))

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint of the models.

        Args:
            checkpoint_path: path to the checkpoint file
        """

        print("A checkpoint detected")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = checkpoint["epoch"]
        self.model = checkpoint["model_state"]
        self.optimizer = checkpoint["optim_state"]
        self.scheduler = checkpoint["scheduler_state"]
        self.metrics = checkpoint['metrics']

        self.best_eer = min(self.metrics['eer'])
        print("The checkpoint loaded")
        print(f"Starting from epoch {self.current_epoch}\n")

    def _contrastive_loss(self, emb1, emb2, label, margin=0.2):        
        """
        Computes Contrastive Loss

        Args:
            emb1: first embedding
            emb2: second embedding
            label: label
            margin: margin
        """

        dist = torch.nn.functional.pairwise_distance(emb1, emb2)

        loss = (1 - label) * torch.pow(dist, 2) \
            + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss
    
    def _num_parameters(self):
        """
        Print the number of total and trainable paramters in the model.
        """
        num_params = sum([params.numel() for params in self.model.parameters()])
        num_trainables = sum([params.numel() for params in self.model.parameters() if params.requires_grad])
        print(f"Number of total parameters in the generator model: {num_params}")
        print(f"Number of trainable parameters in the generator model: {num_trainables}\n")
