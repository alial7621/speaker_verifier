import os
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import BinaryEER

from model import ECAPA_TDNN

class VerifierTrainer:
    """
    Class for training a speaker verification model.
    Handles the training process, checkpointing, and logging.
    """
    def __init__(self, config, device):
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
            in_channels=config.in_channels,
            channels=config.channels,
            embd_dim=config.embd_dim
        ).to(device)
        self._num_parameters(self)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.generator.parameters(),
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
            self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Time: {elapsed:.2f}s - "
                  f"Loss: {self.metrics['g_loss'][-1]:.4f}")
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def _train_epoch(self, train_loader):
        pass

    def _validate(self, val_loader):        
        pass

    def _save_checkpoint(self, mode='last'):
        """ 
        save a checkpoint of the models
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


