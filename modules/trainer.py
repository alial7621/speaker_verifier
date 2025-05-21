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
        self.metric = BinaryEER(thresholds=config.eer_thresh)

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
        pass

    def _train_epoch(self, train_loader):
        pass

    def _validate(self, val_loader):        
        pass

    def _save_checkpoint(self, epoch):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

    def _contrastive_loss(self, emb1, emb2, label, margin=0.2):        
        """
        Computes Contrastive Loss
        """

        dist = torch.nn.functional.pairwise_distance(emb1, emb2)

        loss = (1 - label) * torch.pow(dist, 2) \
            + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss

