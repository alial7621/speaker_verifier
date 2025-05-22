import os

import torch
import numpy as np

from modules import config
from modules.data_loader import get_data_loaders
from modules.trainer import VerifierTrainer
from utils.util import plot_metrics

def train(config):
    
    if config.manual_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(config.checkpoint_dir, "plots"), exist_ok=True)

    # prepare datalaoders
    train_loader, val_loader = get_data_loaders(config.dataset_dir, 
                                                samples_per_epoch=config.samples_per_epoch,
                                                loss_type=config.loss_type,
                                                sample_rate=config.sample_rate,
                                                duration=config.duration, 
                                                vad=config.vad, 
                                                augmentations=config.augmentations,
                                                batch_size=config.batch_size,
                                                mfcc_feat_dim=config.mfcc_feat_dim)
    
    # Initialize trainer
    trainer = VerifierTrainer(config, device)
    
    # Load checkpoint if provided
    if config.checkpoint_path is not None and os.path.isfile(config.checkpoint_path):
        trainer.load_checkpoint(config.checkpoint_path)
        
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Plot metrics
    plot_metrics(trainer.metrics, save_path=os.path.join(config.checkpoint_dir, "plots"))
    
    print("Training completed!")


def test(config):
    pass

if __name__ == "__main__":
    
    parser = config.get_argparser()
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        parser.print_help()