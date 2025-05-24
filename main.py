import os
import warnings

import numpy as np
import torch

from modules import config
from modules.data_loader import get_data_loaders, get_test_loader
from modules.tester import SpeakerVerification, VerifierTester
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.single_pred:
        assert config.audio1 and config.audio2 is not None
        speaker_verifier = SpeakerVerification(config, device)
        speaker_verifier.verify(config.audio1, config.audio2)
        exit()

    tester = VerifierTester(config, device)
    if not config.find_optim_thresh:
        test_loader = get_test_loader(
            config.dataset_dir, 
            sample_rate=config.sample_rate,
            duration=config.duration, 
            vad=config.vad, 
            batch_size=config.batch_size,
            mfcc_feat_dim=config.mfcc_feat_dim,
            testset_only=True
        )
        tester.test(test_loader)
    else:
        val_loader, test_loader = get_test_loader(
            config.dataset_dir, 
            sample_rate=config.sample_rate,
            duration=config.duration, 
            vad=config.vad, 
            batch_size=config.batch_size,
            mfcc_feat_dim=config.mfcc_feat_dim,
            testset_only=False
        )
        tester.validate(val_loader)
        tester.test(test_loader)

    print("Test completed!")
    

if __name__ == "__main__":
    
    parser = config.get_argparser()
    args = parser.parse_args()

    # filter UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        parser.print_help()