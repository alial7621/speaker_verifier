import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--code_dir", type=str, default="./")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", 
                        help="path to the dataset csv and text files")
    parser.add_argument("--data_dir", type=str, default="/dataset/data")
    parser.add_argument("--trained_model", type=str, default="./checkpoints/models/best_model.pt")

    # Data Preprocessing
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration", type=int, default=3,
                        help="Input audio files must have same duration (unit: seconds)")
    parser.add_argument("--no_vad", dest="vad", action="store_false", 
                        help="Disable Voice Active Detection in preprocessing audio samples")
    parser.add_argument("--no_augmentation", dest="augmentations", action="store_false",
                        help="Use Augmentation for audio samples during training")
    
    # Model
    parser.add_argument("--in_channels", type=int, default=80, help="Input channel to the model")
    parser.add_argument("--channels", type=int, default=512, help="Intermediate channels")
    parser.add_argument("--embd_dim", type=int, default=192, help="Output embed size")
    
    # Train
    parser.add_argument("--no_manual_seed", dest="manual_seed", action="store_false", 
                        help="Whether use manual seed or not")
    parser.add_argument("--seed", type=int, default=72322)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples_per_epoch", type=int, default=30000,
                        help="number of the training samples per epoch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--load_checkpoint", type=str, default="./checkpoints/models/last_model.pt",
                        help="Start from the last checkpoint. Always refer as last_model.pt")
    
    # Optimizer and scheduler
    parser.add_argument("--loss_type", type=str, default="contrastive",
                        help="Choose the loss function, possible inputs:[contrastive, triplet, cosineemb]")
    parser.add_argument("--margin_loss", type=float, default=0.2,
                        help="The margin, using in the loss function")
    parser.add_argument("--eer_thresh", type=float, default=None, help="Threshold for EER metric")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Initial learning rate for scheduler")
    parser.add_argument("--final_lr", type=float, default=1e-6,
                        help="Final learning rate for scheduler")
    parser.add_argument("--weight_decay_factor", type=float, default=0.2,
                        help="Weight decay factor for scheduler")
    parser.add_argument("--lr_wait", type=int, default=3,
                        help="Number of cicles without significant improvement in EER for scheduler")
    parser.add_argument("--lr_thresh", type=float, default=1e-3,
                        help="Threshold to check plateau-ing of loss")
    parser.add_argument("--momentum1", type=float, default=0.5,
                        help="Optimizer momentum 1 value")
    parser.add_argument("--momentum2", type=float, default=0.999,
                        help="Optimizer momentum 1 value")
    
    return parser