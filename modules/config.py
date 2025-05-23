import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest="mode", help="Mode")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train model")

    # Data
    train_parser.add_argument("--code_dir", type=str, default="./")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--dataset_dir", type=str, default="./dataset", 
                              help="path to the dataset csv and text files")
    train_parser.add_argument("--data_dir", type=str, default="/dataset/data")

    # Data Preprocessing
    train_parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    train_parser.add_argument("--duration", type=int, default=3,
                              help="Input audio files must have same duration (unit: seconds)")
    train_parser.add_argument("--no_vad", dest="vad", action="store_false", 
                              help="Disable Voice Active Detection in preprocessing audio samples")
    train_parser.add_argument("--no_augmentation", dest="augmentations", action="store_false",
                              help="Use Augmentation for audio samples during training")
    
    # Model
    train_parser.add_argument("--mfcc_feat_dim", type=int, default=80, help="Input channel to the model")
    train_parser.add_argument("--channels", type=int, default=256, help="Intermediate channels")
    train_parser.add_argument("--embd_dim", type=int, default=192, help="Output embed size")
    
    # Train
    train_parser.add_argument("--no_manual_seed", dest="manual_seed", action="store_false", 
                              help="Whether use manual seed or not")
    train_parser.add_argument("--seed", type=int, default=72322)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--samples_per_epoch", type=int, default=20000,
                              help="number of the training samples per epoch")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/models/last.pt",
                              help="Start from the last checkpoint. Always refer as last_model.pt")
    
    # Optimizer and scheduler
    train_parser.add_argument("--loss_type", type=str, default="contrastive",
                              help="Choose the loss function, possible inputs:[contrastive, triplet, cosineemb, aamsoftmax]")
    train_parser.add_argument("--margin_loss", type=float, default=0.2,
                              help="The margin, using in the loss function")
    train_parser.add_argument("--eer_thresh", type=float, default=None, help="Threshold for EER metric")
    train_parser.add_argument("--lr", type=float, default=1e-2,
                              help="Initial learning rate for scheduler")
    train_parser.add_argument("--final_lr", type=float, default=1e-6,
                              help="Final learning rate for scheduler")
    train_parser.add_argument("--weight_decay_factor", type=float, default=0.2,
                              help="Weight decay factor for scheduler")
    train_parser.add_argument("--lr_wait", type=int, default=3,
                              help="Number of cicles without significant improvement in EER for scheduler")
    train_parser.add_argument("--lr_thresh", type=float, default=1e-3,
                              help="Threshold to check plateau-ing of loss")
    train_parser.add_argument("--momentum1", type=float, default=0.5,
                              help="Optimizer momentum 1 value")
    train_parser.add_argument("--momentum2", type=float, default=0.999,
                              help="Optimizer momentum 1 value")
    
    # Test parser
    test_parser = subparsers.add_parser("test", help="Test model")

    # Test arguments
    test_parser.add_argument("--dataset_dir", type=str, default="./dataset", 
                             help="path to the testset csv file")
    test_parser.add_argument("--data_dir", type=str, default="/dataset/data")
    test_parser.add_argument("--checkpoint", type=str, default="./checkpoints/models/best.pt")
    
    test_parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    test_parser.add_argument("--duration", type=int, default=3,
                             help="Input audio files must have same duration (unit: seconds)")
    test_parser.add_argument("--no_vad", dest="vad", action="store_false", 
                             help="Disable Voice Active Detection in preprocessing audio samples")
    test_parser.add_argument("--batch_size", type=int, default=32)
    test_parser.add_argument("--eer_thresh", type=float, default=None, help="Threshold for EER metric")
    test_parser.add_argument("--single_pred", action="store_true", help="Single input prediction")
    test_parser.add_argument("--audio1", type=str, default=None, help="Use in single prediction mode")
    test_parser.add_argument("--audio2", type=str, default=None, help="Use in single prediction mode")
    
    return parser