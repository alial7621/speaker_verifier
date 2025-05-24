import matplotlib.pyplot as plt
import torch


def plot_metrics(metrics, save_path=None):
    """
    Plot training metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        save_path (str, optional): Path to save plot
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Plot training loss
    if 'loss' in metrics:
        axes[0].plot(metrics['loss'])
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
    
    # Plot val eer
    if 'eer' in metrics:
        axes[1].plot(metrics['eer'])
        axes[1].set_title("Val EER")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("EER")
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()