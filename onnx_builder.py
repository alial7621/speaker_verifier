import argparse

import torch
import torch.onnx as onnx

from modules.model import ECAPA_TDNN

def export_model(pt_path, onnx_path, in_channel, channels, embd_dim):
    device = torch.device("cpu")

    model = ECAPA_TDNN(in_channels=in_channel,
                       channels=channels,
                       embd_dim=embd_dim)

    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 301, in_channel)  # (batch_size, seq_len, feat_dim)

    onnx.export(
        model,                      # model 
        dummy_input,                # model input 
        onnx_path,                  # where to save the model
        export_params=True,         # store the trained parameter weights
        opset_version=12,           # ONNX version
        do_constant_folding=True,   # optimization
        input_names=['input'],      # input name
        output_names=['output'],    # output name
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # dynamic batch size
    )

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument('--pt_path', type=str, default="./checkpoints/models/best_triplet.pt", 
                        help='Path of the pytorch model')
    parser.add_argument('--onnx_path', type=str, default='./checkpoints/models/best_triplet.onnx', 
                        help='Path to save the ONNX model')
    parser.add_argument('--in_channel', type=int, default=80, help='in_channel for the pt model')
    parser.add_argument('--channels', type=int, default=256, help='Size of middel layers channel')
    parser.add_argument('--embd_dim', type=int, default=192, help='size of the output vector')

    args = parser.parse_args()
    export_model(pt_path=args.pt_path, 
                 onnx_path=args.onnx_path, 
                 in_channel=args.in_channel, 
                 channels=args.channels, 
                 embd_dim=args.embd_dim)
