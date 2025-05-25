# Speaker Verification System

## Overview
The speaker verification system is a critical application in the field of speech processing and biometric identification. Its primary goal is to identify or verify individuals based on the unique characteristics of their voice. Each person has distinct speech patterns, such as tone, speaking rate, dominant frequencies, and acoustic features, which can be leveraged for identification purposes. These systems are widely used in authentication applications. Additionally, modeling voice characteristics plays a significant role in other speech processing tasks, including multi-speaker text-to-speech, voice imitation, and speaker diarization. Key challenges in designing such systems include achieving high accuracy, robustness to noise, and effective performance under varying conditions. This project focuses on implementing a deep learning-based system for speaker verification from audio signals.

## Directory Structure
```
speaker_verifier/
├── checkpoints/
│   ├── model/
│   └── plots/
├── dataset/
│   ├── data/
│   ├── train_speakers.txt
│   ├── val_speakers.txt
│   ├── test_speakers.txt
│   ├── validation.csv
│   └── test.csv
├── modules/
│   ├── config.py
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   ├── tester.py
│   └── aamsoftmax.py
├── utils/
│   └── util.py
├── main.py
├── onnx_builder.py
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── requirements.txt
└── requirements.yaml
```

## Installation

This project supports three methods for setting up the environment: native Python, Conda, and Docker.

### 1. Native Python Environment
To set up the project using a native Python environment, install the required dependencies with the following command:
```
pip install -r requirements.txt
```
This project uses **PyTorch 2.5.0** with CUDA support for GPU acceleration. To install PyTorch 2.5.0 with CUDA, run:
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```
Ensure you have a compatible CUDA version installed on your system.

### 2. Conda Environment
To set up the project using a Conda environment, follow these steps:

1. Create and activate a new Conda environment using the provided `requirements.yaml` file:
   ```
   conda env create -f requirements.yaml
   conda activate spk-verify-env
   ```
2. The `requirements.yaml` file includes all dependencies, including PyTorch 2.5.0 with CUDA support. Ensure you have a compatible CUDA version installed on your system.

### 3. Docker Container
To set up the project using Docker, follow these steps:

1. Build the Docker image using the provided `Dockerfile` and `docker-compose.yaml`:
   ```
   docker-compose build
   ```
2. Start the container in detached mode:
   ```
   docker-compose up -d
   ```
3. Access the container's shell for running commands:
   ```
   docker exec -it speaker-verifier bash
   ```
The `Dockerfile` and `docker-compose.yaml` handle the installation of dependencies, including PyTorch 2.5.1 with CUDA support. Ensure Docker and Docker Compose are installed on your system.

## Usage
The `main.py` module serves as the entry point for training, evaluating a dataset, and performing inference with the neural network to obtain outputs for a given sample.

### Training
To train the model, use the following command:
```
python main.py train
```
The `train` argument specifies the task. Additional arguments can be provided to customize the training process, as defined in the `config.py` module. These arguments can be modified either in the `config.py` file or via the command line. For example:
```
python main.py train --batch_size=16
```
Note that arguments like `--batch_size` are prefixed with double dashes (`--`).

### Evaluation
To evaluate the model on a dataset, use the `test` task instead of `train`. The list of available arguments for testing is also defined in the `config.py` module, following the training arguments. Example:
```
python main.py test
```

### Single Prediction
To compare the similarity between two input audio samples, use the `--single_pred` argument along with the paths to the two audio files:
```
python main.py test --single_pred --audio1 path_to_audio1 --audio2 path_to_audio2
```
The `--single_pred` flag enables single prediction mode, and `--audio1` and `--audio2` specify the paths to the input audio files.

## Exporting to ONNX
The `onnx_builder.py` module is used to export a PyTorch model to ONNX format. To export the model, run:
```
python onnx_builder.py
```
The module supports the following arguments:
- `--pt_path`: Path to the PyTorch model (default: `./checkpoints/models/best_triplet.pt`)
- `--onnx_path`: Path to save the ONNX model (default: `./checkpoints/models/best_triplet.onnx`)
- `--in_channel`: Input channel for the PyTorch model (default: `80`)
- `--channels`: Size of middle layers' channels (default: `256`)
- `--embd_dim`: Size of the output vector (default: `192`)

Example:
```
python onnx_builder.py --pt_path ./checkpoints/models/custom_model.pt --onnx_path ./checkpoints/models/custom_model.onnx
```

### Single Prediction with ONNX Model
To compare the similarity between two input audio samples using the ONNX model, include the `--onnx_model` flag and specify the path to the ONNX model with `--checkpoint`:
```
python main.py test --single_pred --onnx_model --audio1 path_to_audio1 --audio2 path_to_audio2 --checkpoint path_to_onnx_model
```
- `--single_pred`: Flag to enable single prediction mode
- `--onnx_model`: Flag to use the ONNX model instead of the PyTorch model
- `--audio1`: Path to the first audio sample
- `--audio2`: Path to the second audio sample
- `--checkpoint`: Path to the ONNX model (e.g., `./checkpoints/models/best_triplet.onnx`)