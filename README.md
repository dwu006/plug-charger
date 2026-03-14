# Learning to Pick and Place

cs147 project

### Demo
https://github.com/user-attachments/assets/78281342-b421-4e9d-a988-e4f57dd24456

### Teleoperation
https://github.com/user-attachments/assets/034bc5d7-fdad-4df6-9c0e-021e309ba249

## Setup

```bash
conda env create -f environment.yml
conda activate venv
```

## Project Structure

```
cnn.py              - ResNet-18 visual encoder (4-channel RGBD input)
transformer/        - Transformer policy and positional encoder
load_dataset.py     - Dataset loading, caching, and normalization
train.py            - Two-phase training (cached features then fine-tuning)
deploy.py           - Real-time deployment on the robot
TeleDex/            - Keyboard IK teleoperation for data collection
models/             - Saved model checkpoints
```

## Training

```bash
python train.py
```

Trains in two phases:
1. Transformer + MLP head on cached CNN features (up to 100 epochs)
2. Fine-tuning with unfrozen CNN layers on raw images (20 epochs)

## Deployment

```bash
python deploy.py
```

Runs the policy at 30 FPS on the robot. Press Q to quit.
