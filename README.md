# Learning to Pick and Place

Imitation learning system for a robotic pick-and-place task using a LeRobot SO-100 arm and Intel RealSense RGBD camera. A Transformer policy learns from teleoperated demonstrations to predict motor commands with action chunking.

<table>
<tr>
<td align="center"><b>Deployment</b></td>
<td align="center"><b>Teleoperation</b></td>
</tr>
<tr>
<td>

https://github.com/dwu006/plug-charger/raw/main/demo/demo.mp4

</td>
<td>

https://github.com/dwu006/plug-charger/raw/main/demo/teleop.mp4

</td>
</tr>
</table>

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
