# TeleDex — SO101 Teleoperation

Teleoperate a physical or simulated SO101 arm using your phone via the [TeleDex](https://github.com/omarrayyann/TeleDex) app.

## Setup

**1. Install dependencies**
```bash
pip install lerobot teledex mujoco
```

**2. Clone and enter the repo**
```bash
git clone https://github.com/omarrayyann/TeleDex
cd TeleDex/robots/lerobot
```

## MuJoCo Simulation

```bash
python so101_mujoco_teleop.py
```

Scan the QR code with the TeleDex app. Move your phone to drive the arm.

| Control | Action |
|---|---|
| Move phone | Move end-effector |
| Toggle button | Open / close gripper |

## Real Robot

Plug in the SO101 arm via USB, then:

```bash
python teleop.py
```

If you get a permission error on connect:
```bash
sudo chmod 666 /dev/ttyACM0
```

By default expects the robot on `/dev/ttyACM0`. To change the port:
```python
# bottom of teleop.py
main(robot_port="/dev/ttyACM1")
```

| Control | Action |
|---|---|
| Move phone | Move end-effector |
| Toggle button | Open / close gripper |

## Files

```
SO101/              MuJoCo model (scene.xml, so101_new_calib.xml, assets/)
so101_mujoco_teleop.py    Simulation teleop
teleop.py                 Real robot teleop
```
