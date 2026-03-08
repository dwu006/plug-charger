"""
Record a LeRobot dataset by teleoperation with the Teledex phone app.

Same teleoperation as teleop.py — phone drives the arm, TOGGLE opens/closes gripper.

Keyboard controls:
  S — start recording the current episode
  Y — save the episode and move to the next
  R — discard and re-record the current episode
  Q — stop and finalize the dataset
"""

import os
import time

import numpy as np
from pynput import keyboard

from teledex import Session
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ── Configuration ─────────────────────────────────────────────────────────────
TASK         = "pick up the block"
REPO_ID      = "local/so101"
ROOT         = "data"
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 100
ROBOT_PORT   = "/dev/ttyACM0"
PUSH_TO_HUB  = False
# ──────────────────────────────────────────────────────────────────────────────

import pathlib

FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
R_PHONE_TO_WORLD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def init_keyboard(events: dict):
    """Non-blocking keyboard listener: s=start, y=save, r=rerecord, q=quit."""
    def on_press(key):
        try:
            if hasattr(key, "char") and key.char:
                ch = key.char.lower()
            else:
                ch = None
        except Exception:
            ch = None
        if ch == "s":
            events["start"] = True
            print("\n[Keyboard] S pressed - starting recording...")
        if ch == "y":
            events["save"] = True
            print("\n[Keyboard] Y pressed - saving...")
        if ch == "r":
            events["rerecord"] = True
            print("\n[Keyboard] R pressed - discarding...")
        if ch == "q":
            events["quit"] = True
            print("\n[Keyboard] Q pressed - quitting...")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def teleop_step(latest, robot_obs, phone_origin, gripper_closed, last_toggle,
                teleop_proc, robot_proc, rot_ref=None):
    """Run one teleop step. Returns (joint_action, ee_action, phone_origin, gripper_closed, last_toggle, rot_ref)."""
    pos = latest.get("position")
    if pos is None:
        pos = latest.get("position_hand")
    toggle = latest.get("toggle")
    rotation = latest.get("rotation")

    if pos is not None and phone_origin is None:
        phone_origin = np.array(pos, dtype=float)

    if rotation is not None and rot_ref is None:
        rot_ref = np.array(rotation, dtype=float)

    if pos is not None and phone_origin is not None:
        delta = R_PHONE_TO_WORLD @ (np.array(pos, dtype=float) - phone_origin)
        enabled = True
    else:
        delta, enabled = np.zeros(3), False

    target_wy = 0.0
    if rotation is not None and rot_ref is not None:
        R_rel = rot_ref.T @ np.array(rotation, dtype=float)
        target_wy = float(np.arctan2(-R_rel[2][0], np.sqrt(R_rel[2][1]**2 + R_rel[2][2]**2)))

    if last_toggle is not None and toggle != last_toggle:
        gripper_closed = not gripper_closed
    last_toggle = toggle

    action = {
        "enabled": enabled,
        "target_x": float(delta[0]), "target_y": float(delta[1]), "target_z": float(delta[2]),
        "target_wx": 0.0, "target_wy": target_wy, "target_wz": 0.0,
        "gripper_vel": 1.0 if gripper_closed else -1.0,
    }
    ee_action    = teleop_proc((action, robot_obs))
    joint_action = robot_proc((ee_action, robot_obs))
    return joint_action, ee_action, phone_origin, gripper_closed, last_toggle, rot_ref


def main() -> None:
    here = os.path.dirname(__file__)
    urdf_path = os.path.join(here, "SO101", "so101_new_calib.urdf")

    # --- Robot + kinematics ---
    robot = SO100Follower(SO100FollowerConfig(port=ROBOT_PORT, id="follower_arm", use_degrees=True))
    kinematics = RobotKinematics(urdf_path=urdf_path, target_frame_name="gripper_frame_link",
                                  joint_names=MOTOR_NAMES)

    # --- Pipelines ---
    teleop_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEReferenceAndDelta(kinematics=kinematics, end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
                                motor_names=MOTOR_NAMES, use_latched_reference=True),
            EEBoundsAndSafety(end_effector_bounds={"min": [-0.5, -0.5, -0.1], "max": [0.5, 0.5, 0.6]},
                              max_ee_step_m=0.02),
            GripperVelocityToJoint(speed_factor=2.0),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_proc = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=MOTOR_NAMES)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    # --- Dataset ---
    joint_names = [f"{m}.pos" for m in MOTOR_NAMES]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": joint_names},
        "action":            {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": joint_names},
    }

    root_path = pathlib.Path(ROOT)
    counter = 1
    while root_path.exists():
        root_path = pathlib.Path(f"{ROOT}_{counter}")
        counter += 1

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, root=str(root_path), robot_type=robot.name,
        features=features, use_videos=False,
    )

    # --- Connect ---
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to SO101.")

    session = Session(debug=False)
    session.start()

    print("\n[Record] Waiting for teleop connection...")
    print("Connect your phone to the Teledex app.")
    while len(session.connected_clients) == 0:
        time.sleep(0.1)
    print("[Record] Teleop connected! Ready to record.\n")

    events = {"start": False, "save": False, "rerecord": False, "quit": False}
    kb_listener = init_keyboard(events)

    print(f"[Record] Task: '{TASK}'  |  {NUM_EPISODES} episodes")
    print("[Record] S=start  Y=save  R=re-record  Q=quit\n")

    phone_origin   = None
    rot_ref        = None
    gripper_closed = False
    last_toggle    = None
    recording      = False
    recorded       = 0
    prompt_shown   = False
    steps_recorded = 0
    max_reached    = False

    try:
        while recorded < NUM_EPISODES and not events["quit"]:
            if len(session.connected_clients) == 0:
                print("[Record] Warning: Teleop disconnected. Waiting for reconnection...")
                while len(session.connected_clients) == 0:
                    time.sleep(0.5)
                print("[Record] Teleop reconnected!")
                phone_origin = None
                prompt_shown = False

            if not recording and not prompt_shown:
                if max_reached:
                    print(f"\n[Episode {recorded + 1}/{NUM_EPISODES}] Reached {MAX_STEPS_PER_EPISODE} steps. Y to save, R to discard.")
                else:
                    print(f"[Episode {recorded + 1}/{NUM_EPISODES}] Press S to start recording.")
                prompt_shown = True

            # Always teleoperate (arm moves whether recording or not)
            t0 = time.perf_counter()
            robot_obs = robot.get_observation()
            latest    = session.get_latest_data()

            if latest is None:
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                continue

            joint_action, ee_action, phone_origin, gripper_closed, last_toggle, rot_ref = teleop_step(
                latest, robot_obs, phone_origin, gripper_closed, last_toggle, teleop_proc, robot_proc, rot_ref
            )
            robot.send_action(joint_action)

            # Record frame only when recording
            if recording:
                try:
                    obs_state = np.array([robot_obs.get(k, 0.0) for k in joint_names], dtype=np.float32)
                    act_state = np.array([joint_action.get(k, 0.0) for k in joint_names], dtype=np.float32)
                    frame = {"observation.state": obs_state, "action": act_state, "task": TASK}
                    dataset.add_frame(frame)
                    steps_recorded += 1
                    
                    progress = steps_recorded / MAX_STEPS_PER_EPISODE
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"\r[Episode {recorded + 1}] Steps: {steps_recorded}/{MAX_STEPS_PER_EPISODE} [{bar}] {int(progress*100)}%", end="", flush=True)
                    
                    if steps_recorded >= MAX_STEPS_PER_EPISODE:
                        print(f"\n[Episode {recorded + 1}] Reached {MAX_STEPS_PER_EPISODE} steps. Stopping recording.")
                        recording = False
                        max_reached = True
                        prompt_shown = False
                        
                except Exception as e:
                    print(f"\n[Record] Error recording frame: {e}")
                    print("[Record] Stopping recording. Press S to try again.")
                    recording = False
                    steps_recorded = 0

            # Handle key events
            if events["start"] and not recording:
                events["start"] = False
                recording = True
                prompt_shown = False
                max_reached = False
                steps_recorded = 0
                phone_origin = None  # re-latch phone origin and rotation ref at episode start
                rot_ref = None
                log_say(f"Recording episode {recorded + 1}")
                print(f"\n[Episode {recorded + 1}] Recording ({MAX_STEPS_PER_EPISODE} max steps) — Y to save, R to discard.")

            if events["save"] and (recording or max_reached):
                events["save"] = False
                if steps_recorded == 0:
                    print("\n[Record] No frames recorded. Cannot save empty episode.")
                    continue
                recording = False
                max_reached = False
                prompt_shown = False
                dataset.save_episode()
                recorded += 1
                log_say(f"Episode {recorded} saved")
                print(f"\n[Episode {recorded}/{NUM_EPISODES}] ✓ Saved ({steps_recorded} steps).\n")
                steps_recorded = 0

            if events["rerecord"] and (recording or max_reached):
                events["rerecord"] = False
                recording = False
                max_reached = False
                prompt_shown = False
                steps_recorded = 0
                dataset.clear_episode_buffer()
                log_say("Discarded, re-record")
                print(f"\n[Episode {recorded + 1}] ✗ Discarded. Press S to re-record.\n")

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping")
        try:
            dataset.finalize()
            if PUSH_TO_HUB:
                dataset.push_to_hub()
        except Exception as e:
            print(f"[Record] Warning: Error finalizing dataset: {e}")
        finally:
            session.stop()
            robot.disconnect()
            kb_listener.stop()
            print(f"\n[Record] Done — {recorded} episodes saved to '{root_path}/{REPO_ID}'.")


if __name__ == "__main__":
    main()
