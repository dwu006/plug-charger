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
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ── Configuration ─────────────────────────────────────────────────────────────
TASK         = "pick up the block"
REPO_ID      = "local/so101"
ROOT         = "data"
NUM_EPISODES = 10
ROBOT_PORT   = "/dev/ttyACM0"
PUSH_TO_HUB  = False
# ──────────────────────────────────────────────────────────────────────────────

FPS = 30
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
R_PHONE_TO_WORLD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def init_keyboard(events: dict):
    """Non-blocking keyboard listener: s=start, y=save, r=rerecord, q=quit."""
    def on_press(key):
        try:
            ch = key.char.lower() if hasattr(key, "char") and key.char else None
        except Exception:
            ch = None
        if ch == "s": events["start"]    = True
        if ch == "y": events["save"]     = True
        if ch == "r": events["rerecord"] = True
        if ch == "q": events["quit"]     = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def teleop_step(latest, robot_obs, phone_origin, gripper_closed, last_toggle,
                teleop_proc, robot_proc):
    """Run one teleop step. Returns (joint_action, ee_action, phone_origin, gripper_closed, last_toggle)."""
    pos = latest.get("position")
    if pos is None:
        pos = latest.get("position_hand")
    toggle = latest.get("toggle")

    if pos is not None and phone_origin is None:
        phone_origin = np.array(pos, dtype=float)

    if pos is not None and phone_origin is not None:
        delta = R_PHONE_TO_WORLD @ (np.array(pos, dtype=float) - phone_origin)
        enabled = True
    else:
        delta, enabled = np.zeros(3), False

    if last_toggle is not None and toggle != last_toggle:
        gripper_closed = not gripper_closed
    last_toggle = toggle

    action = {
        "enabled": enabled,
        "target_x": float(delta[0]), "target_y": float(delta[1]), "target_z": float(delta[2]),
        "target_wx": 0.0, "target_wy": 0.0, "target_wz": 0.0,
        "gripper_vel": 1.0 if gripper_closed else -1.0,
    }
    ee_action    = teleop_proc((action, robot_obs))
    joint_action = robot_proc((ee_action, robot_obs))
    return joint_action, ee_action, phone_origin, gripper_closed, last_toggle


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
    obs_proc = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=MOTOR_NAMES)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # --- Dataset ---
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, root=ROOT, robot_type=robot.name,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_proc,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=False,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=obs_proc,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=False,
            ),
        ),
        use_videos=False,
    )

    # --- Connect ---
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("Failed to connect to SO101.")

    session = Session(debug=False)
    session.start()

    events = {"start": False, "save": False, "rerecord": False, "quit": False}
    kb_listener = init_keyboard(events)

    print(f"\n[Record] Task: '{TASK}'  |  {NUM_EPISODES} episodes")
    print("[Record] S=start  Y=save  R=re-record  Q=quit\n")

    phone_origin   = None
    gripper_closed = False
    last_toggle    = None
    recording      = False
    recorded       = 0

    try:
        while recorded < NUM_EPISODES and not events["quit"]:

            if not recording:
                print(f"[Episode {recorded + 1}/{NUM_EPISODES}] Press S to start recording.")

            # Always teleoperate (arm moves whether recording or not)
            t0 = time.perf_counter()
            robot_obs = robot.get_observation()
            latest    = session.get_latest_data()

            joint_action, ee_action, phone_origin, gripper_closed, last_toggle = teleop_step(
                latest, robot_obs, phone_origin, gripper_closed, last_toggle, teleop_proc, robot_proc
            )
            robot.send_action(joint_action)

            # Record frame only when recording
            if recording:
                obs_processed = obs_proc(robot_obs)
                frame = {
                    **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
                    **build_dataset_frame(dataset.features, ee_action, prefix=ACTION),
                    "task": TASK,
                }
                dataset.add_frame(frame)

            # Handle key events
            if events["start"] and not recording:
                events["start"] = False
                recording = True
                phone_origin = None  # re-latch phone origin at episode start
                log_say(f"Recording episode {recorded + 1}")
                print(f"[Episode {recorded + 1}] Recording — Y to save, R to discard.")

            if events["save"] and recording:
                events["save"] = False
                recording = False
                dataset.save_episode()
                recorded += 1
                log_say(f"Episode {recorded} saved")
                print(f"[Episode {recorded}/{NUM_EPISODES}] Saved.\n")

            if events["rerecord"] and recording:
                events["rerecord"] = False
                recording = False
                dataset.clear_episode_buffer()
                log_say("Discarded, re-record")
                print(f"[Episode {recorded + 1}] Discarded. Press S to re-record.\n")

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stopping")
        dataset.finalize()
        if PUSH_TO_HUB:
            dataset.push_to_hub()
        session.stop()
        robot.disconnect()
        kb_listener.stop()
        print(f"\n[Record] Done — {recorded} episodes saved to '{ROOT}/{REPO_ID}'.")


if __name__ == "__main__":
    main()
