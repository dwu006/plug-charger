import os
import time

import numpy as np

from teledex import Session
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

FPS = 30

_R_PHONE_TO_WORLD = np.array([[0, -1, 0],
                               [1,  0, 0],
                               [0,  0, 1]], dtype=float)


def main(robot_port: str = "/dev/ttyACM0") -> None:
    here = os.path.dirname(__file__)
    urdf_path = os.path.join(here, "SO101", "so101_new_calib.urdf")

    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    robot = SO100Follower(SO100FollowerConfig(port=robot_port, id="follower_arm", use_degrees=True))

    kinematics = RobotKinematics(urdf_path=urdf_path, target_frame_name="gripper_frame_link",
                                  joint_names=motor_names)
    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0, "wx": 1.0, "wy": 1.0, "wz": 1.0},
                motor_names=motor_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.5, -0.5, -0.1], "max": [0.5, 0.5, 0.6]},
                max_ee_step_m=0.02,
            ),
            GripperVelocityToJoint(speed_factor=2.0),
            InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motor_names),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("failed to connect to SO101")

    session = Session(debug=True)
    session.start()

    print("ready - phone position drives EE, toggle opens/closes gripper")

    phone_origin = None
    rot_ref = None
    gripper_closed = False
    last_toggle = None

    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()
            latest = session.get_latest_data()

            pos = latest.get("position")
            if pos is None:
                pos = latest.get("position_hand")

            rot = latest.get("rotation")
            toggle = latest.get("toggle")

            if pos is not None and phone_origin is None:
                phone_origin = np.array(pos, dtype=float)

            if rot is not None and rot_ref is None:
                rot_ref = np.array(rot, dtype=float)

            if pos is not None and phone_origin is not None:
                raw = np.array(pos, dtype=float) - phone_origin
                delta = _R_PHONE_TO_WORLD @ raw
                enabled = True
            else:
                delta = np.zeros(3)
                enabled = False

            target_wy = 0.0
            if rot is not None and rot_ref is not None:
                R_rel = rot_ref.T @ np.array(rot, dtype=float)
                target_wy = float(np.arctan2(-R_rel[2][0], np.sqrt(R_rel[2][1]**2 + R_rel[2][2]**2)))

            if last_toggle is not None and toggle != last_toggle:
                gripper_closed = not gripper_closed
            last_toggle = toggle

            action = {
                "enabled": enabled,
                "target_x": float(delta[0]),
                "target_y": float(delta[1]),
                "target_z": float(delta[2]),
                "target_wx": 0.0,
                "target_wy": target_wy,
                "target_wz": 0.0,
                "gripper_vel": 1.0 if gripper_closed else -1.0,
            }

            joint_action = pipeline((action, robot_obs))
            robot.send_action(joint_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        print("stopping")
        session.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
