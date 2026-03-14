import json
import os
import time
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from collections import deque
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from transformer.policy import Policy

MODEL = "transformer"
MODEL_PATH = f"models/model_transformer.pth"
ROBOT_PORT = "/dev/ttyACM0"

def getFrame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color = np.asanyarray(aligned.get_color_frame().get_data())
    depth = np.asanyarray(aligned.get_depth_frame().get_data())
    return color, depth


def preprocess(colorBgr, depthU16):
    rgb = cv2.cvtColor(colorBgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0

    depthU8 = cv2.convertScaleAbs(depthU16, alpha=0.03)
    depthColor = cv2.applyColorMap(depthU8, cv2.COLORMAP_TURBO)
    depth = cv2.cvtColor(depthColor, cv2.COLOR_BGR2RGB)
    depth = cv2.resize(depth, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0

    combined = np.concatenate([rgb, depth[0:1]], axis=0)
    return torch.from_numpy(combined)


def main():
    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Policy(chunkSize=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.to(device).eval()

    stats = np.load("data_50v3/norm_stats.npz")
    stateMean = torch.tensor(stats["state_mean"], dtype=torch.float32)
    stateStd = torch.tensor(stats["state_std"], dtype=torch.float32)
    actionMean = stats["action_mean"]
    actionStd = stats["action_std"]

    robot = SO100Follower(SO100FollowerConfig(port=ROBOT_PORT, id="follower_arm", use_degrees=True))
    robot.connect()
    if not robot.is_connected:
        raise RuntimeError("failed to connect to robot")

    homePath = os.path.join(os.path.dirname(__file__), "TeleDex", "robots", "lerobot", "home_position.json")
    if os.path.exists(homePath):
        home = json.load(open(homePath))
        for _ in range(150):
            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
            action = {f"{j}.pos": current[j] + 0.1 * (home[j] - current[j]) for j in home if j in current}
            robot.send_action(action)
            time.sleep(1.0 / 50)
        time.sleep(5)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    featBuf = deque(maxlen=8)
    stateBuf = deque(maxlen=8)
    actionChunk = []
    chunkIdx = 0

    try:
        while True:
            t0 = time.perf_counter()
            color, depth = getFrame(pipeline, align)

            obs = robot.get_observation()
            current = {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}
            state = (torch.tensor([current[m] for m in motors], dtype=torch.float32) - stateMean) / stateStd

            frame = preprocess(color, depth).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.cnn(frame).squeeze(0).cpu()
            featBuf.append(feat)
            stateBuf.append(state)

            feats = [featBuf[0]] * (8 - len(featBuf)) + list(featBuf)
            states = [stateBuf[0]] * (8 - len(stateBuf)) + list(stateBuf)
            featSeq = torch.stack(feats).unsqueeze(0).to(device)
            stateSeq = torch.stack(states).unsqueeze(0).to(device)

            with torch.no_grad():
                newChunk = model(featSeq, stateSeq).squeeze(0).cpu().numpy() * actionStd + actionMean

            if len(actionChunk) > 0 and chunkIdx < len(actionChunk):
                for k in range(min(len(actionChunk) - chunkIdx, len(newChunk))):
                    newChunk[k] = 0.9 * newChunk[k] + 0.1 * actionChunk[chunkIdx + k]
            actionChunk = newChunk
            chunkIdx = 0

            action = actionChunk[chunkIdx]
            chunkIdx += 1

            cmd = {f"{m}.pos": float(action[i]) for i, m in enumerate(motors) if m in current}
            robot.send_action(cmd)

            cv2.imshow("camera", color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(1/30 - elapsed, 0.0))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.disconnect()

if __name__ == "__main__":
    main()
