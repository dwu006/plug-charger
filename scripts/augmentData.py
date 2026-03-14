import os
import json
import shutil
import random

import cv2
import numpy as np
import pandas as pd

INPUT = "data"
OUTPUT = "data_aug"
MULTIPLIER = 1
JOINT_NOISE = 0.005


def readFrames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def writeVideo(path, frames, fps, width, height):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def augmentFrames(frames, brightness):
    augmented = []
    for frame in frames:
        f = frame.astype(np.float32) + brightness
        augmented.append(np.clip(f, 0, 255).astype(np.uint8))
    return augmented


def sampleAugParams():
    brightness = random.uniform(-30, 30)
    return (brightness,)


def columnStats(values):
    arr = np.array(values, dtype=np.float32)
    if arr.ndim == 1:
        return {
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [int(len(arr))],
            "q01": [float(np.quantile(arr, 0.01))],
            "q10": [float(np.quantile(arr, 0.10))],
            "q50": [float(np.quantile(arr, 0.50))],
            "q90": [float(np.quantile(arr, 0.90))],
            "q99": [float(np.quantile(arr, 0.99))],
        }
    else:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(len(arr))],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }


def main():
    with open(os.path.join(INPUT, "meta", "info.json")) as f:
        info = json.load(f)

    origEpisodes = info["total_episodes"]
    origFrames = info["total_frames"]
    fps = info["fps"]

    sampleVid = os.path.join(INPUT, "videos", "observation.images.rgb", "episode_000000.mp4")
    cap = cv2.VideoCapture(sampleVid)
    vidW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    shutil.copytree(INPUT, OUTPUT)

    dataDir = os.path.join(INPUT, "data", "chunk-000")
    parquetFiles = sorted(os.path.join(dataDir, f) for f in os.listdir(dataDir))
    origDf = pd.concat([pd.read_parquet(f) for f in parquetFiles], ignore_index=True)

    epMetaDir = os.path.join(INPUT, "meta", "episodes", "chunk-000")
    origEpMeta = pd.read_parquet(epMetaDir)

    allNewRows = []
    allNewEpMeta = []
    runningIndex = len(origDf)
    newEpIdx = origEpisodes

    for copyI in range(MULTIPLIER):
        for ep in range(origEpisodes):
            print(f"  ep {ep} -> new ep {newEpIdx}")

            params = sampleAugParams()

            rgbIn = os.path.join(INPUT, "videos", "observation.images.rgb", f"episode_{ep:06d}.mp4")
            rgbOut = os.path.join(OUTPUT, "videos", "observation.images.rgb", f"episode_{newEpIdx:06d}.mp4")
            rgbFrames = readFrames(rgbIn)
            augFrames = augmentFrames(rgbFrames, *params)
            writeVideo(rgbOut, augFrames, fps, vidW, vidH)

            depthIn = os.path.join(INPUT, "videos", "observation.images.depth", f"episode_{ep:06d}.mp4")
            depthOut = os.path.join(OUTPUT, "videos", "observation.images.depth", f"episode_{newEpIdx:06d}.mp4")
            shutil.copy2(depthIn, depthOut)

            epRows = origDf[origDf["episode_index"] == ep].copy()
            epLen = len(epRows)
            epRows["episode_index"] = newEpIdx
            epRows["index"] = list(range(runningIndex, runningIndex + epLen))
            if JOINT_NOISE > 0:
                states = np.stack(epRows["observation.state"].values)
                actions = np.stack(epRows["action"].values)
                states = states + np.random.normal(0, JOINT_NOISE, states.shape).astype(np.float32)
                actions = actions + np.random.normal(0, JOINT_NOISE, actions.shape).astype(np.float32)
                epRows["observation.state"] = list(states)
                epRows["action"] = list(actions)
            allNewRows.append(epRows)

            origMetaRow = origEpMeta[origEpMeta["episode_index"] == ep].iloc[0].to_dict()
            origMetaRow["episode_index"] = newEpIdx
            origMetaRow["dataset_from_index"] = runningIndex
            origMetaRow["dataset_to_index"] = runningIndex + epLen
            origMetaRow["data/chunk_index"] = 0
            origMetaRow["data/file_index"] = 0
            origMetaRow["meta/episodes/chunk_index"] = 0
            origMetaRow["meta/episodes/file_index"] = 0
            newIndices = list(range(runningIndex, runningIndex + epLen))
            origMetaRow["stats/index/min"] = [float(min(newIndices))]
            origMetaRow["stats/index/max"] = [float(max(newIndices))]
            origMetaRow["stats/index/mean"] = [float(np.mean(newIndices))]
            origMetaRow["stats/index/std"] = [float(np.std(newIndices))]
            origMetaRow["stats/index/count"] = [epLen]
            origMetaRow["stats/index/q01"] = [float(np.quantile(newIndices, 0.01))]
            origMetaRow["stats/index/q10"] = [float(np.quantile(newIndices, 0.10))]
            origMetaRow["stats/index/q50"] = [float(np.quantile(newIndices, 0.50))]
            origMetaRow["stats/index/q90"] = [float(np.quantile(newIndices, 0.90))]
            origMetaRow["stats/index/q99"] = [float(np.quantile(newIndices, 0.99))]
            origMetaRow["stats/episode_index/min"] = [newEpIdx]
            origMetaRow["stats/episode_index/max"] = [newEpIdx]
            origMetaRow["stats/episode_index/mean"] = [float(newEpIdx)]
            origMetaRow["stats/episode_index/std"] = [0.0]
            origMetaRow["stats/episode_index/count"] = [epLen]
            origMetaRow["stats/episode_index/q01"] = [float(newEpIdx)]
            origMetaRow["stats/episode_index/q10"] = [float(newEpIdx)]
            origMetaRow["stats/episode_index/q50"] = [float(newEpIdx)]
            origMetaRow["stats/episode_index/q90"] = [float(newEpIdx)]
            origMetaRow["stats/episode_index/q99"] = [float(newEpIdx)]

            allNewEpMeta.append(origMetaRow)
            runningIndex += epLen
            newEpIdx += 1

    if allNewRows:
        newDf = pd.concat(allNewRows, ignore_index=True)
        combinedDf = pd.concat([origDf, newDf], ignore_index=True)
    else:
        combinedDf = origDf

    outDataDir = os.path.join(OUTPUT, "data", "chunk-000")
    shutil.rmtree(outDataDir)
    os.makedirs(outDataDir)
    combinedDf.to_parquet(os.path.join(outDataDir, "file-000.parquet"), index=False)

    totalEpisodes = origEpisodes * (1 + MULTIPLIER)
    totalFrames = len(combinedDf)

    info["total_episodes"] = totalEpisodes
    info["total_frames"] = totalFrames
    info["splits"]["train"] = f"0:{totalEpisodes}"
    with open(os.path.join(OUTPUT, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    if allNewEpMeta:
        newEpMetaDf = pd.DataFrame(allNewEpMeta)
        for col in origEpMeta.columns:
            if col in newEpMetaDf.columns:
                newEpMetaDf[col] = newEpMetaDf[col].astype(origEpMeta[col].dtype)
        combinedEpMeta = pd.concat([origEpMeta, newEpMetaDf], ignore_index=True)
    else:
        combinedEpMeta = origEpMeta

    outEpMetaDir = os.path.join(OUTPUT, "meta", "episodes", "chunk-000")
    shutil.rmtree(outEpMetaDir)
    os.makedirs(outEpMetaDir)
    combinedEpMeta.to_parquet(os.path.join(outEpMetaDir, "file-000.parquet"), index=False)

    stats = {}
    for col in ["observation.state", "action"]:
        vals = np.stack(combinedDf[col].values)
        stats[col] = columnStats(vals)
    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        vals = combinedDf[col].values.astype(np.float64)
        stats[col] = columnStats(vals)

    with open(os.path.join(OUTPUT, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print("done")


if __name__ == "__main__":
    main()
