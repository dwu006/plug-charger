import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

IMG_H = 224
IMG_W = 224

def loadDataset(dataRoot, batchSize=32, shuffle=True, numWorkers=4):
    dataPath = os.path.join(dataRoot, "data", "chunk-000")
    parquetFiles = []
    for fileName in os.listdir(dataPath):
        parquetFiles.append(os.path.join(dataPath, fileName))
    parquetFiles.sort()
    
    dataFrames = []
    for filePath in parquetFiles:
        df = pd.read_parquet(filePath)
        dataFrames.append(df)
    df = pd.concat(dataFrames, ignore_index=True)

    states = []
    for state in df["observation.state"]:
        states.append(state)
    states = np.stack(states).astype(np.float32)
    
    actions = []
    for action in df["action"]:
        actions.append(action)
    actions = np.stack(actions).astype(np.float32)
    
    episodeIndices = []
    for epIdx in df["episode_index"]:
        episodeIndices.append(epIdx)
    
    frameIndices = []
    for frameIdx in df["frame_index"]:
        frameIndices.append(frameIdx)

    rgbPath = os.path.join(dataRoot, "videos", "observation.images.rgb")
    depthPath = os.path.join(dataRoot, "videos", "observation.images.depth")

    cap1 = {}
    ep = set(episodeIndices)
    for e in ep:
        path = os.path.join(rgbPath, f"episode_{e:06d}.mp4")
        cap1[e] = cv2.VideoCapture(path)
    
    cap2 = {}
    for e in ep:
        path = os.path.join(depthPath, f"episode_{e:06d}.mp4")
        cap2[e] = cv2.VideoCapture(path)

    images = np.zeros((len(states), 4, IMG_H, IMG_W), dtype=np.float32)
    for i in range(len(states)):
        e = episodeIndices[i]
        f = frameIndices[i]
        
        rgb = readFrame(cap1[e], f)
        depth = readFrame(cap2[e], f)
        depth = depth[0:1]
        
        combined = np.concatenate([rgb, depth], axis=0)
        images[i] = combined / 255.0

    for cap in cap1.values():
        cap.release()
    for cap in cap2.values():
        cap.release()

    actions = actions/180.0
    states  = states/180.0

    images = torch.from_numpy(images)
    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions)
    dataset = TensorDataset(images, states, actions)
    return DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers), len(states)


def readFrame(cap, frameIdx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (IMG_W, IMG_H))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.transpose(2, 0, 1).astype(np.float32)