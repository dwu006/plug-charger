import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

IMG_H = 224
IMG_W = 224


class Dataset(TorchDataset):
    def __init__(self, features, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize=4):
        self.features = features
        self.states = states
        self.actions = actions
        self.episodeIndices = episodeIndices
        self.seqLen = seqLen
        self.epStart = epStart
        self.epEnd = epEnd
        self.chunkSize = chunkSize

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        ep = self.episodeIndices[i]
        start = self.epStart[ep]
        end = self.epEnd[ep]
        featSeq = np.empty((self.seqLen, self.features.shape[1]), dtype=np.float32)
        stateSeq = np.empty((self.seqLen, self.states.shape[1]), dtype=np.float32)
        for t in range(self.seqLen):
            j = max(i - (self.seqLen - 1 - t), start)
            featSeq[t] = self.features[j]
            stateSeq[t] = self.states[j]
        actionChunk = np.empty((self.chunkSize, self.actions.shape[1]), dtype=np.float32)
        for k in range(self.chunkSize):
            j = min(i + k, end)
            actionChunk[k] = self.actions[j]
        return torch.from_numpy(featSeq), torch.from_numpy(stateSeq), torch.from_numpy(actionChunk)


class ImageDataset(TorchDataset):
    def __init__(self, images, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize=4):
        self.images = images
        self.states = states
        self.actions = actions
        self.episodeIndices = episodeIndices
        self.seqLen = seqLen
        self.epStart = epStart
        self.epEnd = epEnd
        self.chunkSize = chunkSize

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        ep = self.episodeIndices[i]
        start = self.epStart[ep]
        end = self.epEnd[ep]
        imgSeq = np.empty((self.seqLen, 4, IMG_H, IMG_W), dtype=np.float32)
        stateSeq = np.empty((self.seqLen, self.states.shape[1]), dtype=np.float32)
        for t in range(self.seqLen):
            j = max(i - (self.seqLen - 1 - t), start)
            imgSeq[t] = self.images[j]
            stateSeq[t] = self.states[j]
        actionChunk = np.empty((self.chunkSize, self.actions.shape[1]), dtype=np.float32)
        for k in range(self.chunkSize):
            j = min(i + k, end)
            actionChunk[k] = self.actions[j]
        return torch.from_numpy(imgSeq), torch.from_numpy(stateSeq), torch.from_numpy(actionChunk)


def loadImageDataset(dataRoot, batchSize=32, shuffle=True, numWorkers=4, seqLen=8, chunkSize=4):
    dataPath = os.path.join(dataRoot, "data", "chunk-000")
    parquetFiles = sorted(os.path.join(dataPath, f) for f in os.listdir(dataPath))
    df = pd.concat([pd.read_parquet(f) for f in parquetFiles], ignore_index=True)

    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)

    normPath = os.path.join(dataRoot, "norm_stats.npz")
    stats = np.load(normPath)
    states = (states - stats["state_mean"]) / stats["state_std"]
    actions = (actions - stats["action_mean"]) / stats["action_std"]
    episodeIndices = df["episode_index"].tolist()

    epStart = {}
    epEnd = {}
    for i, ep in enumerate(episodeIndices):
        if ep not in epStart:
            epStart[ep] = i
        epEnd[ep] = i

    framesCache = os.path.join(dataRoot, "frames_cache.npy")
    images = np.load(framesCache)

    allIndices = list(range(len(states)))
    allEps = sorted(set(episodeIndices))
    valEps = set(allEps[-max(1, len(allEps) // 7):])
    trainIdx = [i for i in allIndices if episodeIndices[i] not in valEps]
    valIdx = [i for i in allIndices if episodeIndices[i] in valEps]

    trainSet = ImageDataset(images, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize)
    valSet = ImageDataset(images, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize)
    trainLoader = DataLoader(torch.utils.data.Subset(trainSet, trainIdx), batch_size=batchSize,
                             shuffle=shuffle, num_workers=numWorkers, pin_memory=True)
    valLoader = DataLoader(torch.utils.data.Subset(valSet, valIdx), batch_size=batchSize,
                           shuffle=False, num_workers=numWorkers, pin_memory=True)
    return trainLoader, valLoader, len(states)


def loadDataset(dataRoot, cnn, batchSize=32, shuffle=True, numWorkers=4, seqLen=8, chunkSize=4):
    dataPath = os.path.join(dataRoot, "data", "chunk-000")
    parquetFiles = sorted(os.path.join(dataPath, f) for f in os.listdir(dataPath))
    df = pd.concat([pd.read_parquet(f) for f in parquetFiles], ignore_index=True)

    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)

    normPath = os.path.join(dataRoot, "norm_stats.npz")
    if not os.path.exists(normPath):
        stateMean = states.mean(axis=0)
        stateStd = states.std(axis=0) + 1e-6
        actionMean = actions.mean(axis=0)
        actionStd = actions.std(axis=0) + 1e-6
        np.savez(normPath, state_mean=stateMean, state_std=stateStd,
                 action_mean=actionMean, action_std=actionStd)
    else:
        stats = np.load(normPath)
        stateMean, stateStd = stats["state_mean"], stats["state_std"]
        actionMean, actionStd = stats["action_mean"], stats["action_std"]

    states = (states - stateMean) / stateStd
    actions = (actions - actionMean) / actionStd

    episodeIndices = df["episode_index"].tolist()
    frameIndices = df["frame_index"].tolist()

    epStart = {}
    epEnd = {}
    for i, ep in enumerate(episodeIndices):
        if ep not in epStart:
            epStart[ep] = i
        epEnd[ep] = i

    featuresCache = os.path.join(dataRoot, "features_cache.npy")
    cnnCache = os.path.join(dataRoot, "cnn_cache.pth")
    framesCache = os.path.join(dataRoot, "frames_cache.npy")

    if os.path.exists(featuresCache):
        features = np.load(featuresCache)
        if os.path.exists(cnnCache):
            cnn.load_state_dict(torch.load(cnnCache, map_location=next(cnn.parameters()).device))
    else:
        if os.path.exists(framesCache):
            images = np.load(framesCache)
        else:
            rgbPath = os.path.join(dataRoot, "videos", "observation.images.rgb")
            depthPath = os.path.join(dataRoot, "videos", "observation.images.depth")
            lookup = {(e, f): i for i, (e, f) in enumerate(zip(episodeIndices, frameIndices))}
            images = np.zeros((len(states), 4, IMG_H, IMG_W), dtype=np.float32)
            episodes = sorted(set(episodeIndices))

            def loadEpisode(ep):
                rgbFrames = readAllFrames(os.path.join(rgbPath, f"episode_{ep:06d}.mp4"))
                depthFrames = readAllFrames(os.path.join(depthPath, f"episode_{ep:06d}.mp4"))
                result = []
                for fIdx, (rgb, depth) in enumerate(zip(rgbFrames, depthFrames)):
                    row = lookup.get((ep, fIdx))
                    if row is not None:
                        result.append((row, rgb / 255.0, depth[0] / 255.0))
                return result

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(loadEpisode, ep) for ep in episodes]
                for i, (ep, fut) in enumerate(zip(episodes, futures)):
                    for row, rgb, depth in fut.result():
                        images[row, :3] = rgb
                        images[row, 3] = depth

            np.save(framesCache, images)

        device = next(cnn.parameters()).device
        cnn.eval()
        features = np.zeros((len(states), 512), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(states), 256):
                end = min(start + 256, len(states))
                batch = torch.from_numpy(images[start:end]).to(device)
                features[start:end] = cnn(batch).cpu().numpy()

        np.save(featuresCache, features)
        torch.save(cnn.state_dict(), cnnCache)

    allIndices = list(range(len(states)))
    allEps = sorted(set(episodeIndices))
    valEps = set(allEps[-max(1, len(allEps) // 7):])
    trainIdx = [i for i in allIndices if episodeIndices[i] not in valEps]
    valIdx = [i for i in allIndices if episodeIndices[i] in valEps]

    trainSet = Dataset(features, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize)
    valSet = Dataset(features, states, actions, episodeIndices, seqLen, epStart, epEnd, chunkSize)
    trainLoader = DataLoader(torch.utils.data.Subset(trainSet, trainIdx), batch_size=batchSize,
                             shuffle=shuffle, num_workers=numWorkers, pin_memory=True)
    valLoader = DataLoader(torch.utils.data.Subset(valSet, valIdx), batch_size=batchSize,
                           shuffle=False, num_workers=numWorkers, pin_memory=True)
    return trainLoader, valLoader, len(states)


def readAllFrames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.transpose(2, 0, 1).astype(np.float32))
    cap.release()
    return frames
