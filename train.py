import torch
import csv
from loadDataset import loadDataset, loadImageDataset

MODEL = "transformer"
DATA = "data"
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 100
FINETUNE_EPOCHS = 20
FINETUNE_LR = 1e-5
SAVE_PATH = "models/model_transformer.pth"


def evalLoss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for features, states, actions in loader:
            pred = model(features.to(device), states.to(device))
            total += torch.nn.functional.mse_loss(pred, actions.to(device)).item()
    return total / len(loader)


def train():
    device = torch.device("cuda")
    model = Policy().to(device)
    trainLoader, valLoader, nFrames = loadDataset(DATA, model.cnn, batchSize=BATCH_SIZE, seqLen=8)
    print(f"loaded {nFrames} frames")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    trainLosses = []
    valLosses = []
    bestVal = float("inf")
    wait = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        epochLoss = 0.0
        for features, states, actions in trainLoader:
            features = features.to(device)
            states = states.to(device)
            actions = actions.to(device)

            pred = model(features, states)
            mse = torch.nn.functional.mse_loss(pred, actions)
            smoothness = torch.nn.functional.mse_loss(pred[:, 1:], pred[:, :-1])
            loss = mse + 0.1 * smoothness

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epochLoss += loss.item()

        scheduler.step()
        avgTrain = epochLoss / len(trainLoader)
        avgVal = evalLoss(model, valLoader, device)
        trainLosses.append(avgTrain)
        valLosses.append(avgVal)

        tag = ""
        if avgVal < bestVal:
            bestVal = avgVal
            wait = 0
            torch.save(model.state_dict(), SAVE_PATH)
            tag = " *"
        else:
            wait += 1

        print(f"epoch={epoch:3d}/{EPOCHS}  train={avgTrain:.4f}  val={avgVal:.4f}  lr={scheduler.get_last_lr()[0]:.6f}{tag}")

        if wait >= 10:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    model.cnn.finetune()
    ftTrainLoader, ftValLoader, _ = loadImageDataset(DATA, batchSize=BATCH_SIZE // 2, seqLen=8)
    ftOptimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    ftScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ftOptimizer, T_max=FINETUNE_EPOCHS)

    bestFtVal = float("inf")
    wait = 0
    ftTrainLosses = []
    ftValLosses = []

    for epoch in range(1, FINETUNE_EPOCHS+1):
        model.train()
        epochLoss = 0.0
        for images, states, actions in ftTrainLoader:
            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)

            pred = model(images, states)
            mse = torch.nn.functional.mse_loss(pred, actions)
            smoothness = torch.nn.functional.mse_loss(pred[:, 1:], pred[:, :-1])
            loss = mse + 0.1 * smoothness

            ftOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ftOptimizer.step()
            epochLoss += loss.item()

        ftScheduler.step()
        avgTrain = epochLoss / len(ftTrainLoader)
        avgVal = evalLoss(model, ftValLoader, device)
        ftTrainLosses.append(avgTrain)
        ftValLosses.append(avgVal)

        tag = ""
        if avgVal < bestFtVal:
            bestFtVal = avgVal
            wait = 0
            torch.save(model.state_dict(), SAVE_PATH)
            tag = " *"
        else:
            wait += 1

        print(f"ft epoch={epoch:3d}/{FINETUNE_EPOCHS}  train={avgTrain:.6f}  val={avgVal:.6f}  lr={ftScheduler.get_last_lr()[0]:.8f}{tag}")

        if wait >= 10:
            print(f"early stopping at epoch {epoch}")
            break

    with open("loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "phase"])
        for i, (t, v) in enumerate(zip(trainLosses, valLosses), 1):
            w.writerow([i, t, v, "phase1"])
        for i, (t, v) in enumerate(zip(ftTrainLosses, ftValLosses), 1):
            w.writerow([len(trainLosses) + i, t, v, "finetune"])

    print("done")

if __name__ == "__main__":
    train()
