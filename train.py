import sys
import torch
import torch.nn as nn
from policy import Policy
from load_dataset import loadDataset

DATA_ROOT = "data"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 200
SAVE_PATH = "model.pth"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, nFrames = loadDataset(DATA_ROOT, batchSize=BATCH_SIZE)
    print(f"loaded {nFrames} frames")

    model = Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        epochLoss = 0.0
        for images, states, actions in dataloader:
            images = images.to(device)
            actions = actions.to(device)

            pred = model(images)
            loss = torch.nn.functional.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

        scheduler.step()
        print(f"epoch={epoch:3d}/{EPOCHS}  loss={epochLoss/len(dataloader):.3f}  lr={scheduler.get_last_lr()[0]:.6f}")
    torch.save(model.state_dict(), SAVE_PATH)
    print("training done")

if __name__ == "__main__":
    train()
