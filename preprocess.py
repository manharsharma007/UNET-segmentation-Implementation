import torch
import torchvision
from dataset import CarvanaData
from torch.utils.data import DataLoader

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
):
    train_ds = CarvanaData(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transforms=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_ds = CarvanaData(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transforms=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f"Got {correct}/{num_pixels} with acc {correct/num_pixels*100:.2f}"
    )
    model.train()
