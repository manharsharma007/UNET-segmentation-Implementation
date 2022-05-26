import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensor

from model import UNET
from preprocess import get_loaders, check_accuracy


# HYPERPARAMETERS
EPOCHS = 30
BATCH_SIZE = 2
LR = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"
WORKERS = 2
IMG_W = 160
IMG_H = 240


def train_fn(model, dataloader, optimizer, loss_fn):

    loop = tqdm(dataloader)

    for index, (data, target) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.float().to(device=DEVICE)

        output = model(data)
        loss = loss_fn(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def train():
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_H, width=IMG_W),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensor(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_H, width=IMG_W),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensor(),
        ],
    )

    model = UNET(in_features=3, out_features=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders(
        TRAIN_DIR,
        TRAIN_MASK_DIR,
        VAL_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        WORKERS,
    )

    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(EPOCHS):
        train_fn(model, train_loader, optimizer, loss_fn)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
