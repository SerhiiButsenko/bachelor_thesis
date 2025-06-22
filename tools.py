import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm


def calculate_pannuke_accuracy(model, dataloader, device):
    total = 0
    correct = 0
    for images, classes in tqdm(dataloader):
        images, classes = images.to(device), classes.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).int()

        correct += (predictions == classes.int()).sum(dim=1).sum().item()
        total += classes.size(0) * classes.size(1)

    return correct / total