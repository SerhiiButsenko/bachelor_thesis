import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm
import albumentations as A


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

    accuracy = (correct / total) if total != 0 else 0.0
    return accuracy

def calculate_mean_and_std(dataloader):

    # Initialize sums
    mean = 0.
    std = 0.
    n_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        n_samples += batch_samples
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples

    mean /= n_samples
    std /= n_samples

    print('mean:', mean)
    print('std:', std)
    return mean, std

def build_input_for_classification_models(dataloader):
    classes_present = []
    for batch in tqdm(dataloader):
        images, masks, _ = batch
        batch_classes_present = (masks > 0).any(dim=(2, 3)).int()
        classes_present.append(batch_classes_present)
    return torch.cat(classes_present, dim=0)

def create_labels(pannuke_module, purpose, labels_path):
    dataloader = getattr(pannuke_module, f'{purpose}_dataloader')
    labels = build_input_for_classification_models(dataloader)
    torch.save(labels, f'{labels_path}/{purpose}_labels.pt')

def load_pannuke_dataset(download_path, first_download, labels_path):
    n_classes_pannuke = 5
    # check the performance based on including/excluding different transformations

    pannuke = PanNukeDataModule(
        data_dir=download_path,
        download=first_download,
        nucleus_type_labels=True,
        batch_size=32,
        hovernet_preprocess=False,
        split=1)

    purposes = ['train', 'valid', 'test']
    for purpose in purposes:
        create_labels(pannuke, purpose, labels_path)

    return pannuke

def create_multilabel_classification_dataloader(pannuke_module, purpose, labels_path, shuffle, transforms):
    labels = torch.load(f'{labels_path}/{purpose}_labels.pt')
    dataloader = getattr(pannuke_module, f'{purpose}_dataloader')
    dataset = PanNukeMultiLabelClassificationDataset(dataloader.dataset, labels, transforms)
    #dataset = torch.utils.data.Subset(dataset, list(range(32)))
    return DataLoader(dataset, batch_size=32, shuffle=shuffle)