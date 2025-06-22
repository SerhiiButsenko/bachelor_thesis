import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.optim.lr_scheduler import StepLR
import albumentations as A

from pathml.datasets.pannuke import PanNukeDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation
from tqdm import tqdm

n_classes_pannuke = 6
# check the performance based on including/excluding different transformations
transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.5),
        A.MedianBlur(p=0.5, blur_limit=5)],

        additional_targets={f"mask{i}": "mask" for i in range(n_classes_pannuke)})

pannuke = PanNukeDataModule(
    data_dir="./data/pannuke/",
    download=False,
    nucleus_type_labels=True,
    batch_size=8,
    hovernet_preprocess=False,
    split=1)

train_dataloader = pannuke.train_dataloader
valid_dataloader = pannuke.valid_dataloader
test_dataloader = pannuke.test_dataloader

def build_input_for_classification_models(dataloader):
    classes_present = []
    for batch in tqdm(dataloader):
        images, masks, _ = batch
        batch_classes_present = (masks > 0).any(dim=(2, 3)).int()
        classes_present.append(batch_classes_present)
    return torch.cat(classes_present, dim=0)

train_labels = build_input_for_classification_models(train_dataloader)
valid_labels = build_input_for_classification_models(valid_dataloader)
test_labels = build_input_for_classification_models(test_dataloader)

torch.save(train_labels, "./data/pannuke_classification_labels/train_labels.pt")
torch.save(valid_labels, "./data/pannuke_classification_labels/valid_labels.pt")
torch.save(test_labels, "./data/pannuke_classification_labels/test_labels.pt")