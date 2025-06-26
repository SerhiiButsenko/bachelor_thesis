import torch
import torch.nn as nn
import torchvision
import tools
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm

# 1. Load and prepare data
pannuke = PanNukeDataModule(
    data_dir="./data/pannuke/",
    download=False,
    nucleus_type_labels=True,
    batch_size=32,
    hovernet_preprocess=False,
    split=1)

test_labels = torch.load("data/pannuke_classification_labels/test_labels.pt")
test_dataset = PanNukeMultiLabelClassificationDataset(pannuke.test_dataloader.dataset, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 2. Load and prepare model
model = torchvision.models.alexnet()
model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=6)
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#model.load_state_dict(torch.load('alexnet_pannuke.pth', map_location=torch.device('cpu')))
device = torch.device("mps")
model = model.to(device)
model.eval()

criterion = nn.BCEWithLogitsLoss()


with torch.no_grad():
    accuracy = 100 * tools.calculate_pannuke_accuracy(model, test_dataloader, device)
    print(f'Model: AlexNet, Training Set: PanNuke')
    print(f'Accuracy: {accuracy:.4f}')