import torch

class PanNukeMultiLabelClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, labels):
        self.base_dataset = base_dataset
        self.labels = labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _, _ = self.base_dataset.__getitem__(idx)
        image = image.float() / 255.0
        label = self.labels[idx].float()
        return image, label