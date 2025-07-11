import torch

class PanNukeMultiLabelClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, labels, transforms=None):
        self.base_dataset = base_dataset
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _, _ = self.base_dataset.__getitem__(idx)
        image = image.float() / 255.0
        if self.transforms is not None:
           image = self.transforms(image)
        label = self.labels[idx].float()
        return image, label