import torch
import torch.nn as nn
import torchvision
import tools
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm

# 1. Data preparation

# Random crop — makes the model invariant to object scale and location
# Random horizontal flip — teaches the model symmetry

pannuke = PanNukeDataModule(
    data_dir="./data/pannuke/",
    download=False,
    nucleus_type_labels=True,
    batch_size=32,
    hovernet_preprocess=False,
    split=1)

train_labels = torch.load("data/pannuke_classification_labels/train_labels.pt")
validation_labels = torch.load("data/pannuke_classification_labels/valid_labels.pt")

train_dataset = PanNukeMultiLabelClassificationDataset(pannuke.train_dataloader.dataset, train_labels)
validation_dataset = PanNukeMultiLabelClassificationDataset(pannuke.valid_dataloader.dataset, validation_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

# 2. Define the model
model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=6)
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cpu")
model = model.to(device)

# 3. Select the loss function
criterion = nn.BCEWithLogitsLoss()

# 4. Hyperparameters
learning_rate = 0.01
batch_size = 8
num_epochs = 50
best_accuracy = 0
epochs_without_improvement = 0
patience = 5

# 5. Define the optimizer for weight and bias updates
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 6. Define the training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    processed_train = 0
    for images, classes in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, classes = images.to(device), classes.to(device)

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, classes)

        # Computing new gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        running_loss += loss.item() * classes.size(0)
        processed_train += classes.size(0)
        if processed_train % 10000 < classes.size(0):
            print(f'Epoch: {epoch}, Processed {processed_train} / {len(train_dataloader) * classes.size(0)} images')

    # Validation
    model.eval()
    with torch.no_grad():
        accuracy = 100 * tools.calculate_pannuke_accuracy(model, validation_dataloader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'models/AlexNet-{epoch + 1}.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch + 1}")
                break
