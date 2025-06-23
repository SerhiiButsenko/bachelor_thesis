import torch
import torch.nn as nn
import torchvision
import tools
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    for images, classes in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
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


def train_model(data_path, labels_path, model, learning_rate, num_epochs, patience):

    # 1. Data preparation
    pannuke_module = tools.load_pannuke_dataset(data_path, first_download=False, labels_path=labels_path)

    train_dataloader = tools.create_multilabel_classification_dataloader(pannuke_module, purpose='train', labels_path=labels_path, shuffle=True)
    validation_dataloader = tools.create_multilabel_classification_dataloader(pannuke_module, purpose='valid', labels_path=labels_path, shuffle=True)

    # 2. Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. Select the loss function
    criterion = nn.BCEWithLogitsLoss()

    # 4. Hyperparameters
    best_accuracy = 0
    epochs_without_improvement = 0

    # 5. Define the optimizer for weight and bias updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Define the training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs)

        #6.1 Validation
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