import torch
import torch.nn as nn
import torchvision
import tools
from torch.utils.data import DataLoader
from PanNukeMultiLabelClassificationDataset import PanNukeMultiLabelClassificationDataset
from pathml.datasets.pannuke import PanNukeDataModule
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total = 0
    correct = 0
    for images, classes in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, classes = images.to(device), classes.to(device)

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, classes)

        writer.add_scalar("Loss/train", loss, epoch)

        # Computing new gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).int()
        correct += (predictions == classes.int()).sum(dim=1).sum().item()
        total += classes.size(0) * classes.size(1)

    accuracy = (correct / total) if total != 0 else 0.0
    accuracy = accuracy * 100
    writer.add_scalar("Acc/train", accuracy, epoch)


def train_model(data_path, labels_path, model, learning_rate, num_epochs, patience, transforms):
    # 1. Data preparation
    pannuke_module = tools.load_pannuke_dataset(data_path, first_download=False, labels_path=labels_path)

    train_dataloader = tools.create_multilabel_classification_dataloader(pannuke_module, purpose='train', labels_path=labels_path, shuffle=True, transforms=transforms)
    validation_dataloader = tools.create_multilabel_classification_dataloader(pannuke_module, purpose='valid', labels_path=labels_path, shuffle=True, transforms=transforms)

    # 2. Model setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    #tools.calculate_mean_and_std(train_dataloader)

    # 3. Select the loss function
    criterion = nn.BCEWithLogitsLoss()

    # 4. Hyperparameters
    best_accuracy = 0
    epochs_without_improvement = 0

    # 5. Define the optimizer for weight and bias updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # 6. Define the training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs)

        #6.1 Validation
        model.eval()
        with torch.no_grad():
            #accuracy = 100 * tools.calculate_pannuke_accuracy(model, validation_dataloader, device)
            total = 0
            correct = 0
            for images, classes in tqdm(validation_dataloader):
                images, classes = images.to(device), classes.to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).int()
                val_loss = criterion(outputs, classes)
                writer.add_scalar("Loss/validation", val_loss, epoch)

                correct += (predictions == classes.int()).sum(dim=1).sum().item()
                total += classes.size(0) * classes.size(1)

            accuracy = (correct / total) if total != 0 else 0.0
            accuracy = accuracy * 100
            writer.add_scalar("Acc/validation", accuracy, epoch)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'models/{model.__class__.__name__}-{epoch + 1}.pth')
            elif epoch % 50 == 0:
                torch.save(model.state_dict(), f'models/{model.__class__.__name__}-{epoch + 1}.pth')
                epochs_without_improvement += 1
                # if epochs_without_improvement >= patience:
                #     print(f"Stopping early at epoch {epoch + 1}")
                #     break

    print(f"Best Accuracy: {best_accuracy:.4f}")