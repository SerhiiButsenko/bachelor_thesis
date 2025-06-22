import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Data preparation

# Random crop — makes the model invariant to object scale and location
# Random horizontal flip — teaches the model symmetry
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageNet('./imagenet', split='val', transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# 2. Define the model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)
vit_model = vit_model.to(device)

# 3. Select the loss function
loss_fn = nn.CrossEntropyLoss()

# 4. Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# 5. Define the optimizer for weight and bias updates
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=learning_rate, weight_decay=0.5)

# 6. Define the training loop
processed_train = 0
for epoch in range(num_epochs):
    vit_model.train(True)
    running_loss = 0.0
    print('t')
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = vit_model(inputs)

        # Calculate the loss
        loss = loss_fn(outputs, labels)

        # Computing new gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        processed_train += inputs.size(0)
        print(inputs.size(0))
        if processed_train % 10000 < inputs.size(0):
            print(f'Epoch: {epoch}, Processed {processed_train} / {len(val_dataloader) * inputs.size(0)} images')

    epoch_loss = running_loss / len(val_dataloader)

torch.save(vit_model.state_dict(), f'vit_val.pth')

# 7. Validate the model
vit_model.eval()
running_loss = 0.0
correct = 0
total = 0
processed_val = 0
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vit_model(inputs)

        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        processed_val += inputs.size(0)
        if processed_val % 10000 < inputs.size(0):
            print(f'Processed {processed_val} / {len(val_dataloader) * inputs.size(0)} images')


accuracy = 100 * correct / total
print(f'Model: ViT, Training Set: Imagenet')
print(f'Accuracy: {accuracy:.4f}')









