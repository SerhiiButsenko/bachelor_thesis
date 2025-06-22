import torchvision
import torch.nn as nn
import torchvision.models as models
from train_model import train_model
from torchvision import transforms


model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=6)
learning_rate = 0.01
num_epochs = 50
patience = 5

train_model(model, learning_rate, num_epochs, patience)

models_to_train = {
    'alexnet': models.alexnet(pretrained=False),
    'inception_v3': models.inception_v3(pretrained=False),
    'vgg19': models.vgg19(pretrained=False),
    'efficientnet-b0': models.efficientnet_b0(pretrained=False)
}

transforms_for_models = {
    'alexnet':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'inception_v3':
        transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'vgg19':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'efficientnet-b0':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])])
}