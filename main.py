import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from train_model import train_model
from torchvision import transforms
import argparse

def main(args):
    #model = torchvision.models.alexnet(pretrained=False)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=6)
    learning_rate = 0.0005
    num_epochs = 500
    patience = 50
    mean = torch.tensor([0.7522, 0.5992, 0.7306])
    std = torch.tensor([0.1833, 0.2118, 0.1515])
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # RandomRotate90(),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=mean, std=std),
        #transforms.ToTensor()
    ])

    #data_path = './data/pannuke'
    #labels_path = './data/pannuke_classification_labels'

    train_model(args.data_path, args.labels_path, model, learning_rate, num_epochs, patience, transform)

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

parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the ground truth labels")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")

    args = parser.parse_args()
    main(args)


