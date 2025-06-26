import torchvision
import torch.nn as nn
import torchvision.models as models
from train_model import train_model
from torchvision import transforms
import argparse

def main(args):
    # model = torchvision.models.alexnet(pretrained=True)
    # model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=6)
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=6)
    learning_rate = 0.0005 #typically 1e-3 and 1e-5, most often 1e-4
    num_epochs = 500
    patience = 50

    #data_path = './data/pannuke'
    #labels_path = './data/pannuke_classification_labels'

    train_model(args.data_path, args.labels_path, model, learning_rate, num_epochs, patience)

    models_to_train = {
        'alexnet': models.alexnet(pretrained=False),
        'inception_v3': models.inception_v3(pretrained=False),
        'vgg19': models.vgg19(pretrained=False),
        'efficientnet-b0': models.efficientnet_b0(pretrained=False)
    }

    # mean, std, total
    # for every image in validation set
    #   mean += image.mean(every axis except color axis)
    #   std += image.std(every axis except color axis)
    # mean = mean / total
    # std = std / total

    transforms_for_models = {
        'alexnet':
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7406, 0.5332, 0.7059], std=[0.1601, 0.2133, 0.1538])]),
        'inception_v3':
            transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7406, 0.5332, 0.7059], std=[0.1601, 0.2133, 0.1538])]),
        'vgg19':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7406, 0.5332, 0.7059], std=[0.1601, 0.2133, 0.1538])]),
        'efficientnet-b0':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7406, 0.5332, 0.7059], std=[0.1601, 0.2133, 0.1538])])
    }

parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the ground truth labels")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")

    args = parser.parse_args()
    main(args)


