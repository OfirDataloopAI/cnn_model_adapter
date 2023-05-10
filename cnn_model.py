import numpy as np
import torch
import torchvision
import copy

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch


# from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import models
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm
# from imgaug import augmenters as iaa
# import os


# Model define
class CNN(nn.Module):
    def __init__(self, output_size=10, use_dropout=False, use_dropout2d=False):
        super(CNN, self).__init__()

        self.output_size = output_size
        # flags
        self.use_dropout = use_dropout
        self.use_dropout2d = use_dropout2d
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        if self.use_dropout2d:
            self.spatial_dropout = nn.Dropout2d(p=0.2)
        # FC layers - since we use global avg pooling,
        # input to the FC layer = #output_features of the second conv layer
        self.fc1 = nn.Linear(32, 256)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # adaptive_avg_pool2d with output_size=1 = simple global avg pooling
        x = self.conv2(x)
        if self.use_dropout2d:
            x = self.spatial_dropout(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model: CNN, device: torch.device, hyper_parameters: dict, dataloaders: dict, output_path: str):
    #########################
    # Load Hyper Parameters #
    #########################
    num_epochs = hyper_parameters.get("num_epochs", 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters.get("optimizer_lr", 0.01))

    x_axis = list(range(num_epochs))
    CNN_graph_data = dict()
    CNN_graph_data["epochs"] = x_axis
    CNN_graph_data["train"] = dict()
    CNN_graph_data["valid"] = dict()
    CNN_graph_data["train"]["loss"] = list()
    CNN_graph_data["valid"]["loss"] = list()
    CNN_graph_data["train"]["accuracy"] = list()
    CNN_graph_data["valid"]["accuracy"] = list()
    CNN_graph_data["optimal_val_epoch"] = 0
    CNN_graph_data["optimal_val_accuracy"] = 0

    #########
    # Train #
    #########
    optimal_val_epoch = 0
    optimal_val_accuracy = 0

    for epoch in tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                # Set model to training mode
                model.train()

                dataloader = dataloaders["train"]
                dataset_size = len(dataloader.dataset)
            else:
                # Set model to evaluate mode
                model.eval()
                dataloader = dataloaders["valid"]
                dataset_size = len(dataloader.dataset)

            running_loss = 0.0
            running_corrects = 0.0

            for i, data in enumerate(dataloader, start=0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the gradient
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Calculating backward and optimize only during the training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Total loss of the mini batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()

            epoch_loss = running_loss / dataset_size
            epoch_accuracy = (running_corrects / dataset_size) * 100
            CNN_graph_data[phase]["loss"].append(epoch_loss)
            CNN_graph_data[phase]["accuracy"].append(epoch_accuracy)

            if epoch_accuracy > optimal_val_accuracy:
                optimal_val_epoch = epoch
                optimal_val_accuracy = epoch_accuracy.item()
                CNN_graph_data["optimal_val_epoch"] = optimal_val_epoch
                CNN_graph_data["optimal_val_accuracy"] = optimal_val_accuracy

                PATH = "model.pth"
                torch.save(copy.deepcopy(model.state_dict()), PATH)

    return CNN_graph_data


def predict(model: CNN, device: torch.device, batch: np.ndarray, input_size: int):
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ]
    )
    img_tensors = [preprocess(img.astype('uint8')) for img in batch]
    batch_tensor = torch.stack(tensors=img_tensors).to(device)
    batch_output = model(batch_tensor)
    batch_predictions = nn.functional.softmax(batch_output, dim=1)

    return batch_predictions


#######################
# Local Model Testing #
#######################

# Setting device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    return device


# Prepare dataloaders
def get_dataloaders():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    # Number of Training images
    N = 30000
    batch_size = 128

    # Datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    # Train and Validation split
    train_size = len(trainset)
    train_idx = np.arange(train_size)
    train_subset_idx = np.random.choice(train_idx, N)
    train_subset_idx, val_subset_idx = train_test_split(train_subset_idx,
                                                        test_size=0.2,
                                                        random_state=0)

    # Create samplers
    train_sampler = SubsetRandomSampler(train_subset_idx)
    validation_sampler = SubsetRandomSampler(val_subset_idx)

    mean = [0.4914, 0.4822, 0.4465]
    standard_deviation = [0.2471, 0.2435, 0.2616]

    # Training Loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=train_sampler, num_workers=2)
    validationloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=validation_sampler, num_workers=2)

    # MNIST Testing Loader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    trainset_size = len(train_subset_idx)
    validset_size = len(val_subset_idx)
    testset_size = len(testset)


    print('='*25)
    print('Train dataset:', trainset_size)
    print('Validation dataset:', validset_size)
    print('Test dataset:', testset_size)
    print('='*25)

    return trainloader, validationloader, testloader


# Plot graph
def plot_graph(CNN_graph_data: dict):
    plt.title("CNN Loss:")
    plt.plot(CNN_graph_data["epochs"], CNN_graph_data["train"]["loss"], label="Train")
    plt.plot(CNN_graph_data["epochs"], CNN_graph_data["valid"]["loss"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

    plt.title("CNN Accuracy:")
    plt.plot(CNN_graph_data["epochs"], CNN_graph_data["train"]["accuracy"], label="Train")
    plt.plot(CNN_graph_data["epochs"], CNN_graph_data["valid"]["accuracy"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")

    print("Optimal hyper parameters were found at:")
    print("Epoch:", CNN_graph_data["optimal_val_epoch"])
    print("The Validation Accuracy:", CNN_graph_data["optimal_val_accuracy"])


def local_training(model, device, hyper_parameters, dataloaders, output_path):
    CNN_graph_data = train_model(
        model=model,
        device=device,
        hyper_parameters=hyper_parameters,
        dataloaders=dataloaders,
        output_path=output_path
    )
    plot_graph(CNN_graph_data=CNN_graph_data)


def local_predict(model, device, testloader):
    PATH = "model.pth"
    model.load_state_dict(torch.load(PATH))
    for batch in testloader:
        print(predict(model=model, device=device, batch=batch, input_size=10))


def main():
    device = get_device()
    model = CNN(use_dropout=True).to(device=device)

    hyper_parameters = {
        "num_epochs": 5,
        "optimizer_lr": 0.01,
        "output_size": 10,
    }
    trainloader, validationloader, testloader = get_dataloaders()
    dataloaders = {
        "train": trainloader,
        "valid": validationloader
    }
    output_path = "."

    # Model Training
    # local_training(model, device, hyper_parameters, dataloaders, output_path)

    # Model Predict
    local_predict(model, device, testloader)


if __name__ == "__main__":
    main()

# def validate_model(model, dataloader, criterion):
#     model.eval()  # Set model to evaluate mode
#     running_loss = 0
#     running_corrects = 0
#
#     with torch.no_grad():
#         for i, data in enumerate(dataloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             # mode to device/cuda
#             inputs, labels = inputs.to(device), labels.to(device)
#             logits = model(inputs)
#             _, preds = torch.max(logits, 1)
#
#             # Cross-entropy loss
#             loss = criterion(logits, labels)
#
#             # statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#
#     epoch_loss = running_loss / validset_size
#     epoch_accuracy = ((running_corrects.double() / validset_size) * 100).item()
#
#     return epoch_loss, epoch_accuracy
#
#
# for epoch in tqdm(range(num_epochs)):
#     train_loss, train_accuracy = train_epoch(CNN_model, trainloader, CNN_criterion, CNN_optimizer)
#     val_loss, val_accuracy = val_epoch(CNN_model, validationloader, CNN_criterion)
#
#     CNN_graph_data["train"]["loss"].append(train_loss)
#     CNN_graph_data["valid"]["loss"].append(val_loss)
#     CNN_graph_data["train"]["accuracy"].append(train_accuracy)
#     CNN_graph_data["valid"]["accuracy"].append(val_accuracy)
#
#     if val_accuracy > optimal_val_accuracy:
#         optimal_val_epoch = epoch
#         optimal_val_accuracy = val_accuracy
#         PATH = "./cnn_parameters.pth"
#         torch.save(copy.deepcopy(CNN_model.state_dict()), PATH)
#
#
# plt.title("CNN Loss:")
# plt.plot(x_axis, CNN_graph_data["train"]["loss"], label="Train")
# plt.plot(x_axis, CNN_graph_data["valid"]["loss"], label="Validation")
# plt.xlabel("Number of epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# plt.title("CNN Accuracy:")
# plt.plot(x_axis, CNN_graph_data["train"]["accuracy"], label="Train")
# plt.plot(x_axis, CNN_graph_data["valid"]["accuracy"], label="Validation")
# plt.xlabel("Number of epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# print("Optimal hyper parameters were found at:")
# print("Epoch:", optimal_val_epoch)
# print("The Validation Accuracy:", optimal_val_accuracy)
#
# PATH = "./cnn_parameters.pth"
# CNN_model.load_state_dict(torch.load(PATH))
