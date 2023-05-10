import numpy as np
import torch
import torchvision
import copy
import os

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

# from torch.autograd import Variable
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import models
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# from imgaug import augmenters as iaa


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


def get_data_transforms():
    torch.manual_seed(0)
    np.random.seed(0)

    data_transforms = {
        'train': [
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ],
        'valid': [
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    }
    return data_transforms


def train_model(model: CNN, device: torch.device, hyper_parameters: dict, dataloaders: dict, output_path: str):
    #########################
    # Load Hyper Parameters #
    #########################
    num_epochs = hyper_parameters.get("num_epochs", 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=hyper_parameters.get("optimizer_lr", 0.01))

    ######################################
    # Data Structures for Saving Results #
    ######################################
    optimal_val_epoch = 0
    optimal_val_accuracy = 0

    x_axis = list(range(num_epochs))
    cnn_graph_data = dict()
    cnn_graph_data["epochs"] = x_axis
    cnn_graph_data["optimal_val_epoch"] = optimal_val_epoch
    cnn_graph_data["optimal_val_accuracy"] = optimal_val_accuracy

    cnn_graph_data["train"] = dict()
    cnn_graph_data["valid"] = dict()
    cnn_graph_data["train"]["loss"] = list()
    cnn_graph_data["valid"]["loss"] = list()
    cnn_graph_data["train"]["accuracy"] = list()
    cnn_graph_data["valid"]["accuracy"] = list()

    #########
    # Train #
    #########
    for epoch in tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                # Set model to training mode
                model.train()
                dataloader = dataloaders["train"]
                dataset_size = len(dataloader.batch_sampler.sampler)
            else:
                # Set model to evaluate mode
                model.eval()
                dataloader = dataloaders["valid"]
                dataset_size = len(dataloader.batch_sampler.sampler)

            running_loss = 0.0
            running_corrects = 0.0

            # Looping over all the dataloader images
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

            # Saving epoch loss and accuracy
            epoch_loss = running_loss / dataset_size
            epoch_accuracy = (running_corrects / dataset_size) * 100
            cnn_graph_data[phase]["loss"].append(epoch_loss)
            cnn_graph_data[phase]["accuracy"].append(epoch_accuracy)

            # Saving the weights of the best epoch
            if epoch_accuracy > optimal_val_accuracy:
                optimal_val_epoch = epoch
                optimal_val_accuracy = epoch_accuracy.item()
                cnn_graph_data["optimal_val_epoch"] = optimal_val_epoch
                cnn_graph_data["optimal_val_accuracy"] = optimal_val_accuracy

                PATH = "model.pth"
                torch.save(copy.deepcopy(model.state_dict()), PATH)

    return cnn_graph_data


def predict(model: CNN, device: torch.device, batch: np.ndarray, input_size: int) -> torch.Tensor:
    preprocess = torchvision.transforms.Compose(
        [
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ]
    )

    # Predict images
    img_tensors = [preprocess(img.astype('uint8')) for img in batch]
    batch_tensor = torch.stack(tensors=img_tensors).to(device)
    batch_output = model(batch_tensor)
    batch_predictions = nn.functional.softmax(batch_output, dim=1)

    return batch_predictions


#######################
# Local Model Testing #
#######################

# Setting device
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# Prepare dataloaders
def get_dataloaders():
    torch.manual_seed(0)
    np.random.seed(0)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
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

    # Training Loaders
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                              sampler=train_sampler, num_workers=2)
    validationloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                   sampler=validation_sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # Printing Dataset Sizes
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
def plot_graph(cnn_graph_data: dict):
    plt.title("CNN Loss:")
    plt.plot(cnn_graph_data["epochs"], cnn_graph_data["train"]["loss"], label="Train")
    plt.plot(cnn_graph_data["epochs"], cnn_graph_data["valid"]["loss"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

    plt.clf()
    plt.title("CNN Accuracy:")
    plt.plot(cnn_graph_data["epochs"], cnn_graph_data["train"]["accuracy"], label="Train")
    plt.plot(cnn_graph_data["epochs"], cnn_graph_data["valid"]["accuracy"], label="Validation")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")

    print("Optimal hyper parameters were found at:")
    print("Epoch:", cnn_graph_data["optimal_val_epoch"])
    print("The Validation Accuracy:", cnn_graph_data["optimal_val_accuracy"])


# Model Training Local Test
def local_training(model, device, hyper_parameters, dataloaders, output_path):
    cnn_graph_data = train_model(
        model=model,
        device=device,
        hyper_parameters=hyper_parameters,
        dataloaders=dataloaders,
        output_path=output_path
    )
    plot_graph(cnn_graph_data=cnn_graph_data)


# Model Testing Local Test
def local_testing(model, device, dataloader):
    correct_count = 0
    all_count = 0
    y_true = list()
    y_predict = list()

    weights_path = "model.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    ########
    # Test #
    ########
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # logits
            outputs = model(inputs)
            preds_prob = nn.functional.softmax(input=outputs, dim=1)
            _, preds = torch.max(input=preds_prob, dim=1)
            # eval labels here numeric (not one-hot)
            correct_pred = torch.eq(labels, preds).cpu()
            correct_count += correct_pred.numpy().sum()
            all_count += len(labels)

            y_true.extend(labels.tolist())
            y_predict.extend(preds.tolist())

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count/all_count) * 100)


def parse_predict(batch_predictions) -> list:
    import dtlpy as dl

    batch_annotations = list()

    for img_prediction in batch_predictions:
        pred_score, high_pred_index = torch.max(img_prediction, 0)
        pred_label = str(high_pred_index.item())
        collection = dl.AnnotationCollection()
        collection.add(annotation_definition=dl.Classification(label=pred_label),
                       model_info={'name': "CNN",
                                   'confidence': pred_score.item(),
                                   'model_id': "local id",
                                   'dataset_id': "local folder"})
        print("Predicted {:1} ({:1.3f})".format(pred_label, pred_score))
        batch_annotations.append(collection)

    return batch_annotations


# Model Predict Local Test
def local_predict(model, device) -> list:
    weights_path = "model.pth"
    model.load_state_dict(torch.load(weights_path))

    input_size = 28
    image_folder = "./test_images"  # Path to the image folder
    image_files = [file for file in os.listdir(image_folder) if file.endswith(".jpg") or file.endswith(".png")]

    # Load and convert images to np.ndarray
    image_list = list()
    for file in image_files:
        image_path = os.path.join(image_folder, file)
        image = Image.open(image_path)
        image_array = np.array(image)
        image_list.append(image_array)

    # Convert predictions to annotations
    batch = np.stack(image_list)
    batch_predictions = predict(model=model, device=device, batch=batch, input_size=input_size)
    batch_annotations = parse_predict(batch_predictions)

    return batch_annotations


def main():
    device = get_device()
    model = CNN(use_dropout=True).to(device=device)

    hyper_parameters = {
        "num_epochs": 20,
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

    # Model Testing
    # local_testing(model, device, testloader)

    # Model Predict
    local_predict(model, device)


if __name__ == "__main__":
    main()
