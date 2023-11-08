import dtlpy as dl
import os
import re
import json
import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

import cnn_model
import deployment_parameters

logger = logging.getLogger('cnn-adapter')


@dl.Package.decorators.module(name="model-adapter",
                              description="CNN Adapter for CNN Example Model",
                              init_inputs={"model_entity": dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """
    CNN adapter using pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity=None):
        if not isinstance(model_entity, dl.Model):
            if isinstance(model_entity, str):
                model_entity = dl.models.get(model_id=model_entity)
            if isinstance(model_entity, dict) and "model_id" in model_entity:
                model_entity = dl.models.get(model_id=model_entity["model_id"])

        # TODO: INIT MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(ModelAdapter, self).__init__(model_entity=model_entity)

    def load(self, local_path: str, **kwargs):
        # TODO: LOAD MODEL
        weights_filename = self.model_entity.configuration.get("weights_filename", "model.pth")
        weights_filepath = os.path.join(local_path, weights_filename)
        output_size = len(self.model_entity.id_to_label_map)

        if not os.path.isfile(weights_filepath):
            logger.warning(f'Weights path ({weights_filepath}) not found! loading default model weights')
            weights_filepath = weights_filename

        self.model = cnn_model.CNN(output_size=output_size, use_dropout=True).to(self.device)
        self.model.load_state_dict(torch.load(f=weights_filepath, map_location=self.device.type))
        info = "Weights got loaded from path: {}".format(weights_filepath)
        # print(info)
        logging.info(info)
        logger.info("Model loaded successfully")

    def save(self, local_path: str, **kwargs):
        # TODO: SAVE MODEL
        weights_filename = kwargs.get("weights_filename", "model.pth")
        self.model_entity.artifacts.upload(os.path.join(local_path, "*"))
        self.configuration.update({"weights_filename": weights_filename})
        info = "Weights got saved to path: {}".format(os.path.join(local_path, weights_filename))
        # print(info)
        logging.info(info)
        logger.info("Model saved successfully")

    def train(self, data_path: str, output_path: str, **kwargs):
        # Reset model for training
        self.configuration["id_to_label_map"] = self.model_entity.id_to_label_map
        output_size = len(self.model_entity.id_to_label_map)
        self.model = cnn_model.CNN(output_size=output_size, use_dropout=True).to(self.device)

        # Print configuration
        # print("Model Configuration:\n{}".format(self.configuration))
        logger.info("Model Configuration:\n{}".format(self.configuration))

        ######################
        # Create Dataloaders #
        ######################
        dataloader_option = self.configuration.get("dataloader_option", "custom")
        dataloaders = self.get_dataloaders(data_path=data_path, dataloader_option=dataloader_option)

        # TODO: TRAIN MODEL
        logger.info("Model started training")
        hyper_parameters = self.configuration.get("hyper_parameters", None)
        train_results = cnn_model.train_model(model=self.model,
                                              device=self.device,
                                              hyper_parameters=hyper_parameters,
                                              dataloaders=dataloaders,
                                              output_path=output_path,
                                              dataloader_option=dataloader_option)

        # TODO: Save weights to artifacts
        samples = list()
        for epoch in train_results["epochs"]:
            samples.append(dl.PlotSample(
                figure="Loss", legend="train loss", x=epoch, y=train_results["train"]["loss"][epoch])
            )
            samples.append(dl.PlotSample(
                figure="Loss", legend="valid loss", x=epoch, y=train_results["valid"]["loss"][epoch])
            )
            samples.append(dl.PlotSample(
                figure="Accuracy", legend='train accuracy', x=epoch, y=train_results["train"]["accuracy"][epoch])
            )
            samples.append(dl.PlotSample(
                figure="Accuracy", legend='valid accuracy', x=epoch, y=train_results["valid"]["accuracy"][epoch])
            )

        self.model_entity.metrics.create(samples=samples, dataset_id=self.model_entity.dataset_id)
        logger.info("Model trained successfully")

    def predict(self, batch: np.ndarray, **kwargs):
        # TODO: PREDICT MODEL
        input_size = self.configuration.get("input_size", 28)
        batch_predictions = cnn_model.predict(model=self.model, device=self.device, batch=batch, input_size=input_size)
        batch_annotations = list()

        for img_prediction in batch_predictions:
            predict_score, highest_predict_index = torch.max(img_prediction, 0)
            predict_label = self.model_entity.id_to_label_map.get(int(highest_predict_index.item()), "UNKNOWN")
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=predict_label),
                           model_info={"name": self.model_entity.name,
                                       "confidence": predict_score.item(),
                                       "model_id": self.model_entity.id,
                                       "dataset_id": self.model_entity.dataset_id})
            logger.debug("Predicted {:1} ({:1.3f})".format(predict_label, predict_score))
            batch_annotations.append(collection)

        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """
        Convert Dataloop structure data to model structured
        Virtual method - need to implement
        e.g. take dlp dir structure and construct annotation file
        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...

    # dataloader_option="custom"
    def custom_dataloaders(self, data_path: str):
        def get_image_filepaths(directory):
            image_filepaths = list()
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_filepaths.append(os.path.join(root, file))

            return image_filepaths

        def get_json_filepaths(directory):
            json_filepaths = list()
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.json'):
                        json_filepaths.append(os.path.join(root, file))

            return json_filepaths

        def convert_image_filepaths_to_arrays(image_filepaths):
            image_list = list()
            for image_filepath in image_filepaths:
                image = Image.open(fp=image_filepath)
                image_array = np.array(image)
                image_array = image_array.astype(float)
                image_list.append(image_array)
                image.close()

            return image_list

        def convert_json_filepaths_to_labels(json_filepaths):
            label_list = list()
            for json_filepath in json_filepaths:
                json_file = open(file=json_filepath, mode="r")
                json_data = json.load(fp=json_file)
                label = int(json_data["annotations"][0]["label"])
                label_list.append(label)

            return label_list

        # Get data directories
        train_directory = os.path.join(data_path, "train")
        valid_directory = os.path.join(data_path, "validation")

        # Get all filepaths
        train_image_filepaths = get_image_filepaths(train_directory)
        train_json_files = get_json_filepaths(train_directory)
        valid_image_filepaths = get_image_filepaths(valid_directory)
        valid_json_files = get_json_filepaths(valid_directory)

        # Sorting files
        def sort_regex(x):
            ext = x.split('.')[-1]
            return int(re.search(r'img_(\w+).{}'.format(ext), x).group(1))

        train_image_filepaths.sort(key=lambda x: sort_regex(x))
        train_json_files.sort(key=lambda x: sort_regex(x))
        valid_image_filepaths.sort(key=lambda x: sort_regex(x))
        valid_json_files.sort(key=lambda x: sort_regex(x))

        # Extracting data
        train_image_data = convert_image_filepaths_to_arrays(image_filepaths=train_image_filepaths)
        train_image_labels = convert_json_filepaths_to_labels(json_filepaths=train_json_files)
        valid_image_data = convert_image_filepaths_to_arrays(image_filepaths=valid_image_filepaths)
        valid_image_labels = convert_json_filepaths_to_labels(json_filepaths=valid_json_files)

        # Custom Dataset Creation
        class CustomDataset(Dataset):
            def __init__(self, data_list, labels_list, transforms_list):
                self.dataset = [(data, label) for data, label in zip(data_list, labels_list)]
                self.length = len(self.dataset)
                self.transforms = transforms_list

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                image, label = self.dataset[idx]
                for transform in self.transforms:
                    image = transform(image)

                return image, label

        input_size = self.configuration.get("input_size", 28)
        batch_size = self.configuration.get("batch_size", 16)
        data_transforms = cnn_model.get_data_transforms(input_size=input_size)
        train_dataset = CustomDataset(
            data_list=train_image_data,
            labels_list=train_image_labels,
            transforms_list=data_transforms["train"]
        )
        valid_dataset = CustomDataset(
            data_list=valid_image_data,
            labels_list=valid_image_labels,
            transforms_list=data_transforms["valid"]
        )

        dataloaders = {
            "train": torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2),
            "valid": torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=2)
        }
        return dataloaders

    # dataloader_option="dataloop"
    def dataloop_dataloader(self, data_path: str):
        input_size = self.configuration.get("input_size", 28)
        batch_size = self.configuration.get("batch_size", 16)
        data_transforms = cnn_model.get_data_transforms(input_size=input_size)

        train_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, "train"),
            dataset_entity=self.model_entity.dataset,
            annotation_type=dl.AnnotationType.CLASSIFICATION,
            transforms=data_transforms["train"],
            id_to_label_map=self.model_entity.id_to_label_map,
            class_balancing=True
        )

        valid_dataset = DatasetGeneratorTorch(
            data_path=os.path.join(data_path, "validation"),
            dataset_entity=self.model_entity.dataset,
            annotation_type=dl.AnnotationType.CLASSIFICATION,
            transforms=data_transforms["valid"],
            id_to_label_map=self.model_entity.id_to_label_map
        )
        dataloaders = {
            "train": DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_torch
            ),
            "valid": DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                collate_fn=collate_torch,
                shuffle=True
            )
        }
        return dataloaders

    def get_dataloaders(self, data_path, dataloader_option: str = "custom"):
        dataloader_options = {
            "custom": self.custom_dataloaders,
            "dataloop": self.dataloop_dataloader
        }
        return dataloader_options[dataloader_option](data_path=data_path)


####################
# Package Creation #
####################
def package_creation(project: dl.Project, package_name: str):
    module = dl.PackageModule.from_entry_point(entry_point="model_adapter.py")

    # Default Hyper Parameters
    default_configuration = {
        "weights_filename": "model.pth",
        "dataloader_option": "custom",
        "batch_size": 16,
        "input_size": 28,
        "hyper_parameters": {
            "num_epochs": 50,
            "optimizer_lr": 0.01
        }
    }

    # Service Kubernetes Parameters
    # TODO: Verify that the correct image is in use
    service_config = {
        "runtime": dl.KubernetesRuntime(
            pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
            runner_image="gcr.io/viewo-g/modelmgmt/resnet:0.0.7",
            autoscaler=dl.KubernetesRabbitmqAutoscaler(
                min_replicas=0,
                max_replicas=1
            ),
            concurrency=1
        ).to_json()
    },

    # Package Metadata
    metadata = dl.Package.get_ml_metadata(
        cls=ModelAdapter,
        default_configuration=default_configuration,
        output_type=dl.AnnotationType.CLASSIFICATION
    )

    # Package Creation
    package = project.packages.push(
        package_name=package_name,
        src_path=os.getcwd(),
        is_global=False,
        package_type="ml",
        codebase=dl.GitCodebase(
            type=dl.PackageCodebaseType.GIT,
            git_url=deployment_parameters.git_url,
            git_tag=deployment_parameters.git_tag
        ),
        modules=[module],
        service_config=service_config,
        metadata=metadata
    )

    return package


##################
# Model Creation #
##################
def dql_filters():
    train_filter = dl.Filters()
    validation_filter = dl.Filters()

    train_paths = ["/training/{}".format(i) for i in range(10)]
    validation_paths = ["/validation/{}".format(i) for i in range(10)]

    train_filter.add(field=dl.FiltersKnownFields.DIR,
                     values=train_paths,
                     operator=dl.FiltersOperations.IN.value)
    validation_filter.add(field=dl.FiltersKnownFields.DIR,
                          values=validation_paths,
                          operator=dl.FiltersOperations.IN.value)

    return train_filter, validation_filter


def model_creation(model_name: str, package: dl.Package, project: dl.Project):
    description = "cnn model for MNIST dataset"
    tags = ["pretrained", "MNIST"]
    dataset = project.datasets.get(dataset_name="MNIST_Dataset")

    # Hyper Parameters
    configuration = {
        "weights_filename": "model.pth",
        "dataloader_option": "custom",
        "batch_size": 16,
        "input_size": 28,
        "hyper_parameters": {
            "num_epochs": 50,
            "optimizer_lr": 0.01
        }
    }

    # Labels and Filters
    labels = [str(i) for i in range(10)]
    train_filter, validation_filter = dql_filters()

    # Model Creation
    model = package.models.create(
        model_name=model_name,
        description=description,
        tags=tags,
        dataset_id=dataset.id,
        scope="private",
        status="created",
        configuration=configuration,
        project_id=project.id,
        labels=labels,
        train_filter=train_filter,
        validation_filter=validation_filter
    )

    return model


def main():
    # project_name = "Abeer N Ofir Project"
    project = dl.projects.get(project_name=deployment_parameters.project_name)

    # Package Creation
    package_creation(project=project, package_name=deployment_parameters.package_name)

    # Model Creation
    package = project.packages.get(package_name=deployment_parameters.package_name)
    # model_name = "cnn_model"
    model_creation(model_name=deployment_parameters.model_name, package=package, project=project)


if __name__ == "__main__":
    main()

    # Useful link:
    # https://github.com/dataloop-ai/pytorch_adapters/blob/mgmt3/resnet_adapter.py
