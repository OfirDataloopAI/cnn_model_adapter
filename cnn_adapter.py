# from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
# import torch.optim as optim
import torch.nn.functional
# import pandas as pd
import numpy as np
import dtlpy as dl
import torch.nn
import logging
import torch
# import time
# import copy
# import tqdm
import os

from torch.utils.data import DataLoader
from dtlpy.utilities.dataset_generators.dataset_generator import collate_torch
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch

import cnn_model

logger = logging.getLogger('cnn-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='CNN Adapter for CNN Example Model',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):
    """
    CNN adapter using pytorch.
    The class bind Dataloop model and model entities with model code implementation
    """

    def __init__(self, model_entity=None):
        if not isinstance(model_entity, dl.Model):
            if isinstance(model_entity, str):
                model_entity = dl.models.get(model_id=model_entity)
            if isinstance(model_entity, dict) and 'model_id' in model_entity:
                model_entity = dl.models.get(model_id=model_entity['model_id'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: GET MODEL
        self.model = cnn_model.CNN(use_dropout=True).to(self.device)
        super(ModelAdapter, self).__init__(model_entity=model_entity)
        logger.info("Model init completed")

    def load(self, local_path: str, **kwargs):
        weights_filename = self.model_entity.configuration.get('weights_filename', 'model.pth')

        # TODO: LOAD MODEL - CURRENTLY WORK WITH GPU ONLY
        if not self.model:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = cnn_model.CNN(use_dropout=True).to(self.device)

        self.model.load_state_dict(torch.load(weights_filename))
        logger.info("Model loaded successfully")

    def save(self, local_path: str, **kwargs):
        # TODO: SAVE MODEL
        weights_filename = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights_filename))
        self.configuration['weights_filename'] = weights_filename
        logger.info("Model saved successfully")

    def train(self, data_path: str, output_path: str, **kwargs):
        if not self.model:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = cnn_model.CNN(use_dropout=True).to(self.device)

        ######################
        # Create Dataloaders #
        ######################
        input_size = self.configuration.get('input_size', 28)
        data_transforms = cnn_model.get_data_transforms(input_size=input_size)

        train_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'train'),
                                              dataset_entity=self.model_entity.dataset,
                                              annotation_type=dl.AnnotationType.CLASSIFICATION,
                                              transforms=data_transforms['train'],
                                              id_to_label_map=self.model_entity.id_to_label_map,
                                              class_balancing=True)

        valid_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'validation'),
                                              dataset_entity=self.model_entity.dataset,
                                              annotation_type=dl.AnnotationType.CLASSIFICATION,
                                              transforms=data_transforms['valid'],
                                              id_to_label_map=self.model_entity.id_to_label_map)

        batch_size = self.configuration.get("batch_size", 16)
        dataloaders = {'train': DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_torch),
                       'valid': DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_torch,
                                           shuffle=True)}

        # TODO: TRAIN MODEL
        logger.info("Model started training")
        hyper_parameters = self.configuration.get('hyper_parameters', None)
        labels_count = self.configuration.get('id_to_label_map', None)
        if labels_count:
            output_size = len(labels_count)
        else:
            output_size = hyper_parameters.get("output_size", 10)

        self.model = cnn_model.CNN(output_size=output_size,
                                   use_dropout=True).to(self.device)
        train_results = cnn_model.train_model(model=self.model,
                                              device=self.device,
                                              hyper_parameters=hyper_parameters,
                                              dataloaders=dataloaders,
                                              output_path=output_path)

        #TODO: Upload metric
        import matplotlib.pyplot as plt

        samples = list()

        # plt.title("CNN Loss:")
        # plt.plot(train_results["epochs"], train_results["train"]["loss"], label="Train")
        # plt.plot(train_results["epochs"], train_results["valid"]["loss"], label="Validation")
        # plt.xlabel("Number of epochs")
        # plt.ylabel("Loss")
        # plt.legend()

        # samples.append(dl.PlotSample(figure="loss",
        #                              legend='train',
        #                              x=train_results["epochs"],
        #                              y=train_results["train"]["loss"]))

        # plt.clf()
        # plt.title("CNN Accuracy:")
        # plt.plot(train_results["epochs"], train_results["train"]["accuracy"], label="Train")
        # plt.plot(train_results["epochs"], train_results["valid"]["accuracy"], label="Validation")
        # plt.xlabel("Number of epochs")
        # plt.ylabel("Accuracy")
        # plt.legend()

        # samples.append(dl.PlotSample(figure="accuracy",
        #                              legend='train',
        #                              x=train_results["epochs"],
        #                              y=train_results["train"]["accuracy"]))




        # num_epochs = hyper_parameters.get("num_epochs", 50)
        # for epoch in range(num_epochs):
        #     for phase in ["'train", "valid"]:
        #         self.model_entity.add_log_samples(samples=dl.LogSample(figure='loss',
        #                                                                legend=phase,
        #                                                                x=epoch,
        #                                                                y=train_results[phase]["loss"][epoch]),
        #                                           dataset_id=self.model_entity.dataset_id)
        #         self.model_entity.add_log_samples(samples=dl.LogSample(figure='accuracy',
        #                                                                legend=phase,
        #                                                                x=epoch,
        #                                                                y=train_results[phase]["accuracy"][epoch]),
        #                                           dataset_id=self.model_entity.dataset_id)

        logger.info("Model trained successfully")

    def predict(self, batch: np.ndarray, **kwargs):
        if not self.model:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = cnn_model.CNN(use_dropout=True).to(self.device)

        # TODO: PREDICT MODEL
        input_size = self.configuration.get('input_size', 28)
        batch_predictions = cnn_model.predict(model=self.model, device=self.device, batch=batch, input_size=input_size)
        batch_annotations = list()

        for img_prediction in batch_predictions:
            pred_score, high_pred_index = torch.max(img_prediction, 0)
            pred_label = self.model_entity.id_to_label_map.get(int(high_pred_index.item()), 'UNKNOWN')
            collection = dl.AnnotationCollection()
            collection.add(annotation_definition=dl.Classification(label=pred_label),
                           model_info={'name': self.model_entity.name,
                                       'confidence': pred_score.item(),
                                       'model_id': self.model_entity.id,
                                       'dataset_id': self.model_entity.dataset_id})
            logger.debug("Predicted {:1} ({:1.3f})".format(pred_label, pred_score))
            batch_annotations.append(collection)

        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured
            Virtual method - need to implement
            e.g. take dlp dir structure and construct annotation file
        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

        ...


####################
# Package Creation #
####################
def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(
        cls=ModelAdapter,
        default_configuration={
            "weights_filename": "model.pth",
            "batch_size": 16,
            "input_size": 28,
            "hyper_parameters": {
                "num_epochs": 50,
                "optimizer_lr": 0.01,
                "output_size": 10,
            }
        },
        output_type=dl.AnnotationType.CLASSIFICATION,
    )

    # TODO: Very important to add tag
    module = dl.PackageModule.from_entry_point(entry_point='cnn_adapter.py')
    package = project.packages.push(
        package_name='cnn',
        src_path=os.getcwd(),
        is_global=False,
        package_type='ml',
        codebase=dl.GitCodebase(
            type=dl.PackageCodebaseType.GIT,
            git_url='https://github.com/OfirDataloopAI/cnn_model_adapter',
            git_tag='v13'
        ),
        modules=[module],
        service_config={
            'runtime': dl.KubernetesRuntime(
                pod_type=dl.INSTANCE_CATALOG_HIGHMEM_L,
                #TODO: check different image
                runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.7',
                autoscaler=dl.KubernetesRabbitmqAutoscaler(
                    min_replicas=0,
                    max_replicas=1
                ),
                concurrency=1
            ).to_json()
        },
        metadata=metadata
    )

    # package.metadata = {'system': {'ml': {'defaultConfiguration': {'weights_filename': 'model.pth',
    #                                                                'input_size': 256},
    #                                       'outputType': dl.AnnotationType.CLASSIFICATION,
    #                                       'tags': ['torch'], }}}
    # package = package.update()
    # s = package.services.list().items[0]
    # s.package_revision = package.version
    # s.versions['dtlpy'] = '1.74.9'
    # s.update(True)
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


def model_creation(package: dl.Package, project: dl.Project):
    labels = [str(i) for i in range(10)]
    dataset = project.datasets.get(dataset_name="MNIST_Dataset")
    train_filter, validation_filter = dql_filters()

    model = package.models.create(
        model_name="cnn",
        description="cnn-model for MNIST dataset",
        tags=['pretrained', 'MNIST'],
        dataset_id=dataset.id,
        scope="public",
        status="created",
        configuration={
            "weights_filename": "model.pth",
            "batch_size": 16,
            "input_size": 28,
            "hyper_parameters": {
                "num_epochs": 50,
                "optimizer_lr": 0.01,
                "output_size": 10,
            }
        },
        project_id=project.id,
        labels=labels,
        train_filter=train_filter,
        validation_filter=validation_filter
    )

    return model


def main():
    project = dl.projects.get(project_name='Abeer N Ofir Project')

    # Package Creation
    package_creation(project=project)

    # Model Creation
    # package = project.packages.get(package_name='cnn')
    # model_creation(package=package, project=project)


if __name__ == "__main__":
    main()

    # Useful link:
    # https://github.com/dataloop-ai/pytorch_adapters/blob/mgmt3/resnet_adapter.py
