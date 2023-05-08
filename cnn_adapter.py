# from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms
# import torch.optim as optim
import torch.nn.functional
# import pandas as pd
# import numpy as np
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

    def load(self, local_path: str, **kwargs):
        weights = self.model_entity.configuration.get('weights_filename', 'model.pth')


        # TODO: LOAD MODEL
        if not self.model:
            self.model = cnn_model.CNN(use_dropout=True).to(self.device)

        self.model.load_state_dict(torch.load(weights))
        logger.info("Model loaded successfully")

    def save(self, local_path: str, **kwargs):

        # TODO: SAVE MODEL
        weights = kwargs.get('weights_filename', 'model.pth')
        torch.save(self.model, os.path.join(local_path, weights))
        self.configuration['weights'] = weights
        logger.info("Model saved successfully")

    def train(self, data_path: str, output_path: str, **kwargs):
        ######################
        # Create Dataloaders #
        ######################
        input_size = self.configuration.get('input_size', 256)

        def gray_to_rgb(x):
            return x.convert('RGB')

        data_transforms = {
            'train': [
                iaa.Resize({"height": input_size, "width": input_size}),
                iaa.flip.Fliplr(p=0.5),
                iaa.flip.Flipud(p=0.2),
                torchvision.transforms.ToPILImage(),
                gray_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ],
            'valid': [
                torchvision.transforms.ToPILImage(),
                gray_to_rgb,
                torchvision.transforms.Resize((input_size, input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        }

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
        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_torch),
                       'valid': DataLoader(valid_dataset,
                                           batch_size=batch_size,
                                           collate_fn=collate_torch,
                                           shuffle=True)}

        # TODO: TRAIN MODEL
        logger.info("Model started training")
        hyper_parameters = self.configuration.get('hyper_parameters', None)
        self.model = cnn_model.CNN(output_size=hyper_parameters.get("output_size", 10),
                                   use_dropout=True).to(self.device)
        cnn_model.train_model(model=self.model,
                              device=self.device,
                              hyper_parameters=hyper_parameters,
                              dataloaders=dataloaders,
                              output_path=output_path)

        logger.info("Model trained successfully")

    def predict(self, batch: list, **kwargs):
        # TODO: PREDICT MODEL
        input_size = self.configuration.get('input_size', 256)
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
            logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
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


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'model.pth',
                                                                 'input_size': 256,
                                                                 'hyper_parameters': {
                                                                     "num_epochs": 50,
                                                                     "optimizer_lr": 0.01,
                                                                     "output_size": 10,
                                                                 }},
                                          output_type=dl.AnnotationType.CLASSIFICATION,
                                          )

    # TODO: Very important to add tag
    module = dl.PackageModule.from_entry_point(entry_point='cnn_adapter.py')
    package = project.packages.push(package_name='cnn',
                                    src_path=os.getcwd(),
                                    # description='CNN implemented in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(
                                        type=dl.PackageCodebaseType.GIT,
                                        git_url='https://github.com/OfirDataloopAI/cnn_model_adapter',
                                        git_tag='v4'),
                                    modules=[module],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_HIGHMEM_L,
                                                                        runner_image='gcr.io/viewo-g/modelmgmt/resnet:0.0.7',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
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


def model_creation(package: dl.Package, project: dl.Project):
    labels = list()
    for i in range(10):
        labels.append(str(i))

    dataset = project.datasets.get(dataset_name="MNIST_Dataset")

    model = package.models.create(model_name='cnn',
                                  description='cnn-model for testing',
                                  tags=['pretrained', 'MNIST'],
                                  dataset_id=dataset.id,
                                  scope='public',
                                  status='created',
                                  configuration={'weights_filename': 'model.pth',
                                                 'batch_size': 16,
                                                 'hyper_parameters': {
                                                     "num_epochs": 50,
                                                     "optimizer_lr": 0.01,
                                                     "output_size": 10,
                                                 }},
                                  project_id=project.id,
                                  labels=labels,
                                  )
    return model


if __name__ == "__main__":
    dl.setenv('prod')
    project = dl.projects.get(project_name='Abeer N Ofir Project')
    package_creation(project=project)
    package = project.packages.get(package_name='cnn')
    # package.artifacts.list()
    model_creation(package=package, project=project)

    # Useful:
    #https://github.com/dataloop-ai/pytorch_adapters/blob/mgmt3/resnet_adapter.py
