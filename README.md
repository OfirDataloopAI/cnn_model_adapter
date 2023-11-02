# Convolutional Neural Network (CNN) with Adapter to Dataloop

![cnn.png](assets%2Fcnn.png)

A simple CNN model with an adapter to support deployment and execution of the model in the Dataloop platform.


### The CNN Model:

A simple CNN model with the following architecture:
1. Convolution (in_channels=1, out_channels=64, kernel_size=5)
2. Convolution (in_channels=64, out_channels=32, kernel_size=3)
3. (Optional) Dropout (p=0.2)
4. Fully Connected (in_features=32, out_features=256)
5. (Optional) Dropout (p=0.5)
6. Fully Connected (in_features=256, out_features=output_size)

### Usage:

A template for how connecting a custom model to the Dataloop AI platform.

#### 1.model script:

The [cnn_model.py](cnn_model.py) is an example of a simple CNN model for MNIST dataset classification.
Replace this script with the requested model to be integrated with dataloop, and make sure to implement the `train` and 
`predict` functions with the same signature so the data from the dataloop platform will be parsed from the model 
correctly.

#### 2.adapter script:

The [model_adapter.py](model_adapter.py) is an example of an adapter implementation with the required functions:
`load`, `save`, `train`, `predict`, `convert_from_dtlpy`.

`load` - A function to load the model with the most recent weights (and artifacts, if there are). This function is 
being called before sending the model for Training and Prediction.

`save` - A function to save the model generated weights (and artifacts, if there are), after the Training is completed.

`train` - A function to call the model train method, and upload the model results as graphs to the platform.

`predict` - A function to call the model predict method, and upload the model results as annotations to the given item.

`convert_from_dtlpy` - (Optional) A function to convert the downloaded dataloop json format to other know json formats.
For example: COCO, VOC, YOLO.

#### 3.tests script:

The [model_adapter_tests.py](model_adapter_tests.py) is a script to evaluate the model performance on the local machine 
before running it on a remote machine. A script for debugging the model performance and would be in the remote machine 
to detect bugs and failures before they occur on the remote machine.  



### Prerequisites:

Deploy MNIST_Dataset to the project.

Add DQL Filters for the train and validation images paths on the Dataloop platform.

Link to MNIST dataset uploader: https://github.com/OfirDataloopAI/mnist_uploader

### How to deploy the package:

From function `package_creation` in [model_adapter.py](model_adapter.py)

### How to deploy the model:

From function `model_creation` in [model_adapter.py](model_adapter.py)

### How to test locally:

From functions `train_test` and `predict_test` in [model_adapter_tests.py](model_adapter_tests.py)

And from function `local_training`, `local_testing` and `local_predict` in [cnn_model.py](cnn_model.py)

### Model functions:

`train`
`predict`


### Requirements: 

See [_requirements.txt](_requirements.txt)

