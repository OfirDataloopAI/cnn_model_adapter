# Convolutional Neural Network (CNN) with Adapter to Dataloop

![cnn.png](images%2Fcnn.png)

A simple CNN model with an adapter to support deployment and execution of the model in the Dataloop platform.


### The CNN Model:

A simple CNN model with the following architecture:
1. Convolution (in_channels=1, out_channels=64, kernel_size=5)
2. Convolution (in_channels=64, out_channels=32, kernel_size=3)
3. (Optional) Dropout (p=0.2)
4. Fully Connected (in_features=32, out_features=256)
5. (Optional) Dropout (p=0.5)
6. Fully Connected (in_features=256, out_features=output_size)

### Prerequisites:

Deploy MNIST_Dataset to the project.

Add DQL Filters for the train and validation images paths on the Dataloop platform. 

### How to deploy the package:

### How to deploy the model:

### How to test locally:


### Model functions:

`train`
`predict`


### Requirements: 

See [requirements.txt](requirements.txt)

