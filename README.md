# Learning how to build a Convolutional Neural Network

## What is in this repository?

`cnn-design` folder contains the implementation for Convolutional Neural Networks (CNNs) from scratch using NumPy. <br>
The project presents how to create a CNN with three layers: convolution (conv), ReLU and max pooling. The steps are the following:
<ol>
  <li> Reading the input image </li>
  <li> Preparing the filters </li>
  <li> Conv layer: convolving each filter with the input image </li>
  <li> ReLU layer: applying ReLU activation function on the feature maps (output of conv layer) </li>
  <li> Max Pooling layer: applying the pooling operation on the output of ReLU layer </li>
  <li> Stacking conv, ReLU and max pooling layers </li>
</ol>

Code available in the book: "Practical Computer Vision Applications Using Deep Learning with CNNs" and at <a href="https://github.com/ahmedfgad/NumPyCNN"> this GitHub address </a>.

`cnn-classifier` folder contains the implementation (with Tensorflow) of a neural network machine learning model that classifies images, trains the network and evaluates the accuracy of the model. The steps are the following:
<ol>
  <li> Load the dataset (MNIST/)</li>
  <li> Build the model </li>
  <li> Define the optimizer and the loss function  </li>
  <li> Train and evaluate the model </li>
</ol>

Code available also in the <a href="https://www.tensorflow.org/tutorials/quickstart/advanced">Tensorflow </a> documentation.


## Setting up the environment
In the current directory, execute the following commands in order to set up the conda environment:

`conda env create --file environment.yaml`

`conda activate cnn-tf`

`conda install -c conda-forge tensorflow`

