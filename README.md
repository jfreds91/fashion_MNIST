### Fashion MNIST

Jesse Fredrickson

### Introduction

The purpose of this project is to experiment with getting a maximized classification score (accuracy, f1) on the [fashion MNIST data set](https://github.com/zalandoresearch/fashion-mnist). It is possible to get ~92% accuracy on this data set relatively quickly and easily with just two layers of 3x3 convolutions and a fully connected output layer, but I'm interested in doing better than that.

The fashion MNIST repository readme has a great table of accuracies achieved by different NN architectures, and I will try to learn from those who have already been high achievers.

### Methods
In the ipython n
otebook main.ipynb, I have defined a few functions for training and analyzing neural network performance. The Fashion_MNIST_Competition file is a google colab file containing starter code for building, training, and analyzing a basic but well-performing NN. The rest of the files that I add I expect to be .py files which are to be called by
