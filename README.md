# CIFAR10 Classification

CIFAR10 dataset classification with transfer learning using VGG16 Convolutional Neural Network architecture. Keras, with backend of Theano has been used for this project.

### Instructions:

#### For downloading Theano and Keras

##### Theano
* Please follow this [link](http://deeplearning.net/software/theano/install.html#install) for installing Theano.

##### Keras
* Please follow this [link](https://keras.io/#installation) for installing Keras.
  Or one could directly follow these:

  *To install Keras, cd to the Keras folder and run the install command:*

  ```python
  sudo python setup.py install
  ```

 *One can also install Keras from PyPI:*

  ```python
  sudo pip install keras
  ```


#### For downloading the dataset
* Please refer to this [link](https://www.cs.toronto.edu/~kriz/cifar.html) to download the CIFAR10 dataset.


#### Files
* finetuning.py contains the code with the last Convolutional Block (rest frozen) of VGG16 modified according to the CIFAR10 dataset.
* sop.py contains the code with only the fully connected layers modified.
* sop_cv.py contains the code with 10 fold cross-validation of the above model (sop.py).
