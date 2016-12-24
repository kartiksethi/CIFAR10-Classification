import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.constraints import maxnorm

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model_saved.h5'
# dimensions of our images.
img_width, img_height = 32, 32

nb_epoch = 50

img_width, img_height = 32, 32
model = Sequential()
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network

    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    # f = h5py.File(weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         # we don't look at the last (fully-connected) layers in the savefile
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()
    # print('Model loaded.')
      
    
save_bottlebeck_features()

def train_top_model():
	np.random.seed()

	train_data = np.load(open('bottleneck_features_train.npy'))
	validation_data = np.load(open('bottleneck_features_validation.npy'))
	# load data
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	# normalize inputs from 0-255 to 0.0-1.0
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = X_train / 255.0
	X_test = X_test / 255.0
	# one hot encode outputs
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]
	top_model = Sequential()
	top_model.add(Flatten(input_shape=train_data.shape[1:]))
	top_model.add(Dense(512, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(512, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(512, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(num_classes, activation='softmax'))
	# top_model.load_weights(top_model_weights_path)
	model.add(top_model)
	for layer in model.layers[:25]:
		layer.trainable = False
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	model.load_weights(top_model_weights_path)
	
	# model.fit(X_train, y_train, 
	#           nb_epoch=nb_epoch, batch_size=64,
	#           validation_data=(X_test, y_test),verbose=True,shuffle=True)
	# model.save_weights(top_model_weights_path)
	print "The accuracy on the test set = " + str(model.evaluate(X_test,y_test,batch_size=64,verbose=True)[1]*100) + "%"


train_top_model()




