import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
from keras.applications.densenet import DenseNet121
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


def show_metrics(res):
    plt.figure(figsize=(12, 16))

    plt.subplot(4, 2, 1)
    plt.plot(res.history['loss'], label='Loss')
    plt.plot(res.history['val_loss'], label='val_Loss')
    plt.title('Loss Function Evolution')
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(res.history['accuracy'], label='accuracy')
    plt.plot(res.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy Function Evolution')
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(res.history['precision'], label='precision')
    plt.plot(res.history['val_precision'], label='val_precision')
    plt.title('Precision Function Evolution')
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(res.history['recall'], label='recall')
    plt.plot(res.history['val_recall'], label='val_recall')
    plt.title('Recall Function Evolution')
    plt.legend()
    plt.savefig(fname='dense-model.png', orientation='landscape')
    plt.show()


def build_model(net_model):
    # Convolutional Layer
    net_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())
    net_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())

    # Pooling layer
    net_model.add(MaxPool2D(pool_size=(2, 2)))

    # Dropout layers
    net_model.add(Dropout(0.25))

    net_model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())
    net_model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())
    net_model.add(MaxPool2D(pool_size=(2, 2)))
    net_model.add(Dropout(0.25))

    net_model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())
    net_model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    net_model.add(BatchNormalization())
    net_model.add(MaxPool2D(pool_size=(2, 2)))
    net_model.add(Dropout(0.25))

    net_model.add(Flatten())
    net_model.add(Dense(128, activation='relu'))
    net_model.add(Dropout(0.25))
    net_model.add(Dense(10, activation='softmax'))
    return net_model


def build_dense_net_model(net_model):
    base_model = DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet', pooling='avg')
    net_model.add(base_model)
    net_model.add(Dense(10, activation='softmax'))
    return net_model


# Load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target variable into one-hotencoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# Building model
model = Sequential()
# model = build_model(model)
model = build_dense_net_model(model)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=2)

batch_size = 32
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, y_cat_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size


r = model.fit(train_generator,
              epochs=100,
              steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_cat_test))

# r = model.fit(train_generator,
#               epochs=50,
#               steps_per_epoch=steps_per_epoch,
#               validation_data=(X_test, y_cat_test))

show_metrics(r)
model.save('cnn_epochs.h5')
