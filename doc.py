import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load model
# model = keras.models.load_model('models/conv256.h5')
# model = keras.models.load_model('models/kernel_size_1-1.h5')
# model = keras.models.load_model('models/kernel_size_3-3.h5')
# model = keras.models.load_model('models/kernel_size_5-5.h5')
# model = keras.models.load_model('models/kernel_size_7-7.h5')
model = keras.models.load_model('models/densenet.h5')

# Load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target variable into one-hotencoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# Model evaluation
evaluation = model.evaluate(X_test, y_cat_test)
print(evaluation)
print(f'Test Loss : {evaluation[0]:.2f}')
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')
# print(f'Test Precision : {evaluation[2] * 100:.2f}%')
# print(f'Test Recall : {evaluation[3] * 100:.2f}%')
