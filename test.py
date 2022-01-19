from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('kernel_size_3-3.h5')

# Load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target variable into one-hotencoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

# Image tests
# my_image = X_test[100]
# plt.imshow(my_image)
# plt.savefig(fname='test-44.png', orientation='landscape')
# plt.show()
# print(y_test[100])
# print(classes_name[np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))])

# my_image = X_test[70]
# plt.imshow(my_image)
# plt.savefig(fname='test-70.png', orientation='landscape')
# plt.show()
# print(y_test[70])
# print(classes_name[np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))])

my_image = X_test[44]
plt.imshow(my_image)
plt.savefig(fname='test-44.png', orientation='landscape')
plt.show()
print(y_test[44])
print(classes_name[np.argmax(model.predict(my_image.reshape(1, 32, 32, 3)))])
