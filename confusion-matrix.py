from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

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
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes_name)

fig, ax = plt.subplots(figsize=(10, 10))
disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='copper')
# plt.savefig(fname='confusion-matrix/conv256.png', orientation='landscape')
# plt.savefig(fname='confusion-matrix/confusion_1-1.png', orientation='landscape')
# plt.savefig(fname='confusion-matrix/confusion_3-3.png', orientation='landscape')
# plt.savefig(fname='confusion-matrix/confusion_5-5.png', orientation='landscape')
# plt.savefig(fname='confusion-matrix/confusion_7-7.png', orientation='landscape')
plt.savefig(fname='confusion-matrix/confusion_densenet.png', orientation='landscape')
plt.show()
