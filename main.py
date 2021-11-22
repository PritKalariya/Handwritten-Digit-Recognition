import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the dataset directly from the tf module
mnist = tf.keras.datasets.mnist

# Pre-Processing
# Split train test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalising
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Modeling
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Complie
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=3)

model.save('Handwritten.model')