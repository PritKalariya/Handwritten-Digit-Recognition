import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the dataset directly from the tf module
# mnist = tf.keras.datasets.mnist

# # Pre-Processing
# # Split train test
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# # Normalising
# X_train = tf.keras.utils.normalize(X_train, axis=1)
# X_test = tf.keras.utils.normalize(X_test, axis=1)

# # Modeling
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # Complie
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Fit the model
# model.fit(X_train, y_train, epochs=3)

# model.save('handwritten_digits.model')

# Laod the model
model = tf.keras.models.load_model('handwritten_digits.model')


# Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(loss)
# print(accuracy)


# Testing
# Load custom images and predict them
image_number = 1

while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
    finally:
        image_number += 1
