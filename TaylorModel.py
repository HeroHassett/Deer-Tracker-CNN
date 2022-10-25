import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.preprocessing import image

# Import images and split into training and validation sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_images, test_images = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def LeNet_model(input_shape=(32, 32, 3)):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # First set of CONV_RELU_POOL layers
    model.add(layers.Conv2D(filters=20, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    # Second set of CONV_RELU_Pool layers
    model.add(layers.Conv2D(filters=50, kernel_size=5, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    # Flatten
    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(units=10))

    return model

Deer_model = LeNet_model()
Deer_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
Deer_model.summary()

history = Deer_model.fit(
    train_images, y_train, epochs=1, callbacks=callbacks.Callback(), validation_data=(test_images, y_test)
)

test_loss, test_acc = Deer_model.evaluate(test_images,  y_test, verbose=2)

print(test_acc)

img = image.load_img('/Users/alexk/Downloads/Deer/Deer/X2BU31OQDWKE.jpg', target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array[0] = None

y_proba = Deer_model.predict(img_array)
y_proba = y_proba.transpose()
y_proba = y_proba.astype(int)

for i in y_proba:
    is_deer = True
    if y_proba[4] < y_proba[i]:
        is_deer = False
if is_deer:
    print("This is a deer")
else:
    print("This is not a deer")

#Deer_model.save('directory')