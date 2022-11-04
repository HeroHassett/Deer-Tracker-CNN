import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_path = "/Users/alexk/Documents/GitHub/Deep_Learning_for_Binary_Classification_of_Infectious_Keratitis/Stuff for " \
           "Chase/IMG_1312.jpeg" # Image Path
img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.show()

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

VGG16_img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)

VGG16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')

VGG16_prediction = VGG16.predict(VGG16_img_preprocessed)

prediction = tf.keras.applications.vgg16.decode_predictions(VGG16_prediction, top=3)[0]

deer = False
for animal in prediction:
    for i in range(len(prediction)):
        if animal[i] == "gazelle":
            deer = True
if deer:
    print("This is a deer")
else:
    print("This is not a deer")