import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import pathlib

train_data_dir = pathlib.WindowsPath('C:/Users/fabio/Desktop/Borsa/Datasets/Face_expression_recognition_dataset/train')
val_data_dir = pathlib.WindowsPath('C:/Users/fabio/Desktop/Borsa/Datasets/Face_expression_recognition_dataset/validation')

#image_count = len(list(train_data_dir.glob('*/*.jpg')))
#print(image_count)

batch_size = 32
img_height = 48
img_width = 48

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    labels='inferred',
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    labels='inferred',
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
#print(class_names)

# Visualizza i dati
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
