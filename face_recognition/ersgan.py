"""
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os


model = tf.keras.Sequential([
    tf.keras.Input(shape=[None, None, 3]),
    tf.keras.layers.Lambda(lambda x: x * 255.0),  # Normalizar a [0, 255]

    # Llama a la función preprocess_input pasándole la imagen
    tf.keras.layers.Lambda(lambda x: preprocess_input(x)),

    # ... (resto de las capas sin cambios)
])

model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(128, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(256, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(256, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(512, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(512, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(1024, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2D(1024, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(512, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(512, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(256, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(256, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(128, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(128, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(64, 3, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
model.add(tf.keras.layers.Conv2DTranspose(3, 3, strides=(2, 2), padding='same'))
model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x / 255.0, 0, 1)))

model.build([None, None, None, 3])
if os.path.exists("/models_ia/esrgan.pb"):
   model.load_weights("./models_ia/esrgan.pb")

def super_resolve_image(input_image_path, output_image_path):
    img = image.load_img(input_image_path)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    super_resolved = model.predict(img_array)

    super_resolved = tf.squeeze(super_resolved, axis=0)
    super_resolved = tf.clip_by_value(super_resolved, 0, 1)

    super_resolved_image = tf.keras.preprocessing.image.array_to_img(super_resolved)
    super_resolved_image.save(output_image_path)

# Ejemplo de uso
input_image_path = "./SegmentacionDocumento.jpg"
output_image_path = "./SegmentacionDocumentoErsgan.jpg"

super_resolve_image(input_image_path, output_image_path)
"""
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('./models_ia/esrgan.pth', download=True)

path_to_image = './SegmentacionDocumento.jpg'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('./sr_cedula.jpg')