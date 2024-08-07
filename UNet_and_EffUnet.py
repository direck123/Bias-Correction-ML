import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import tarfile
import os
import numpy.ma as ma
import pandas as pd
from scipy.interpolate import griddata
#from keras.utils import np_utils

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import ImageGrid
import datetime
from pathlib import Path
import scipy.io as io
import mat73

from keras.models import Sequential
from keras import backend as K
from keras import initializers, constraints, regularizers, layers
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Conv2D, BatchNormalization, add, LeakyReLU, UpSampling2D
from keras.layers import Conv2DTranspose, Dense, Flatten, Concatenate,concatenate, MaxPooling2D
from keras.losses import binary_crossentropy

import efficientnet.keras as efn
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
import tensorflow as tf
from tensorflow import keras
from numpy import pad
from collections import OrderedDict


##Unet
OUTPUT_CHANNELS = 1
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

# Define the upsampler (decoder):
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 2])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
  ]
  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='relu')
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = Generator()
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

# Print model summary
model.summary()


##EfficientUNetB4
import tensorflow_addons as tfa
def H(lst, name, use_gn=False):
    if use_gn:
        norm = GroupNormalization(groups=1, name=name+'_gn')
    else:
        norm = BatchNormalization(name=name+'_bn')

    x = concatenate(lst)
    num_filters = int(x.shape.as_list()[-1]/2)

    x = Conv2D(num_filters, (3, 3), padding='same', name=name)(x)
    x = norm(x)
    x = LeakyReLU(alpha=0.1, name=name+'_activation')(x)

    return x

def U(x, use_gn=False):
    if use_gn:
        norm = GroupNormalization(groups=1)
    else:
        norm = BatchNormalization()

    num_filters = int(x.shape.as_list()[-1]/2)

    x = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = norm(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

def EfficientUNet(input_shape):
    backbone = efn.EfficientNetB5(
        weights=None,
        include_top=True,
        pooling = 'max',
        input_shape=input_shape
    )

    input = backbone.input
    x00 = backbone.input  # (384,384,2)
    x10 = backbone.get_layer('stem_activation').output  # (128, 256, 4)
    x20 = backbone.get_layer('block2d_add').output  # (64, 128, 32)
    x30 = backbone.get_layer('block3d_add').output  # (32, 64, 56)
    x40 = backbone.get_layer('block5f_add').output  # (16, 32, 160)
    x50 = backbone.get_layer('block7b_add').output  # (8, 16, 448)

    x01 = H([x00, U(x10)], 'X01')
    x11 = H([x10, U(x20)], 'X11')
    x21 = H([x20, U(x30)], 'X21')
    x31 = H([x30, U(x40)], 'X31')
    x41 = H([x40, U(x50)], 'X41')

    x02 = H([x00, x01, U(x11)], 'X02')
    x12 = H([x11, U(x21)], 'X12')
    x22 = H([x21, U(x31)], 'X22')
    x32 = H([x31, U(x41)], 'X32')

    x03 = H([x00, x01, x02, U(x12)], 'X03')
    x13 = H([x12, U(x22)], 'X13')
    x23 = H([x22, U(x32)], 'X23')

    x04 = H([x00, x01, x02, x03, U(x13)], 'X04')
    x14 = H([x13, U(x23)], 'X14')

    x05 = H([x00, x01, x02, x03, x04, U(x14)], 'X05')

    x_out = Concatenate(name='bridge')([x01, x02, x03, x04, x05])
    x_out = Conv2D(1, (3,3), padding="same", name='final_output', activation=tfa.activations.mish)(x_out)
    #x_out = Conv2D(1, (3,3), padding="same", name='final_output', activation='relu')(x_out)

    return Model(inputs=input, outputs=x_out)

model = EfficientUNet((256, 256,2))

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

##Training, assume that you have data ready
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

# Create the training and validation datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Define the batch size and preprocessing function
BATCH_SIZE = 4

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label

# Apply the preprocessing function, shuffle, batch, prefetch, and repeat the datasets
train_ds = train_ds.map(preprocess_data).shuffle(buffer_size=len(x_train)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(preprocess_data).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

#model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
# Train the model using the iterators and the early stopping callback
history = model.fit(train_ds, validation_data=val_ds, epochs=20)
