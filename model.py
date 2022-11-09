#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from params import pr


def get_model(num_classes, weights_path=None):

    model = create_model(num_classes)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model


def create_model(num_classes):
    input_tensor = Input(shape=(pr.n_features, pr.feature_size, 1), name='feature_input')

    x = Conv2D(filters=16,
               kernel_size=5,
               activation='relu',
               strides=[1, 1],
               padding='SAME', use_bias=True)(input_tensor)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32,
               kernel_size=3,
               activation='relu',
               strides=[1, 1],
               padding='SAME', use_bias=True)(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=64,
               kernel_size=3,
               activation='relu',
               strides=[2, 2],
               padding='SAME', use_bias=True)(x)

    x = Conv2D(filters=128,
               kernel_size=3,
               activation='relu',
               strides=[1, 1],
               padding='SAME', use_bias=True)(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)

    x = Dense(num_classes, activation='softmax', name='score_predict')(x)

    # Create model.
    model = Model(input_tensor, x)

    return model

