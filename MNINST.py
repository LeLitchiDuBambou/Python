# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import jedi
import os

train=pd.read_csv("train.csv")
test_images=pd.read_csv('test.csv')
sample_submissions=pd.read_csv('sample_submission')

#On plot un des chiffres
plt.figure()
plt.imshow(train.iloc[12,1:].values.reshape(28,28))
plt.colorbar()
plt.grid(False)

train_images=train.iloc[:,1:]
train_labels=train.iloc[:,0]
train_images=train_images/255.0
test_images=test_images/255.0

model = keras.Sequential([
    keras.layers.Dense(64, input_dim=784, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(train_images, train_labels)

predictions = model.predict(test_images)