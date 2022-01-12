from utils import *
from neuralnet_code import plot_history
import time
from pathlib import Path
import tensorflow as tf
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot


##### Variables #####

# Select nr of authors to use from dataset to only use a subset of the data
NR_AUTHORS = 8

# Size of test dataset - Split data between train and test
SPLIT_SIZE = 0.2

# How many epochs to train for
EPOCHS = 20

# Batch size to train/evaluate with
BS = 64

##### Load and Pre-Process Data #####
print_header("Setup: Load and Process Data")
filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')
data = read_data(filepath)

# Select subset of data to use
subset_data = get_rand_rows(data, NR_AUTHORS)

authors_ = np.unique(subset_data.author)
print(f"Selected {len(authors_)} authors: {[a for a in authors_]}")
print(f"{subset_data.shape[0]} out of {data.shape[0]} rows used "
      f"=> {100*(subset_data.shape[0]/data.shape[0]):.3}% of the data")

# Split data
train, test = split_data(subset_data, SPLIT_SIZE)

# Code inspired by "3. Example of Learning an Embedding"
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# Code for the model is inspired by
# https://djajafer.medium.com/multi-class-text-classification-with-keras-and-lstm-4c5525bef592

# Make one-hot vectors for embedding layer
vocab_size = 10000
x_train = [one_hot(d, vocab_size) for d in train.text]
x_test = [one_hot(d, vocab_size) for d in test.text]

# Make Y-values categorical to classify authors
y_train = to_categorical(train.author.to_numpy())
y_test = to_categorical(test.author.to_numpy())

print("\nSetup complete!")

##### Classification #####
print_header("Running classification")
s_time = time.time()

### Model ###
# All rows have the same length (1000)
max_length = 1000
embedding_dim = 128

inputs = tf.keras.Input(shape=(max_length,))

x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                     input_length=max_length)(inputs)
x = layers.SpatialDropout1D(0.4)(x)
x = layers.Bidirectional(layers.LSTM(max_length//2, dropout=0.2))(x)

outputs = layers.Dense(44, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['acc'])

print("\nTraining model...\n")

history = model.fit(
    np.array(x_train),
    y_train,
    epochs=EPOCHS,
    batch_size=BS,
    validation_split=0.15)

print_header("Results")

loss, acc = model.evaluate(np.array(x_test), y_test,
                           verbose=False, batch_size=BS)
print("\nTest Accuracy: {:.4f}".format(acc))

print(f"\nTook: {time.time()-s_time} seconds")

plot_history(history)
