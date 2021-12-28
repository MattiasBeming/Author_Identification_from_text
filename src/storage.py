
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

if False:
    from tensorflow.keras.layers import TextVectorization
    from keras.backend import clear_session
    import tensorflow as tf

    # Reset weights of last training
    clear_session()

filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

data = pd.read_csv(filepath, encoding="ISO-8859-1")

train, test = train_test_split(data, test_size=0.2)

# Bad rows ? 51082
# print(data['text'].iloc[51082])


# Code from https://keras.io/examples/nlp/text_classification_from_scratch/

def read_data(filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    return data


def split_data(filepath, split=0.2):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")

    # stratify:
    # rel. class frequencies is approx. preserved in each train, test fold
    train, test = train_test_split(
        data, test_size=split, stratify=data.author)

    return train, test


if False:
    def custom_standardization(input_data):
        return input_data

    # Model constants
    max_features = 20000
    embedding_dim = 128
    sequence_length = 1000

    # positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams

    # ngrams - Optional specification for ngrams to create from the possibly-split input text

    vectorize_layer = TextVectorization(
        # standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # embedding_layer : input_dim (summation of unique words in your corpus)
    # output_dim (size of corresponding dense vectors)
    # input_length (length of input sequences)

BATCH_SIZE = 64
EPOCHS = 100

if False:
    plt.style.use('ggplot')
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=BATCH_SIZE)

    def plot_history(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

    plot_history(history)
