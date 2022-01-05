
import matplotlib.pyplot as plt

if False:
    from tensorflow.keras.layers import TextVectorization
    from keras.backend import clear_session
    import tensorflow as tf

    # Reset weights of last training
    clear_session()


# Code from https://keras.io/examples/nlp/text_classification_from_scratch/


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
