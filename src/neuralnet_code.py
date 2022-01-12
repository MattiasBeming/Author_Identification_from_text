import tensorflow as tf
from keras import layers
from keras.backend import clear_session
import matplotlib.pyplot as plt


def clear():
    # Clear session - Reset weights of last training
    clear_session()


def get_NN_model(input_dim, output_dim):
    # Define model
    inputs = tf.keras.Input(shape=(input_dim,),
                            sparse=True, dtype="int64")

    x = layers.Dense(128, input_dim=input_dim, activation="relu")(inputs)
    predictions = layers.Dense(
        output_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    return model


def compile(model):
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['acc'])
    model.summary()


def fit(model, x_train, y_train, epochs=1):
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        # validation_data=validation,  # TODO
        validation_split=0.15,
        epochs=epochs
    )
    return history


def evaluate(model, X_ds, y_ds):
    _, acc = model.evaluate(X_ds, y_ds, verbose=False, batch_size=64)
    print("Accuracy: {:.4f}".format(acc))
    return acc


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
