from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
import matplotlib.pyplot as plt


def clear():
    # Clear session - Reset weights of last training
    clear_session()

def set_model():
    pass

def code(train, test):
    # TEMPORARY
    X_train = train.text
    Y_train = train.author
    X_test = test.text
    Y_test = test.author

    input_dim = X_train.shape[1]  # Number of features
    input_dim = 1  # Number of features

    # Define model
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Run classification
    model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
    model.summary()

    # Plot
    history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

    # Evaluate accuracy
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))





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
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()