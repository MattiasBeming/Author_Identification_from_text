from utils import *
from baseline import *
from neuralnet_code import plot_history
import time
import tensorflow as tf
from keras import layers
import numpy as np
import nltk
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


def lstm_run(PATH_DS, OS, SW, L, DL_PATH_NLTK, NR_AUTHORS,
             SPLIT_SIZE, VALIDATION_SPLIT, EPOCHS,
             BS, PRE_TRAINED_EMB, EMB_VEC, PATH_EMB,
             EMB_DIM, BI_DIR_LSTM, DROPOUT_LAYER,
             DROPOUT_LSTM, PLOT_HISTORY, PLOT_ROW_LEN_DIST):

    ##### Functions #####

    def _pad_data(x_train, x_test, plot=False):
        # If stop words are removed
        # Pad train and test data to have the same length

        # Create set with length of rows in x_train and x_test
        # after stop words are removed
        set_train = get_set_of_rowlengths(x_train)
        set_test = get_set_of_rowlengths(x_test)

        # Plot distribution of row lengths in x_train
        if plot:
            plot_row_length_distribution(set_train)

        # Update max_length to the length of the longest row
        # between train and test
        max_length = max(max(set_train), max(set_test))  # 691

        # Pad x_train and x_test with zeros
        # Needed to make all rows have the same length
        x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
        x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
        return x_train, x_test, max_length

    ##### Load and Pre-Process Data #####
    print_header("Setup: Load and Process Data")
    s_time = time.time()

    data = read_data(PATH_DS)

    if SW:
        s_t = time.time()
        nltk.data.path.append(DL_PATH_NLTK)
        nltk.download('stopwords', download_dir=DL_PATH_NLTK)

        # use 'set' to make python faster (takes 10x longer otherwise)
        stop_w = set(nltk.corpus.stopwords.words('english'))

        # Remove stop words
        data.text = data.text.apply(lambda x: ' '.join(
            [word for word in x.split() if word not in (stop_w)]))

        print(
            f"\nTook {time.time() - s_t:.3}s to download and remove stop words")

    if L:
        s_t = time.time()
        from nltk.stem import WordNetLemmatizer
        nltk.data.path.append(DL_PATH_NLTK)
        nltk.download('wordnet', download_dir=DL_PATH_NLTK)

        wnl = WordNetLemmatizer()

        # Lemmatize words
        data.text = data.text.apply(lambda x: ' '.join(
            [wnl.lemmatize(word) for word in x.split()]))

        print(
            f"\nTook {time.time() - s_t:.3}s to lemmatize words")

    # Select subset of data to use
    subset_data = get_rand_rows(data, NR_AUTHORS)

    authors_ = np.unique(subset_data.author)
    print(f"\nSelected {len(authors_)} authors: {[a for a in authors_]}")
    print(f"{subset_data.shape[0]} out of {data.shape[0]} rows used "
          f"=> {100*(subset_data.shape[0]/data.shape[0]):.3}% of the data")

    if OS:
        print("\nOversampling data...")
        s_os = time.time()
        subset_data = oversample(subset_data)
        print(f"Took {time.time() - s_os:.3}s to oversample data")

    # Split data
    train, test = split_data(subset_data, SPLIT_SIZE)

    # Make Y-values categorical to classify authors
    y_train = to_categorical(train.author.to_numpy())
    y_test = to_categorical(test.author.to_numpy())

    # All rows have the same length (1000) if stop words are not removed
    max_length = 1000

    # Use pre-trained embedding
    if PRE_TRAINED_EMB:
        # Code inspired by
        # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

        # Prepare tokenizer
        tok = Tokenizer()
        tok.fit_on_texts(train.text)
        vocab_size = len(tok.word_index) + 1  # 10001

        # Encode the text with integers

        x_train = tok.texts_to_sequences(train.text)
        x_test = tok.texts_to_sequences(test.text)

        # Some data seems to be missing,
        # always pad train and test data to have the same length
        x_train, x_test, max_length = _pad_data(
            x_train, x_test, PLOT_ROW_LEN_DIST)

        # TODO cite GloVe: https://nlp.stanford.edu/projects/glove/

        print("\nLoading pre-trained embedding...")
        load_time = time.time()
        # load pre-trained embedding
        emb_ind = dict()
        file = open(PATH_EMB, encoding="utf8")
        for line in file:
            line_ = line.split()
            coeff = np.asarray(line_[1:], dtype='float32')
            emb_ind[line_[0]] = coeff
        file.close()
        print(f"{len(emb_ind)} word vectors loaded"
              f" from GloVe in {time.time() - load_time:.3}s")

        # Create a weight matrix for words
        emb_mat = np.zeros((vocab_size, EMB_VEC))
        for word, i in tok.word_index.items():
            emb_vec = emb_ind.get(word)
            if emb_vec is not None:
                emb_mat[i] = emb_vec

    # Train an embedding layer with one-hot encoded vectors
    else:
        # Code inspired by "3. Example of Learning an Embedding"
        # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

        # Make one-hot vectors for embedding layer
        vocab_size = 10000
        x_train = [one_hot(d, vocab_size) for d in train.text]
        x_test = [one_hot(d, vocab_size) for d in test.text]

        # Some data seems to be missing,
        # always pad train and test data to have the same length
        x_train, x_test, max_length = _pad_data(
            x_train, x_test, PLOT_ROW_LEN_DIST)

    print("\nSetup complete!")
    print(f"Took: {time.time()-s_time} seconds")

    ##### Classification #####
    print_header("Running classification")
    s_time = time.time()

    ### Model ###
    # Code for the model is inspired by
    # https://djajafer.medium.com/multi-class-text-classification-with-keras-and-lstm-4c5525bef592

    embedding_dim = EMB_VEC if PRE_TRAINED_EMB else EMB_DIM
    total_nr_authors = y_train.shape[1]

    inputs = tf.keras.Input(shape=(max_length,))

    if PRE_TRAINED_EMB:
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             input_length=max_length, weights=[emb_mat],
                             trainable=False)(inputs)
    else:
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             input_length=max_length)(inputs)

    x = layers.SpatialDropout1D(DROPOUT_LAYER)(x)

    if BI_DIR_LSTM:
        x = layers.Bidirectional(layers.LSTM(
            max_length//2, dropout=DROPOUT_LSTM))(x)
    else:
        x = layers.LSTM(max_length//2, dropout=DROPOUT_LSTM)(x)

    outputs = layers.Dense(total_nr_authors, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['acc'])

    print("\nTraining model...\n")

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BS,
        validation_split=VALIDATION_SPLIT)

    print_header("Results")

    loss, acc = model.evaluate(x_test, y_test,
                               verbose=False, batch_size=BS)
    print("\nTest Accuracy: {:.5f}".format(acc))

    print(f"\nClassification took: {time.time()-s_time} seconds")

    if PLOT_HISTORY:
        plot_history(history)

    # Model summary as string
    s = []
    model.summary(print_fn=lambda x: s.append(x))
    model_summary = '\n'.join(s)

    return model_summary, acc, history
