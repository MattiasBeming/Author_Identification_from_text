from utils import *
from baseline import *
from neuralnet_code import plot_history
import time
from pathlib import Path
import tensorflow as tf
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

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


##### Variables #####

# Dataset filepath
PATH_DS = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

# If stop words should be removed
SW = False

# Path where the stop-word-data should be stored
DL_PATH_SW = Path('E:/Projects/Author_Identification/data/nltk')

# Select nr of authors to use from dataset to only use a subset of the data
NR_AUTHORS = 8

# Size of test dataset - Split data between train and test
SPLIT_SIZE = 0.2

# Splits data when training from train dataset into train and validation
VALIDATION_SPLIT = 0.15

# How many epochs to train for
EPOCHS = 50

# Batch size to train/evaluate with
BS = 64

# If pre-trained embeddings should be used
# If False, embeddings will be trained
PRE_TRAINED_EMB = True

# Embedding vectors available: 50, 100, 200, 300 (GloVe)
EMB_VEC = 300

# Path to the pre-trained embeddings
PATH_EMB = Path(f"data/glove.6B/glove.6B.{EMB_VEC}d.txt")

# Embedding dim used if embedding is not pre-trained else
# EMB_VEC is used as embedding dimension
EMB_DIM = 128

# If the model should use Bidirectional LSTM
# If False, only one LSTM layer is used
BI_DIR_LSTM = True

# Dropouts when training the model
DROPOUT_LAYER = 0.4
DROPOUT_LSTM = 0.2

# Plots
PLOT_HISTORY = False
PLOT_ROW_LEN_DIST = False


##### Load and Pre-Process Data #####
print_header("Setup: Load and Process Data")
s_time = time.time()

data = read_data(PATH_DS)

if SW:
    s_t = time.time()
    import nltk
    nltk.data.path.append(DL_PATH_SW)
    nltk.download('stopwords', download_dir=DL_PATH_SW)

    # use 'set' to make python faster (takes 10x longer otherwise)
    stop_w = set(nltk.corpus.stopwords.words('english'))

    # Remove stop words
    data.text = data.text.apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop_w)]))

    print(f"\nTook {time.time() - s_t:.3}s to download and remove stop words\n")

# Select subset of data to use
subset_data = get_rand_rows(data, NR_AUTHORS)

authors_ = np.unique(subset_data.author)
print(f"Selected {len(authors_)} authors: {[a for a in authors_]}")
print(f"{subset_data.shape[0]} out of {data.shape[0]} rows used "
      f"=> {100*(subset_data.shape[0]/data.shape[0]):.3}% of the data")

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

    # Pad train and test data to have the same length
    if SW:
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

    # Pad train and test data to have the same length
    if SW:
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
    np.array(x_train),
    y_train,
    epochs=EPOCHS,
    batch_size=BS,
    validation_split=VALIDATION_SPLIT)

print_header("Results")

loss, acc = model.evaluate(np.array(x_test), y_test,
                           verbose=False, batch_size=BS)
print("\nTest Accuracy: {:.4f}".format(acc))

print(f"\nClassification took: {time.time()-s_time} seconds")

if PLOT_HISTORY:
    plot_history(history)
