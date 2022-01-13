from lstm_run import PLOT_ROW_LEN_DIST
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

##### Variables #####

# Dataset filepath
PATH_DS = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

# If stop words should be removed
SW = False

# Path to where nltk-data should be stored
DL_PATH_NLTK = Path('E:/Projects/Author_Identification/data/nltk')

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

# Embedding vectors available: 50, 100, 200, 300 (GloVe)
# (EMB_VEC is used as embedding dimension)
EMB_VEC = 300

# Path to the pre-trained embeddings
PATH_EMB = Path(f"data/glove.6B/glove.6B.{EMB_VEC}d.txt")

# Plot
PLOT_HISTORY = True
PLOT_ROW_LEN_DIST = False

##### Load and Pre-Process Data #####
print_header("Setup: Load and Process Data")
s_time = time.time()

data = read_data(PATH_DS)

if SW:
    s_t = time.time()
    import nltk

    nltk.data.path.append(DL_PATH_NLTK)
    nltk.download('stopwords', download_dir=DL_PATH_NLTK)

    # use 'set' to make python faster (takes 10x longer otherwise)
    stop_w = set(nltk.corpus.stopwords.words('english'))

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


# Code inspired by
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# Prepare tokenizer
tok = Tokenizer()
tok.fit_on_texts(train.text)
vocab_size = len(tok.word_index) + 1  # 10001

# Encode the text with integers
x_train = tok.texts_to_sequences(train.text)
x_test = tok.texts_to_sequences(test.text)

# All rows have the same length (1000) if stop words are not removed
max_length = 1000

# Pad train and test data to have the same length
if SW:
    # Create set with length of rows in x_train after stop words removed
    set_train = {}
    for w in x_train:
        len_ = len(w)
        if len_ not in set_train:
            set_train[len_] = 1
        else:
            set_train[len_] += 1

    set_test = {}
    for w in x_test:
        len_ = len(w)
        if len_ not in set_test:
            set_test[len_] = 1
        else:
            set_test[len_] += 1

    # Plot distribution of word lengths in x_train
    if PLOT_ROW_LEN_DIST:
        plt.bar(list(set_train.keys()), list(set_train.values()))
        plt.show()

    max_length = max(max(set_train), max(set_test))  # 691

    if SW:
        # Pad sequences with zeros - needed to make all rows have the same length
        x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
        x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

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

print("\nSetup complete!")
print(f"Took: {time.time()-s_time} seconds")

##### Classification #####
print_header("Running classification")
s_time = time.time()

### Model ###
embedding_dim = EMB_VEC
total_nr_authors = y_train.shape[1]

inputs = tf.keras.Input(shape=(max_length,))

trainable = False  # Update embedding weights during training (Tune or not)
x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                     input_length=max_length, weights=[emb_mat],
                     trainable=trainable)(inputs)
# x = layers.SpatialDropout1D(0.4)(x)  # Gave bad results

# Code inspired by
# https://www.wintellect.com/text-classification-with-neural-networks/

# Did not yeild good enough results
x = layers.Flatten()(x)

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
