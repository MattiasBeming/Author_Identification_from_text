from pathlib import Path
from lstm import *
from neuralnet_code import plot_history

##### Variables #####

# Dataset filepath
PATH_DS = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

# If stop words should be removed
SW = False

# If words should be lemmatized
L = False

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

#### Class #####


class P():
    def __init__(self, os, sw, l, pte, bdl, nr_authors):
        self.os = os
        self.sw = sw
        self.l = l
        self.pte = pte
        self.bdl = bdl
        self.nr_authors = nr_authors

        self.model_sum = None
        self.acc = None
        self.history = None

    def __str__(self):
        return (
            ("Bidir-lstm_" if self.bdl else "lstm_")
            + ("OS_" if self.os else "")
            + ("sw_" if self.sw else "")
            + ("l_" if self.l else "")
            + ("pre_" if self.pte else "")
            + f"{self.nr_authors}"
        )

    def set_model_sum(self, ms):
        self.model_sum = ms

    def set_acc(self, acc):
        self.acc = acc

    def set_history(self, history):
        self.history = history

    def get_os(self):
        return self.os

    def get_sw(self):
        return self.sw

    def get_l(self):
        return self.l

    def get_pte(self):
        return self.pte

    def get_bdl(self):
        return self.bdl

    def get_nr_authors(self):
        return self.nr_authors

    def get_model_sum(self):
        return self.model_sum

    def get_acc(self):
        return self.acc

    def get_history(self):
        return self.history


def print_all(profiles):
    print("\n\n=================================================================\n\n")
    print("#################################################################")
    print("#################      Summary of Results      ##################")
    print("#################################################################")
    print()

    p_0 = profiles[0]
    print("\n#################################################################")
    print("Profile:")
    print(f"OS={p_0.get_os()}, SW={p_0.get_sw()}, L={p_0.get_l()}, "
          f"PRE_TRAINED_EMB={p_0.get_pte()}, \n"
          f"BI_DIR_LSTM={p_0.get_bdl()}")

    print(f"{p_0.get_model_sum()}\n")

    for p in profiles:
        print(f"{str(p)}: Test Accuracy: {p.get_acc():.6f}")
        plot_history(p.get_history())


##### Run #####

EPOCHS = 50

# This is plotted in the end (print_all(picked_profiles))
PLOT_HISTORY = False

# Select nr of authors to use from dataset to only use a subset of the data
NR_AUTHORS = [4, 6, 8, 10, 14, 18, 24, 30, 35, 45]  # 10

# Chosen best profile from execution of lstm_run.py
OS = True
SW = False
L = True
PRE_TRAINED_EMB = True
BI_DIR_LSTM = False

profiles = []

profiles.extend([
    P(OS, SW, L, PRE_TRAINED_EMB, BI_DIR_LSTM, nrA)
    for nrA in NR_AUTHORS
])

picked_profiles = pick_multiple_profiles(profiles)

for p in picked_profiles:
    model_sum, acc, history = \
        lstm_run(PATH_DS, p.get_os(), p.get_sw(), p.get_l(),
                 DL_PATH_NLTK, p.get_nr_authors(),
                 SPLIT_SIZE,
                 VALIDATION_SPLIT, EPOCHS,
                 BS, p.get_pte(), EMB_VEC, PATH_EMB,
                 EMB_DIM, p.get_bdl(), DROPOUT_LAYER,
                 DROPOUT_LSTM, PLOT_HISTORY,
                 PLOT_ROW_LEN_DIST)
    p.set_model_sum(model_sum)
    p.set_acc(acc)
    p.set_history(history)

print_all(picked_profiles)
