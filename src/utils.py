from enum import Enum
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from tensorflow.keras.utils import to_categorical

###################
##### Classes #####
###################


class VectorizationMode(Enum):
    COUNT = 0
    TFIDF = 1


class Profile:
    def __init__(self, ds_name, DS, lemmatize,
                 remove_stop_words, mode=VectorizationMode.COUNT,
                 nr_authors=None):
        # From argument-list
        self.ds_name = ds_name
        self.DS = DS
        self.lemmatize = lemmatize
        self.remove_stop_words = remove_stop_words
        self.mode = mode

        # Baseline
        self.baseline_pipe = None
        self.baseline_res = None

        # Keras
        self.nr_authors = nr_authors
        self.x_train = None
        self.y_train = to_categorical(DS.get_train(ds_name).author)
        self.x_test = None
        self.y_test = to_categorical(DS.get_test(ds_name).author)
        self.vectorizer = None
        self.model = None

        # All
        self.predictions = None

    def __str__(self):
        return (
            f"{self.ds_name}_"
            + ("l_" if self.lemmatize else "")
            + ("sw_" if self.remove_stop_words else "")
            + (f"{self.mode.name}_")
            + (str(self.nr_authors) if self.nr_authors is not None else "")
        ).strip("_")

    def set_baseline_pipe(self, baseline_pipe):
        self.baseline_pipe = baseline_pipe

    def set_preds(self, preds):
        self.predictions = preds

    def set_baseline_res(self, baseline_res):
        self.baseline_res = baseline_res

    def get_train(self):
        return self.DS.get_train(self.ds_name)

    def get_test(self):
        return self.DS.get_test(self.ds_name)

    def get_valid(self):
        return self.DS.get_valid(self.ds_name)

    def is_lemmatized(self):
        return self.lemmatize

    def is_stop_words_removed(self):
        return self.remove_stop_words

    def get_mode(self):
        return self.mode

    def get_baseline_pipe(self):
        if self.baseline_pipe is None:
            raise ValueError('Pipe not set')
        return self.baseline_pipe

    def get_predictions(self):
        if self.predictions is None:
            raise ValueError('Predictions not set')
        return self.predictions

    def get_baseline_res(self):
        if self.baseline_res is None:
            raise ValueError('Baseline res not set')
        return self.baseline_res

    def get_baseline_acc(self):
        if self.baseline_res is None:
            raise ValueError('Baseline res not set')
        return self.baseline_res['accuracy']

    ### Keras ###
    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer
        self.fit_transform()

    def fit_transform(self):
        if self.vectorizer is None:
            raise ValueError('Vectorizer not set')
        self.vectorizer.fit(self.get_train().text)
        self.x_train = self.vectorizer.transform(self.get_train().text)
        self.x_test = self.vectorizer.transform(self.get_test().text)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def get_transformed_x_train(self, as_numpy=False):
        if self.x_train is None:
            raise ValueError('Train data not transformed via vectorizer')
        return self.x_train.todense() if as_numpy else self.x_train

    def get_transformed_y_train(self):
        return self.y_train

    def get_transformed_x_test(self, as_numpy=False):
        if self.x_test is None:
            raise ValueError('Test data not transformed via vectorizer')
        return self.x_test.todense() if as_numpy else self.x_test

    def get_transformed_y_test(self):
        return self.y_test

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history

    def set_acc(self, acc):
        self.acc = acc

    def get_acc(self):
        return self.acc

#####################
##### Functions #####
#####################


def read_data(filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    return data


def get_rand_rows(data, nr_authors):
    unique_authors = np.unique(data.author)
    nr_unique_authors = len(unique_authors)

    nr_authors_to_draw = max(
        min(nr_authors, nr_unique_authors), 2)  # Minimum 2 authors

    np.random.seed(69)
    rand_authors = np.random.choice(
        unique_authors, nr_authors_to_draw, replace=False)

    return data.loc[data.author.isin(rand_authors)]


def split_data(data, split=0.2):
    # stratify:
    # rel. class frequencies is approx. preserved in each train, test fold
    train, test = train_test_split(
        data, test_size=split, stratify=data.author)
    return train, test


def get_set_of_rowlengths(data):
    set_ = {}
    for w in data:
        len_ = len(w)
        if len_ not in set_:
            set_[len_] = 1
        else:
            set_[len_] += 1
    return set_


##### Bar plots #####

def bar_plot(data, title=""):
    data.author.value_counts().plot(kind='bar')
    plt.title(title)
    plt.show()
    return


def plot_row_length_distribution(set_):
    plt.bar(list(set_.keys()), list(set_.values()))
    plt.show()
    return

##### Over under-spamling #####


def oversample(data):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=69)

    # Oversample
    temp_data = np.array(data.text).reshape(-1, 1)  # transpose
    data_text_oversample, data_author_oversample = \
        ros.fit_resample(temp_data, data.author)

    # convert to pandas series again
    data_text_oversample = pd.Series(data_text_oversample.flatten())

    # Add to new dataframe
    data_oversample = pd.DataFrame(
        {"text": data_text_oversample, "author": data_author_oversample})

    return data_oversample


def undersample(data):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=69)

    # Undersample
    temp_data = np.array(data.text).reshape(-1, 1)  # transpose
    data_text_undersample, data_author_undersample = \
        rus.fit_resample(temp_data, data.author)

    # convert to pandas series again
    data_text_undersample = pd.Series(data_text_undersample.flatten())

    # Add to new dataframe
    data_undersample = pd.DataFrame(
        {"text": data_text_undersample, "author": data_author_undersample})

    return data_undersample


##### User interface #####

# Code from https://github.com/MattiasBeming/LiU-AI-Project-Active-Learning-for-Music
def print_header(header):
    n = len(header)
    print()
    print("#" * (n + 8))
    print(f"# {'=' * (n + 4)} #")
    print(f"# = {header} = #")
    print(f"# {'=' * (n + 4)} #")
    print("#" * (n + 8))
    print()


def pick_multiple_profiles(profiles: list):

    # Initial prompt
    print("Pick what profiles to evaluate.")

    indexed_ps = {i: p for i, p in enumerate(profiles)}

    picked_inds = []
    while True:
        # Print unpicked Ps
        print("Profiles to pick from:")
        if len(picked_inds) == len(indexed_ps):
            print("\t-")
        else:
            for i, p in indexed_ps.items():
                if i not in picked_inds:
                    print(f"\t{i}: {str(p)}")

        # Print picked Ps
        print("Picked Profiles:")
        if not picked_inds:
            print("\t-")
        else:
            for i in sorted(picked_inds):
                print(f"\t{i}: {str(indexed_ps[i])}")

        # Input prompt
        print("Enter indices on format 'i' or 'i-j'.")
        print("Drop staged Profiles with 'drop i'.")
        print("Write 'done' when you are done.")

        # Handle input
        try:
            idx = input("> ")
            if idx == "done":  # Check if done
                break
            elif bool(re.match("^[0-9]+-[0-9]+$", idx)):  # Check if range
                span_str = idx.split("-")
                picked_inds += [i for i in range(
                    int(span_str[0]), int(span_str[1]) + 1)
                    if i not in picked_inds]
            elif bool(re.match("^drop [0-9]+$", idx)):
                picked_inds.remove(int(idx.split()[1]))
            elif int(idx) in indexed_ps.keys() \
                    and int(idx) not in picked_inds:  # Check if singular
                picked_inds.append(int(idx))
        except ValueError:
            continue

    return [indexed_ps[i] for i in picked_inds]
