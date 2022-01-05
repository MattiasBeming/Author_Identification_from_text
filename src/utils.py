from enum import Enum
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class VectorizationMode(Enum):
    COUNT = 0
    TFIDF = 1


class Profile:
    def __init__(self, ds_name, DS, lemmatize,
                 remove_stop_words, mode=VectorizationMode.COUNT):
        self.ds_name = ds_name
        self.DS = DS
        self.lemmatize = lemmatize
        self.remove_stop_words = remove_stop_words
        self.mode = mode

        self.pipe = None
        self.predictions = None
        self.res = None

    def __str__(self):
        return (
            self.ds_name + "_"
            + ("l_" if self.lemmatize else "")
            + ("sw_" if self.remove_stop_words else "")
            + self.mode.name
        )

    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_preds(self, preds):
        self.predictions = preds

    def set_res(self, res):
        self.res = res

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

    def get_pipe(self):
        if self.pipe is None:
            raise ValueError('Pipe not set')
        return self.pipe

    def get_predictions(self):
        if self.predictions is None:
            raise ValueError('Predictions not set')
        return self.predictions

    def get_res(self):
        if self.res is None:
            raise ValueError('Res not set')
        return self.res

    def get_acc(self):
        if self.res is None:
            raise ValueError('Res not set')
        return self.res['accuracy']


def read_data(filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    return data


def split_data(data, split=0.2):
    # stratify:
    # rel. class frequencies is approx. preserved in each train, test fold
    train, test = train_test_split(
        data, test_size=split, stratify=data.author)
    return train, test
