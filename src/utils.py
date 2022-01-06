from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


###################
##### Classes #####
###################

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


#####################
##### Functions #####
#####################

def read_data(filepath):
    data = pd.read_csv(filepath, encoding="ISO-8859-1")
    return data


def split_data(data, split=0.2):
    # stratify:
    # rel. class frequencies is approx. preserved in each train, test fold
    train, test = train_test_split(
        data, test_size=split, stratify=data.author)
    return train, test


##### Bar plots #####

def bar_plot(data, title=""):
    data.author.value_counts().plot(kind='bar')
    plt.title(title)
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
