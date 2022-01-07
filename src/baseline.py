
from utils import VectorizationMode as VM
import numpy as np
import pandas as pd


##### Preprocess #####

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


##### Pipe #####

def get_vectorizer(mode=VM.COUNT, lemma=False, stop_words=False):
    from sklearn.feature_extraction.text import \
        CountVectorizer, TfidfVectorizer

    if stop_words:
        from nltk.corpus import stopwords
        stop_w = stopwords.words('english')

    if mode == VM.COUNT:
        if lemma:
            if stop_words:
                vectorizer = ('CV', CountVectorizer(
                    tokenizer=LemmaTokenizer(), stop_words=stop_w))
            else:
                vectorizer = ('CV', CountVectorizer(
                    tokenizer=LemmaTokenizer()))
        else:
            if stop_words:
                vectorizer = ('CV', CountVectorizer(stop_words=stop_w))
            else:
                vectorizer = ('CV', CountVectorizer())
    elif mode == VM.TFIDF:
        if lemma:
            if stop_words:
                vectorizer = ('TFIDF', TfidfVectorizer(
                    tokenizer=LemmaTokenizer(), stop_words=stop_w))
            else:
                vectorizer = ('TFIDF', TfidfVectorizer(
                    tokenizer=LemmaTokenizer()))
        else:
            if stop_words:
                vectorizer = ('TFIDF', TfidfVectorizer(stop_words=stop_w))
            else:
                vectorizer = ('TFIDF', TfidfVectorizer())

    return vectorizer


def define_pipe(mode=VM.COUNT, lemma=False, stop_words=False):
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    vect = get_vectorizer(mode, lemma, stop_words)
    return Pipeline([vect, ('MNB', MultinomialNB())])


##### Classification #####

def naive_classification(train, test, pipe):
    pipe.fit(train.text, train.author)
    return pipe.predict(test.text)


def get_class_rep(train, test, predictions):
    from sklearn.metrics import classification_report

    author_labels = sorted(train.author.unique())
    res = classification_report(
        test.author, predictions, labels=author_labels, output_dict=True)
    return res


##### Over under-spamling #####
# TODO: Does not work -- check if possible
def over_undersampling(data):
    print("over-undersampling")
    from imblearn.combine import SMOTETomek
    rous = SMOTETomek(random_state=69, sampling_strategy='str')

    # Over-undersample
    temp_data = np.array(data.text).reshape(-1, 1)  # transpose
    data_text_ousample, data_author_ousample = \
        rous.fit_resample(temp_data, data.author)

    # convert to pandas series again
    data_text_ousample = pd.Series(data_text_ousample.flatten())

    # Add to new dataframe
    data_ousample = pd.DataFrame(
        {"text": data_text_ousample, "author": data_author_ousample})

    return data_ousample


##### Load and split data #####
# over under-sampling
# https://github.com/scikit-learn-contrib/imbalanced-learn
# TODO: They want citation in paper!


# Standard
# tfidf - 0.51
# count - 0.87
# count,lemma - 0.85

if False:
    data = read_data(filepath)
    print(data['text'].iloc[51082])

# TODO: look at this
# https://www.kaggle.com/colearninglounge/nlp-data-preprocessing-and-cleaning
