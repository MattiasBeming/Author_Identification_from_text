
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

def define_pipe(mode=VM.COUNT, lemma=False, stop_words=False):
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    if stop_words:
        from nltk.corpus import stopwords
        stop_w = stopwords.words('english')

    if mode == VM.COUNT:
        from sklearn.feature_extraction.text import CountVectorizer

        if lemma:
            if stop_words:
                pipe = Pipeline([('CV', CountVectorizer(
                    tokenizer=LemmaTokenizer(), stop_words=stop_w)), ('MNB', MultinomialNB())])
            else:
                pipe = Pipeline(
                    [('CV', CountVectorizer(tokenizer=LemmaTokenizer())), ('MNB', MultinomialNB())])
        else:
            if stop_words:
                pipe = Pipeline(
                    [('CV', CountVectorizer(stop_words=stop_w)), ('MNB', MultinomialNB())])
            else:
                pipe = Pipeline([('CV', CountVectorizer()),
                                ('MNB', MultinomialNB())])

    elif mode == VM.TFIDF:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if lemma:
            if stop_words:
                pipe = Pipeline([('CV', TfidfVectorizer(
                    tokenizer=LemmaTokenizer(), stop_words=stop_w)), ('MNB', MultinomialNB())])
            else:
                pipe = Pipeline(
                    [('CV', TfidfVectorizer(tokenizer=LemmaTokenizer())), ('MNB', MultinomialNB())])
        else:
            if stop_words:
                pipe = Pipeline(
                    [('CV', TfidfVectorizer(stop_words=stop_w)), ('MNB', MultinomialNB())])
            else:
                pipe = Pipeline([('CV', TfidfVectorizer()),
                                ('MNB', MultinomialNB())])

    return pipe


##### Classification #####

def naive_classification_old(train, test, mode=VM.COUNT):
    print("Classification")
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report

    if mode == VM.COUNT:
        print("Count vectorizer")
        from sklearn.feature_extraction.text import CountVectorizer
        # pipe = Pipeline([('CV', CountVectorizer(tokenizer=LemmaTokenizer())), ('MNB', MultinomialNB())])
        pipe = Pipeline([('CV', CountVectorizer()), ('MNB', MultinomialNB())])

        pipe.fit(train.text, train.author)

        predictions = pipe.predict(test.text)
        author_labels = sorted(train.author.unique())
        res = classification_report(
            test.author, predictions, labels=author_labels)
    elif mode == VM.TFIDF:
        print("Tfidf vectorizer")
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vect = TfidfVectorizer()  # stop_words='english')

        train_tfidf = tfidf_vect.fit_transform(train.text)

        clf = MultinomialNB()
        clf.fit(train_tfidf, train.author)

        test_tfidf = tfidf_vect.transform(test.text)
        predictions = clf.predict(test_tfidf)

        author_labels = sorted(train.author.unique())
        res = classification_report(
            test.author, predictions, labels=author_labels)

    print(res)
    return res


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
