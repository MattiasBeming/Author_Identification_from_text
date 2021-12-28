
from storage import *
import matplotlib.pyplot as plt
from utils import VectorizationMode as VM


##### Classification #####

def naive_classification(train, test, mode=VM.COUNT):
    print("Classification")
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report

    if mode == VM.COUNT:
        print("Count vectorizer")
        from sklearn.feature_extraction.text import CountVectorizer
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


##### Over under-spamling #####

def oversample(data):
    print("Oversampling")
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
    print("Undersampling")
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

##### Bar plots #####


def bar_plot(data, title=""):
    data.author.value_counts().plot(kind='bar')
    plt.title(title)
    plt.show()
    return


##### Load and split data #####
# over under-sampling
# https://github.com/scikit-learn-contrib/imbalanced-learn
# TODO: They want citation in paper!

filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

# Oversampling
if False:
    data = read_data(filepath)
    data_os = oversample(data)
    train, test = train_test_split(
        data_os, test_size=0.2, stratify=data_os.author)
    naive_classification(train, test)
    bar_plot(data_os)
    naive_classification(train, test)

# Undersampling
if False:
    data = read_data(filepath)
    data_us = undersample(data)
    train, test = train_test_split(
        data_us, test_size=0.2, stratify=data_us.author)
    naive_classification(train, test)

# Over-undersampling
if False:
    data = read_data(filepath)
    data_ous = over_undersampling(data)
    train, test = train_test_split(
        data_ous, test_size=0.2, stratify=data_ous.author)
    naive_classification(train, test)

# Standard
if True:
    train, test = split_data(filepath, 0.2)
    # print(train.iloc[0:1])
    naive_classification(train, test, VM.TFIDF)


if False:
    data = read_data(filepath)
    print(data['text'].iloc[51082])

# TODO: look at this
# https://www.kaggle.com/colearninglounge/nlp-data-preprocessing-and-cleaning

if False:
    from sklearn.feature_extraction.text import TfidfVectorizer
    train, test = split_data(filepath, 0.2)

    tfidf_vect = TfidfVectorizer()  # stop_words='english')
    train_tfidf = tfidf_vect.fit_transform(train.text)
    print(train_tfidf)
