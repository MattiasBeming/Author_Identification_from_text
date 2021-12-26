
from storage import *
import matplotlib.pyplot as plt


##### Classification #####
def naive_classification(train, test):
    print("Classification")
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report

    if True:
        print("Count vectorizer")
        from sklearn.feature_extraction.text import CountVectorizer
        pipe = Pipeline([('CV', CountVectorizer()), ('MNB', MultinomialNB())])
    else:
        print("Tfidf vectorizer")
        from sklearn.feature_extraction.text import TfidfVectorizer
        pipe = Pipeline([('TFIDF', TfidfVectorizer()), ('MNB', MultinomialNB())])

    pipe.fit(train.text, train.author)

    predictions = pipe.predict(test.text)
    author_labels = sorted(train.author.unique())
    res = classification_report(test.author, predictions, labels=author_labels)

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

    data_text_oversample = pd.Series(data_text_oversample.flatten())  # convert to pandas series again

    # Add to new dataframe
    data_oversample = pd.DataFrame({"text": data_text_oversample, "author": data_author_oversample})

    return data_oversample

def undersample(data):
    print("Undersampling")
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=69)

    # Undersample
    temp_data = np.array(data.text).reshape(-1, 1)  # transpose
    data_text_undersample, data_author_undersample = \
        rus.fit_resample(temp_data, data.author)

    data_text_undersample = pd.Series(data_text_undersample.flatten())  # convert to pandas series again

    # Add to new dataframe
    data_undersample = pd.DataFrame({"text": data_text_undersample, "author": data_author_undersample})

    return data_undersample

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
    train, test = train_test_split(data_os, test_size=0.2)
    naive_classification(train, test)

# Undersampling
if False:
    data = read_data(filepath)
    data_us = undersample(data)
    train, test = train_test_split(data_us, test_size=0.2)
    naive_classification(train, test)

# Standard
if True:
    train, test = split_data(filepath, 0.2)
    naive_classification(train, test)
