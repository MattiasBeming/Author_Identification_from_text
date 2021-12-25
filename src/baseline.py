
from storage import *
import matplotlib.pyplot as plt

##### Load and split data #####

filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

train, test = split_data(filepath)

# over under-sampling 
# https://github.com/scikit-learn-contrib/imbalanced-learn
# TODO: They want citation in paper!

##### Bar plots #####

def bar_plot(data, title=""):
  data.author.value_counts().plot(kind='bar')
  plt.title(title)
  plt.show()
  return

if False:
    bar_plot(train, "Train")
    bar_plot(test, "Test")

##### Over under-spamling #####

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=69)
# TODO: check if it works with text, and look for other methods
train_text_resampled, train_author_resampled = sm.fit_resample(train.text, train.author)
# train_text_resampled = pd.Series(train_text_resampled.flatten())
print(type(train_text_resampled))
print(type(train_author_resampled))
train_resampled = pd.DataFrame({"text": train_text_resampled, "author": train_author_resampled})

print("before: ", train.author.value_counts())
print("After: ", train_resampled.author.value_counts())
bar_plot(train_resampled, "Train Resampled")

exit()
##### Classification #####

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

pipe = Pipeline([('CV', CountVectorizer()), ('MNB', MultinomialNB())])
pipe.fit(train.text, train.author)

predictions = pipe.predict(test.text)
author_labels = sorted(train.author.unique())
res = classification_report(test.author, predictions, labels=author_labels)

print(res)
