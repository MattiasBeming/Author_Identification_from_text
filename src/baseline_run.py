import nltk
from storage import *
from utils import VectorizationMode as VM
from utils import Profile, read_data, split_data
from baseline import *
import time

s_time = time.time()

##### Setup #####
print("Setup running...")


download_path = Path('E:/Projects/Author_Identification/data/nltk')
nltk.data.path.append(download_path)

nltk.download('punkt', download_dir=download_path)
nltk.download('wordnet', download_dir=download_path)
nltk.download('omw-1.4', download_dir=download_path)
nltk.download('stopwords')


##### Load and Pre-Process Data #####
print("\nLoad and Process Data...")
filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')
data = read_data(filepath)

SPLIT_SIZE = 0.2
train, test = split_data(data, SPLIT_SIZE)

##### Pre-process Data #####
data_os = oversample(data)
train_os, test_os = split_data(data_os, SPLIT_SIZE)

data_us = undersample(data)
train_us, test_us = split_data(data_us, SPLIT_SIZE)

if False:
    bar_plot(train, "train")
    bar_plot(train_os, "train_os")
    bar_plot(train_us, "train_us")

##### Create Datasets #####
print("\nCreating Datasets...")

DS = Dataset("Victorian Era")

datasets = [
    ("standard", train, test),
    ("oversample", train_os, test_os),
    ("undersample", train_us, test_us)
]
[DS.add_dataset(ds_name, tr, te) for ds_name, tr, te in datasets]

##### Define Profiles #####
print("\nCreating Profiles...")

profiles = []

lemma = [False, True]
stop_words = [False, True]
modes = [VM.COUNT, VM.TFIDF]

profiles.extend([
    Profile(ds_name, DS, l, sw, mode)
    for ds_name, _, __ in datasets
    for l in lemma
    for sw in stop_words
    for mode in modes
])

print(f"Created {len(profiles)} profiles")
print(f"Took: {time.time()-s_time} seconds")

s_time = time.time()

##### Classification #####
print("\nRunning classification...")

# TEMP
profiles = [profiles[0]]

# Set pipe for all profiles
[p.set_pipe(define_pipe(p.get_mode(), p.is_lemmatized(),
            p.is_stop_words_removed())) for p in profiles]

# Set predictions for all profiles
[p.set_preds(naive_classification(p.get_train(), p.get_test(), p.get_pipe()))
 for p in profiles]

print("\nClassification done!")

# Set results from classification report for all profiles
[p.set_res(get_class_rep(p.get_train(), p.get_test(), p.get_predictions()))
 for p in profiles]

# Print results from classification report for all profiles
print("\nResults")
[print("Profile:", str(p), " -- Accuracy: ", p.get_acc()) for p in profiles]


print(f"Took: {time.time()-s_time} seconds")