import nltk
from storage import *
from utils import VectorizationMode as VM
from utils import *
from baseline import *
import time
from pathlib import Path

s_time = time.time()

##### Setup #####
print_header("Setup running")


download_path = Path('E:/Projects/Author_Identification/data/nltk')
nltk.data.path.append(download_path)

nltk.download('punkt', download_dir=download_path)
nltk.download('wordnet', download_dir=download_path)
nltk.download('omw-1.4', download_dir=download_path)
nltk.download('stopwords')

print("Setup Complete!")

##### Load and Pre-Process Data #####
print_header("Load and Process Data")
filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')
data = read_data(filepath)

# TODO: function for selecting authors per dataset? 
# or part of profiles?

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

print("Load and Process Data Complete!")

##### Create Datasets #####
print_header("Creating Datasets")

DS = Dataset("Victorian Era")

datasets = [
    ("standard", train, test),
    ("oversample", train_os, test_os),
    ("undersample", train_us, test_us)
]
[DS.add_dataset(ds_name, tr, te) for ds_name, tr, te in datasets]

print("Create Datasets Complete!")

##### Define Profiles #####
print_header("Create Profiles")

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
print(f"Took: {time.time()-s_time} seconds\n")

s_time = time.time()

picked_profiles = pick_multiple_profiles(profiles)

print("\nCreate Profiles Complete!")

##### Classification #####
print_header("Running classification")

# Set pipe for all profiles
[p.set_pipe(define_pipe(p.get_mode(), p.is_lemmatized(),
            p.is_stop_words_removed())) for p in picked_profiles]

# Save predictions for all picked profiles
[p.set_preds(naive_classification(p.get_train(), p.get_test(), p.get_pipe()))
 for p in picked_profiles]

print("Classification Complete!")

# Save results from classification report for all picked profiles
[p.set_res(get_class_rep(p.get_train(), p.get_test(), p.get_predictions()))
 for p in picked_profiles]

# Print results from classification report for all picked profiles
print_header("Results")
[print("Profile:", str(p), "| Accuracy:", p.get_acc()) for p in picked_profiles]

print(f"\nTook: {time.time()-s_time} seconds")
