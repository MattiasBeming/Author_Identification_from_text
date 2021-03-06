import nltk
from storage import *
from utils import VectorizationMode as VM
from utils import *
from baseline import *
import time
from pathlib import Path


##### Variables #####

# Dataset filepath
PATH_DS = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')

# Path to where the nltk-data should be stored
DL_PATH_NLTK = Path('E:/Projects/Author_Identification/data/nltk')

# Select nr of authors to use from dataset to only use a subset of the data
NR_AUTHORS = 8

# Size of test dataset - Split data between train and test
SPLIT_SIZE = 0.2

# If plots should be shown
PLOT_AUTHOR_DIST = False


##### Setup #####
s_time = time.time()
print_header("Setup running")

nltk.data.path.append(DL_PATH_NLTK)
nltk.download('punkt', download_dir=DL_PATH_NLTK)
nltk.download('wordnet', download_dir=DL_PATH_NLTK)
nltk.download('omw-1.4', download_dir=DL_PATH_NLTK)
nltk.download('stopwords', download_dir=DL_PATH_NLTK)

print("Setup Complete!")

##### Load and Pre-Process Data #####
print_header("Load and Process Data")

data = read_data(PATH_DS)

# Select subset of data to use
subset_data = get_rand_rows(data, NR_AUTHORS)

authors_ = np.unique(subset_data.author)
print(f"Selected {len(authors_)} authors: {[a for a in authors_]}")
print(f"{subset_data.shape[0]} out of {data.shape[0]} rows used "
      f"=> {100*(subset_data.shape[0]/data.shape[0]):.3}% of the data")

train, test = split_data(subset_data, SPLIT_SIZE)

##### Pre-process Data #####
data_os = oversample(subset_data)
train_os, test_os = split_data(data_os, SPLIT_SIZE)

data_us = undersample(subset_data)
train_us, test_us = split_data(data_us, SPLIT_SIZE)

if PLOT_AUTHOR_DIST:
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

picked_profiles = pick_multiple_profiles(profiles)

print("\nCreate Profiles Complete!")

if not picked_profiles:
    print("No profiles picked, exiting")
    exit()

##### Classification #####
print_header("Running classification")
s_time = time.time()

# Set pipe for all profiles
[p.set_baseline_pipe(define_pipe(p.get_mode(), p.is_lemmatized(),
                                 p.is_stop_words_removed())) for p in picked_profiles]

# Save predictions for all picked profiles
[p.set_preds(naive_classification(p.get_train(), p.get_test(), p.get_baseline_pipe()))
 for p in picked_profiles]

print("Classification Complete!")

# Save results from classification report for all picked profiles
[p.set_baseline_res(get_class_rep(p.get_train(), p.get_test(), p.get_predictions()))
 for p in picked_profiles]

# Print results from classification report for all picked profiles
print_header("Results")
print(f"Dataset with {len(authors_)} selected authors: "
      f"{[a for a in authors_]}\n")

best_profile = picked_profiles[0]
for p in picked_profiles:
    print("Profile:", str(p), "| Accuracy:", p.get_baseline_acc())
    if p.get_baseline_acc() > best_profile.get_baseline_acc():
        best_profile = p

print("\nBest Profile was:", str(best_profile),
      "with accuracy:", best_profile.get_baseline_acc())

print(f"\nTook: {time.time()-s_time} seconds")
