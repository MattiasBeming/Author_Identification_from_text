import nltk
from storage import *
from utils import VectorizationMode as VM
from utils import *
from baseline import *
from neuralnet_code import *
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

# Splits data when training from train dataset into train and validation
VALIDATION_SPLIT = 0.15

# How many epochs to train for
EPOCHS = 14

# Batch size to train/evaluate with
BS = 64

# If plots should be shown
PLOT_AUTHOR_DIST = False
PLOT_BEST_PROFILE = False
PLOT_HISTORY = True

##### Setup #####
print_header("Setup running")
s_time = time.time()

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

# Split data
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
# nr_authors = [4, 8, 12]

profiles.extend([
    Profile(ds_name, DS, l, sw, mode, NR_AUTHORS)
    for ds_name, _, __ in datasets
    for l in lemma
    for sw in stop_words
    for mode in modes
    # for nr_a in nr_authors
])

print(f"Created {len(profiles)} profiles")
print(f"Took: {time.time()-s_time} seconds\n")

picked_profiles = pick_multiple_profiles(profiles)

if not picked_profiles:
    exit()

print("\nCreate Profiles Complete!")

##### Classification #####
print_header("Running classification")
s_time = time.time()

# Set vectorizer for all profiles (which runs fit-transform)
for p in picked_profiles:
    name, vect = get_vectorizer(p.get_mode(),
                                p.is_lemmatized(),
                                p.is_stop_words_removed())
    p.set_vectorizer(vect)

print("Set vectorizer for all profiles | Complete!")

# Set model for all profiles
for p in picked_profiles:
    p.set_model(get_NN_model(
        input_dim=p.get_transformed_x_train().shape[1],
        output_dim=p.get_transformed_y_train().shape[1]))

print("Create and set model for all profiles | Complete!")

# Compile and train model for all picked profiles
for p in picked_profiles:
    compile(p.get_model())
    history = fit(p.get_model(),
                  p.get_transformed_x_train(True),
                  p.get_transformed_y_train(),
                  EPOCHS, BS, VALIDATION_SPLIT)
    p.set_history(history)

print("Training complete for all profiles")


clear()  # Clear session - Reset weights of last training

print("\nEvaluate model performance on testdata:")
# Evaluate model for all picked profiles and save accuracy
[p.set_acc(evaluate(p.get_model(),
                    p.get_transformed_x_test(True),
                    p.get_transformed_y_test(), BS))
 for p in picked_profiles]

# Print results from classification for all picked profiles
print_header("Results")
print(f"Dataset with {len(authors_)} selected authors: "
      f"{[a for a in authors_]}\n")

best_profile = picked_profiles[0]
for p in picked_profiles:
    print("Profile:", str(p), "| Accuracy:", p.get_acc())
    if PLOT_HISTORY:
        plot_history(p.get_history())

    if p.get_acc() > best_profile.get_acc():
        best_profile = p

print("\nBest Profile was:", str(best_profile),
      "with accuracy:", best_profile.get_acc())

print(f"\nTook: {time.time()-s_time} seconds")

if PLOT_BEST_PROFILE:
    plot_history(best_profile.get_history())
