import nltk
from storage import *
from utils import VectorizationMode as VM
from utils import *
from baseline import *
from keras_code import *
import time
from pathlib import Path

##### Variables #####

# Select nr of authors to use from dataset to only use a subset of the data
NR_AUTHORS = 8

# Size of test dataset - Split data between train and test
SPLIT_SIZE = 0.2

# How many epochs to train for
EPOCHS = 6

# If plots should be shown
PLOT = False

##### Setup #####
print_header("Setup running")
s_time = time.time()

download_path = Path('E:/Projects/Author_Identification/data/nltk')
nltk.data.path.append(download_path)

nltk.download('punkt', download_dir=download_path)
nltk.download('wordnet', download_dir=download_path)
nltk.download('omw-1.4', download_dir=download_path)
nltk.download('stopwords', download_dir=download_path)

print("Setup Complete!")

##### Load and Pre-Process Data #####
print_header("Load and Process Data")
filepath = Path(
    'data/victorian_era/dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv')
data = read_data(filepath)

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

if PLOT:
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
    p.set_model(get_model(
        input_dim=p.get_transformed_x_train().shape[1],
        output_dim=p.get_transformed_y_train().shape[1]))

print("Create and set model for all profiles | Complete!")

# Compile and train model for all picked profiles
for p in picked_profiles:
    compile(p.get_model())
    history = fit(p.get_model(),
                  p.get_transformed_x_train(True),
                  p.get_transformed_y_train(),
                  EPOCHS)
    p.set_history(history)

print("Training complete for all profiles")


clear()  # Clear session - Reset weights of last training

print("Evaluate model performance on testdata:")
# Evaluate model for all picked profiles and save accuracy
[p.set_acc(evaluate(p.get_model(),
                    p.get_transformed_x_test(True),
                    p.get_transformed_y_test()))
 for p in picked_profiles]

print("Classification Complete!")

# Print results from classification for all picked profiles
print_header("Results")
print(f"Dataset with {len(authors_)} selected authors: "
      f"{[a for a in authors_]}\n")

best_profile = picked_profiles[0]
for p in picked_profiles:
    print("Profile:", str(p), "| Accuracy:", p.get_acc())
    if PLOT:
        plot_history(p.get_history())

    if p.get_acc() > best_profile.get_acc():
        best_profile = p

print("\nBest Profile was:", str(best_profile),
      "with accuracy:", best_profile.get_acc())

print(f"\nTook: {time.time()-s_time} seconds")

if False:
    plot_history(best_profile.get_history())
