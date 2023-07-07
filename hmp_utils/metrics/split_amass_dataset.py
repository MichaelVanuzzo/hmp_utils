import numpy as np
import math
import os

SEED = 130297

# In the sub datasets present in AMASS_DATASETS_BY_SUBJECT the train, validation and
# test sets are created by separating the subjects present into equal parts
AMASS_DATASETS_BY_SUBJECT = [
    "BMLmovi",
    "BMLrub",
    "DFaust",
    "EyesJapanDataset/Eyes_Japan_Dataset",
    "HDM05",
    "MoSh",
    "TotalCapture",
]

# In the sub datasets present in AMASS_DATASETS_BY_ACTION the train, validation and
# test sets are created by separating the actions present in the subfolders into equal parts
AMASS_DATASETS_BY_ACTION = [
    "ACCAD",
    "CMU",
    "EKUT",
    "SFU",
    "Transitions",
]


def get_sets_indexes_dict(
    samples_num: int,
    test_set_percentage: float = 5,
    valid_set_percentage: float = 5,
):
    """
    Returns a list of indices for the train, test and validation sets.
    The indices are shuffled randomly.
    ---
    ### Args:
    - `samples_num` (int): number of samples.
    - `test_set_percentage` (float): percentage of samples to be used for the test set.
    - `valid_set_percentage` (float): percentage of samples to be used for the validation set.
    """

    # Calculate the number of samples for each set
    num_test_samples = math.ceil(samples_num * (test_set_percentage / 100))
    num_valid_samples = math.ceil(samples_num * (valid_set_percentage / 100))
    num_train_samples = samples_num - num_test_samples - num_valid_samples

    # Create a list of all sample indices
    all_samples = np.arange(samples_num)

    # Set a seed for reproducibility
    rng = np.random.RandomState(SEED)
    # Shuffle the sample indices randomly
    rng.shuffle(all_samples)

    # Split the shuffled indices into train, test, and validation sets
    train_set_indexes = all_samples[:num_train_samples]
    test_set_indexes = all_samples[
        num_train_samples : num_train_samples + num_test_samples
    ]
    valid_set_indexes = all_samples[num_train_samples + num_test_samples :]

    return {
        "train": train_set_indexes,
        "test": test_set_indexes,
        "validation": valid_set_indexes,
    }


def get_mocap_lists_by_dataset(dataset_list: list):
    """
    Returns a list of lists of mocap files for each subject in each dataset.
    ---
    ### Args:
    - `dataset_list` (list): list of lists of subjects in each dataset.
    """
    mocap_lists_by_dataset = []
    # Iterate over each dataset in subject_lists_dict["action"]
    for dataset in dataset_list:
        mocap_list_by_subject = []
        # Iterate over each subject in the dataset
        for subject in dataset:
            # Use a generator expression along with os.scandir() to find all .npz files
            mocap_list = [
                entry.path
                for entry in os.scandir(subject)
                if entry.is_file()
                and entry.name.endswith(".npz")
                and "ei.npz" not in entry.name
            ]
            mocap_list_by_subject.append(mocap_list)
        mocap_lists_by_dataset.append(mocap_list_by_subject)
    return mocap_lists_by_dataset


def get_random_subject_list_by_dataset(dataset_list: list):
    """
    Returns a list of lists of subjects for each dataset.
    The subjects are shuffled randomly.
    ---
    ### Args:
    - `dataset_list` (list): list of lists of subjects in each dataset.
    """
    subject_list_by_dataset = []
    for dataset in dataset_list:
        rng = np.random.RandomState(SEED)
        dataset = sorted(dataset)
        rng.shuffle(dataset)
        subject_list_by_dataset.append(dataset)
    return subject_list_by_dataset


def get_random_mocap_list_by_dataset(dataset_list: list):
    """
    Returns a list of lists of mocap files for each subject in each dataset.
    The mocap files are shuffled randomly.
    ---
    ### Args:
    - `dataset_list` (list): list of lists of mocap files for each subject in each dataset.
    """
    mocap_lists_by_dataset = []
    for mocap_list_by_subject in dataset_list:
        rng = np.random.RandomState(SEED)
        shuffle_mocap_list_by_subject = []
        for mocap_list in mocap_list_by_subject:
            mocap_list = sorted(mocap_list)
            rng.shuffle(mocap_list)
            shuffle_mocap_list_by_subject.append(mocap_list)
        mocap_lists_by_dataset.append(shuffle_mocap_list_by_subject)
    return mocap_lists_by_dataset


def get_few_mocaps_subject_list(dataset: list):
    """
    Returns a list of subjects with less than half the mean number of mocaps.
    ---
    ### Args:
    - `dataset` (list): list of subjects in a dataset.
    """
    num_files = [
        sum(file.endswith(".npz") and "ei.npz" not in file for file in files)
        for subject in dataset
        for root, dirs, files in os.walk(subject)
    ]  # number of mocaps for each subject
    mean_num_files = np.mean(num_files)  # mean number of mocaps
    return [
        subject
        for subject in dataset
        for root, dirs, files in os.walk(subject)
        if sum(file.endswith(".npz") and "ei.npz" not in file for file in files)
        < mean_num_files * 0.5  # if the number of mocaps is less than half the mean
    ]


def split_amass_dataset(amass_directory: str = "/home/devlab/datasets/AMASS-SMPL-X"):
    """
    Returns a dictionary of dictionaries containing the train, test and validation sets
    for each sub dataset in AMASS_DATASETS_BY_SUBJECT and AMASS_DATASETS_BY_ACTION.
    ---
    ### Args:
    - `amass_directory` (str): path to the AMASS dataset root directory.
    """
    subdataset_lists = {
        "subject": AMASS_DATASETS_BY_SUBJECT,
        "action": AMASS_DATASETS_BY_ACTION,
    }

    subdataset_path_lists_dict = {
        key: [os.path.join(amass_directory, subdataset) for subdataset in value]
        for key, value in subdataset_lists.items()
    }

    subject_list_by_dataset_dict = {}
    for key, value in subdataset_path_lists_dict.items():
        subject_list_by_dataset = []
        for subdataset in value:
            subject_list_by_dataset.append(
                [
                    os.path.join(subdataset, dir)
                    for dir in os.listdir(subdataset)
                    if os.path.isdir(os.path.join(subdataset, dir))
                ]
            )
        subject_list_by_dataset_dict[key] = subject_list_by_dataset

    mocap_lists_by_dataset_dict = {}
    mocap_lists_by_dataset_dict["action"] = get_mocap_lists_by_dataset(
        subject_list_by_dataset_dict["action"]
    )

    subject_list_by_dataset_dict["subject"] = get_random_subject_list_by_dataset(
        subject_list_by_dataset_dict["subject"]
    )

    mocap_lists_by_dataset_dict["action"] = get_random_mocap_list_by_dataset(
        mocap_lists_by_dataset_dict["action"]
    )

    sets_by_dataset_dict = {"train": {}, "test": {}, "validation": {}}
    for subdataset_idx, dataset in enumerate(subject_list_by_dataset_dict["subject"]):
        sets_dict = {}
        few_mocaps_subjects = get_few_mocaps_subject_list(dataset)
        dataset = [subject for subject in dataset if subject not in few_mocaps_subjects]

        sets_indexes_dict = get_sets_indexes_dict(len(dataset))

        for key, value in sets_indexes_dict.items():
            sets_dict[key] = [dataset[index] for index in value]
        sets_dict["train"].extend(few_mocaps_subjects)

        mocap_lists_by_set = get_mocap_lists_by_dataset(sets_dict.values())

        sets_keys = list(sets_dict.keys())
        sets_dict = {}
        for idx, mocap_list_by_subject in enumerate(mocap_lists_by_set):
            sets_dict[sets_keys[idx]] = [
                item for mocap_list in mocap_list_by_subject for item in mocap_list
            ]

        subdatset_name = subdataset_lists["subject"][subdataset_idx]
        for key, value in sets_dict.items():
            sets_by_dataset_dict[key][subdatset_name] = value

    for subdataset_idx, mocap_list_by_subject in enumerate(
        mocap_lists_by_dataset_dict["action"]
    ):
        sets_dict = {key: [] for key in sets_by_dataset_dict.keys()}
        for mocap_list in mocap_list_by_subject:
            if len(mocap_list) < 3:
                sets_dict["train"].extend(mocap_list)
            else:
                sets_indexes_dict = get_sets_indexes_dict(len(mocap_list))
                for key, value in sets_indexes_dict.items():
                    sets_dict[key].extend([mocap_list[index] for index in value])

        subdatset_name = subdataset_lists["action"][subdataset_idx]
        for key, value in sets_dict.items():
            sets_by_dataset_dict[key][subdatset_name] = value
