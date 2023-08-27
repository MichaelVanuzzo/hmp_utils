import numpy as np
import math
import os
import json

from typing import Optional
from tqdm import tqdm

from .. import MocapLoader, BodyModelType

SEED = 130297
TEST_SET_PERCENTAGE = 5
VALID_SET_PERCENTAGE = 5
TARGET_FRAMERATE = 60
MIN_MOCAP_FRAMES = TARGET_FRAMERATE * 3  # 3 seconds

# In the sub datasets present in AMASS_DATASETS_BY_SUBJECT the train, validation and
# test sets are created by separating the subjects present into equal parts
AMASS_DATASETS_BY_SUBJECT = [
    "BMLmovi",
    "BMLrub",
    "EyesJapanDataset",
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

AMASS_DATASETS = sorted(AMASS_DATASETS_BY_SUBJECT + AMASS_DATASETS_BY_ACTION)


def get_sets_indexes_dict(
    samples_num: int,
    test_set_percentage: float = TEST_SET_PERCENTAGE,
    valid_set_percentage: float = VALID_SET_PERCENTAGE,
) -> dict:
    """
    Returns a dictionary containing a list of indices for the train, test and validation sets.
    The indices are shuffled randomly.

    ---
    ### Args:
    - `samples_num` (int): number of samples.
    - `test_set_percentage` (float): percentage of samples to be used for the test set.
    - `valid_set_percentage` (float): percentage of samples to be used for the validation set.
    ---
    ### Returns:
    - `sets_indexes_dict` (dict): dictionary containing a list of indices for the train, test and validation sets.
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


def is_valid_mocap(mocap_str: str) -> bool:
    """
    Returns True if the mocap file name is valid, otherwise False.

    ---
    ### Args:
    - `mocap_str` (str): mocap file name.
    ---
    ### Returns:
    - `is_valid` (bool): True if the mocap file name is valid, otherwise False.
    """
    return (
        mocap_str.endswith(".npz")
        and "stagei.npz" not in mocap_str
        and "shape.npz" not in mocap_str
    )


def get_num_frames(mocap_path: str, mocap_loader: MocapLoader) -> int:
    """
    Returns the number of frames in a mocap file if it has enough frames to be downsampled, otherwise -1.

    ---
    ### Args:
    - `mocap_path` (str): path to the mocap file.
    - `mocap_loader` (MocapLoader): mocap loader.
    ---
    ### Returns:
    - `num_frames` (int): number of frames in the mocap file.
    """
    mocap = mocap_loader.load_mocap(mocap_path)
    if mocap is None:
        print("Error loading", mocap_path, "(not enough frames)")
        return -1  # if the mocap file doesn't have enough frames

    return mocap.trans.shape[0]  # trans: translation of root joint (num_frames, 1, 3)


def get_mocap_lists_by_dataset(dataset_list: list, mocap_loader: MocapLoader) -> list:
    """
    Returns a list of lists of mocap files for each subject in each dataset.

    ---
    ### Args:
    - `dataset_list` (list): list of lists of subjects in each dataset.
    - `mocap_loader` (MocapLoader): mocap loader.
    ---
    ### Returns:
    - `mocap_lists_by_dataset` (list): list of lists of mocap files for each subject in each dataset.
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
                and is_valid_mocap(entry.name)
                and get_num_frames(entry.path, mocap_loader) >= MIN_MOCAP_FRAMES
            ]
            mocap_list_by_subject.append(mocap_list)
        mocap_lists_by_dataset.append(mocap_list_by_subject)
    return mocap_lists_by_dataset


def get_random_subject_list_by_dataset(dataset_list: list) -> list:
    """
    Returns a list of lists of subjects for each dataset.
    The subjects are shuffled randomly.

    ---
    ### Args:
    - `dataset_list` (list): list of lists of subjects in each dataset.
    ---
    ### Returns:
    - `subject_list_by_dataset` (list): list of lists of subjects for each dataset.
    """
    subject_list_by_dataset = []
    for dataset in dataset_list:
        rng = np.random.RandomState(SEED)
        dataset = sorted(dataset)
        rng.shuffle(dataset)
        subject_list_by_dataset.append(dataset)
    return subject_list_by_dataset


def get_random_mocap_list_by_dataset(dataset_list: list) -> list:
    """
    Returns a list of lists of mocap files for each subject in each dataset.
    The mocap files are shuffled randomly.

    ---
    ### Args:
    - `dataset_list` (list): list of lists of mocap files for each subject in each dataset.
    ---
    ### Returns:
    - `mocap_lists_by_dataset` (list): list of lists of mocap files for each subject in each dataset.
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


def get_few_mocaps_subject_list(dataset: list) -> list:
    """
    Returns a list of subjects with less than half the mean number of mocaps.

    ---
    ### Args:
    - `dataset` (list): list of subjects in a dataset.
    ---
    ### Returns:
    - `few_mocaps_subjects` (list): list of subjects with less than half the mean number of mocaps.
    """
    num_files = [
        sum(is_valid_mocap(file) for file in files)
        for subject in dataset
        for root, dirs, files in os.walk(subject)
    ]  # number of mocaps for each subject
    mean_num_files = np.mean(num_files)  # mean number of mocaps
    return [
        subject
        for subject in dataset
        for root, dirs, files in os.walk(subject)
        if sum(is_valid_mocap(file) for file in files)
        < mean_num_files * 0.5  # if the number of mocaps is less than half the mean
    ]


def split_amass_dataset(
    amass_directory: str,
    sets_by_dataset_dict_file_name: Optional[str] = None,
    use_relative_path: bool = True,
    test_set_percentage: float = TEST_SET_PERCENTAGE,
    valid_set_percentage: float = VALID_SET_PERCENTAGE,
) -> dict:
    """
    Returns a dictionary of dictionaries containing the train, test and validation sets
    for each sub dataset in AMASS_DATASETS_BY_SUBJECT and AMASS_DATASETS_BY_ACTION.

    If `sets_by_dataset_dict_file_name` is not None, the dictionary is saved to a file.

    ---
    ### Args:
    - `amass_directory` (str): absolute path to the AMASS dataset root directory.
    - `sets_by_dataset_dict_file_name` (str)(optional): absolute path and name (without .json) of
        the file containing the dictionary. If None, the dictionary is not saved to a file.
        Example: '/path/to/folder/sets_by_dataset_dict'.
    - `use_relative_path` (bool): if True, the mocap paths in the dictionary are relative (default: True).
    - `test_set_percentage` (float): percentage of samples to be used for the test set (default: 5).
    - `valid_set_percentage` (float): percentage of samples to be used for the validation set (default: 5).
    ---
    ### Returns:
    - `sets_by_dataset_dict` (dict): dictionary of dictionaries containing the train,
        test and validation sets for each sub dataset in AMASS_DATASETS_BY_SUBJECT and
        AMASS_DATASETS_BY_ACTION.
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
                    dirpath
                    for dirpath, _, filenames in os.walk(subdataset)
                    if any(filename.endswith(".npz") for filename in filenames)
                ]
            )
        subject_list_by_dataset_dict[key] = subject_list_by_dataset

    mocap_loader = MocapLoader(
        body_model_type=BodyModelType.SMPL,
        keypoint_ids_to_load=[0],
        target_framerate=TARGET_FRAMERATE,
    )
    mocap_lists_by_dataset_dict = {}
    mocap_lists_by_dataset_dict["action"] = get_mocap_lists_by_dataset(
        subject_list_by_dataset_dict["action"], mocap_loader
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

        sets_indexes_dict = get_sets_indexes_dict(
            len(dataset),
            test_set_percentage=test_set_percentage,
            valid_set_percentage=valid_set_percentage,
        )

        for key, value in sets_indexes_dict.items():
            sets_dict[key] = [dataset[index] for index in value]
        sets_dict["train"].extend(few_mocaps_subjects)

        mocap_lists_by_set = get_mocap_lists_by_dataset(
            sets_dict.values(), mocap_loader
        )

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
                sets_indexes_dict = get_sets_indexes_dict(
                    len(mocap_list),
                    test_set_percentage=test_set_percentage,
                    valid_set_percentage=valid_set_percentage,
                )
                for key, value in sets_indexes_dict.items():
                    sets_dict[key].extend([mocap_list[index] for index in value])

        subdatset_name = subdataset_lists["action"][subdataset_idx]
        for key, value in sets_dict.items():
            sets_by_dataset_dict[key][subdatset_name] = value

    if sets_by_dataset_dict_file_name != None:
        if use_relative_path:
            for key, value in sets_by_dataset_dict.items():
                for subdataset_name, mocap_list in value.items():
                    sets_by_dataset_dict[key][subdataset_name] = [
                        os.path.relpath(mocap_path, amass_directory)
                        for mocap_path in mocap_list
                    ]

        # Convert the dictionary to JSON format
        json_data = json.dumps(sets_by_dataset_dict)

        # Create the directories if they don't exist
        directory = os.path.dirname(sets_by_dataset_dict_file_name)
        os.makedirs(directory, exist_ok=True)

        # Save the JSON to a file
        with open(sets_by_dataset_dict_file_name + ".json", "w") as file:
            file.write(json_data)

    return sets_by_dataset_dict


def load_sets_by_dataset_dict(
    sets_by_dataset_dict_file_name: str, amass_directory: Optional[str] = None
) -> dict:
    """
    Returns a dictionary of dictionaries containing the train, test and validation sets
    for each sub dataset in AMASS_DATASETS_BY_SUBJECT and AMASS_DATASETS_BY_ACTION.

    ---
    ### Args:
    - `sets_by_dataset_dict_file_name` (str): absolute path and name (without .json) of
        the file containing the dictionary.
        Example: '/path/to/folder/sets_by_dataset_dict'.
    - `amass_directory` (str)(optional): absolute path to the AMASS dataset root directory if the
        mocap paths in the dictionary are relative.
    ---
    ### Returns:
    - `sets_by_dataset_dict` (dict): dictionary of dictionaries containing the train,
        test and validation sets for each sub dataset in AMASS_DATASETS_BY_SUBJECT and
        AMASS_DATASETS_BY_ACTION.
    """
    with open(sets_by_dataset_dict_file_name + ".json", "r") as file:
        json_data = file.read()
    sets_by_dataset_dict = json.loads(json_data)

    if amass_directory != None:
        for key, value in sets_by_dataset_dict.items():
            for subdataset_name, mocap_list in value.items():
                sets_by_dataset_dict[key][subdataset_name] = [
                    os.path.join(amass_directory, mocap_path)
                    for mocap_path in mocap_list
                ]
    return sets_by_dataset_dict


def compute_mocap_info(dataset_name: str, mocap_info_by_set_dict: dict) -> dict:
    """
    Returns a dictionary containing the number of mocaps, frames, and other informations
    joining the sets of a dataset.

    ---
    ### Args:
    - `dataset_name` (str): name of the dataset.
    - `mocap_info_by_set_dict` (dict): dictionary of dictionaries containing the number of
    mocaps and frames for each set in a dataset.
    ---
    ### Returns:
    - `mocap_info` (dict): dictionary containing the number of mocaps, frames, and other informations
    """
    dataset_sets_info = [set[dataset_name] for set in mocap_info_by_set_dict.values()]
    return {
        "num_mocaps": sum([set_info["num_mocaps"] for set_info in dataset_sets_info]),
        "num_frames": sum([set_info["num_frames"] for set_info in dataset_sets_info]),
        "min_num_frames": min(
            [set_info["min_num_frames"] for set_info in dataset_sets_info]
        ),
        "max_num_frames": max(
            [set_info["max_num_frames"] for set_info in dataset_sets_info]
        ),
        "mean_num_frames": np.average(
            [set_info["mean_num_frames"] for set_info in dataset_sets_info],
            weights=[set_info["num_mocaps"] for set_info in dataset_sets_info],
        ),
    }


def print_amass_sets_info(sets_by_dataset_dict: dict) -> None:
    """
    Prints information about the train, test and validation sets for each sub dataset
    in AMASS_DATASETS_BY_SUBJECT and AMASS_DATASETS_BY_ACTION.

    ---
    ### Args:
    - `sets_by_dataset_dict` (dict): dictionary of dictionaries containing the train,
    test and validation sets for each sub dataset in AMASS_DATASETS_BY_SUBJECT and
    AMASS_DATASETS_BY_ACTION.
    """
    mocap_info_by_set_dict = {"train": {}, "test": {}, "validation": {}}
    mocap_info_dict = {}

    mocap_loader = MocapLoader(
        body_model_type=BodyModelType.SMPL,
        keypoint_ids_to_load=[0],
        target_framerate=TARGET_FRAMERATE,
    )
    for dataset_name in AMASS_DATASETS:
        for set_name, set in sets_by_dataset_dict.items():
            dataset = set[dataset_name]
            list_of_num_frames = [
                get_num_frames(mocap, mocap_loader)
                for mocap in tqdm(
                    dataset,
                    desc=f"Getting number of frames in {dataset_name}/{set_name}",
                )
            ]
            mocap_info_by_set_dict[set_name][dataset_name] = {
                "num_mocaps": len(list_of_num_frames),
                "num_frames": sum(list_of_num_frames),
                "min_num_frames": min(list_of_num_frames),
                "max_num_frames": max(list_of_num_frames),
                "mean_num_frames": np.mean(list_of_num_frames),
                "std_num_frames": np.std(list_of_num_frames),
            }
        mocap_info_dict[dataset_name] = compute_mocap_info(
            dataset_name, mocap_info_by_set_dict
        )

    for set_name, set in mocap_info_by_set_dict.items():
        datasets_info = [set[dataset_name] for dataset_name in AMASS_DATASETS]
        mocap_info_by_set_dict[set_name]["AMASS"] = {
            "num_mocaps": sum(
                [dataset_info["num_mocaps"] for dataset_info in datasets_info]
            ),
            "num_frames": sum(
                [dataset_info["num_frames"] for dataset_info in datasets_info]
            ),
            "min_num_frames": min(
                [dataset_info["min_num_frames"] for dataset_info in datasets_info]
            ),
            "max_num_frames": max(
                [dataset_info["max_num_frames"] for dataset_info in datasets_info]
            ),
            "mean_num_frames": np.average(
                [dataset_info["mean_num_frames"] for dataset_info in datasets_info],
                weights=[dataset_info["num_mocaps"] for dataset_info in datasets_info],
            ),
        }
    mocap_info_dict["AMASS"] = compute_mocap_info("AMASS", mocap_info_by_set_dict)

    for dataset_name in ["AMASS"] + AMASS_DATASETS:
        print(f"Dataset name: {dataset_name}")

        mocap_info = mocap_info_dict[dataset_name]
        tot_num_mocaps = mocap_info["num_mocaps"]
        tot_num_frames = mocap_info["num_frames"]
        print(f"  Number of mocaps: {tot_num_mocaps}")
        print(f"  Number of frames: {tot_num_frames}")
        print(
            f"  Total number of minutes: {mocap_info['num_frames']/TARGET_FRAMERATE/60:.3f}"
        )
        print(f"  Min number of frames: {mocap_info['min_num_frames']}")
        print(f"  Max number of frames: {mocap_info['max_num_frames']}")
        print(f"  Mean number of frames: {mocap_info['mean_num_frames']:.3f}")
        print()

        for set_name, set in mocap_info_by_set_dict.items():
            mocap_info = set[dataset_name]
            print(f"  Set name: {set_name}")
            print(f"    Number of mocaps: {mocap_info['num_mocaps']}")
            print(
                f"    Percentage of mocaps: {mocap_info['num_mocaps']/tot_num_mocaps*100:.3f}%"
            )
            print(f"    Number of frames: {mocap_info['num_frames']}")
            print(
                f"    Total number of minutes: {mocap_info['num_frames']/TARGET_FRAMERATE/60:.3f}"
            )
            print(
                f"    Percentage of frames: {mocap_info['num_frames']/tot_num_frames*100:.3f}%"
            )
            print(f"    Min number of frames: {mocap_info['min_num_frames']}")
            print(f"    Max number of frames: {mocap_info['max_num_frames']}")
            print(f"    Mean number of frames: {mocap_info['mean_num_frames']:.3f}")
            if dataset_name != "AMASS":
                print(f"    Std number of frames: {mocap_info['std_num_frames']:.3f}")
        print()
