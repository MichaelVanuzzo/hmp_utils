import numpy as np
import math
import os
import json

from tqdm import tqdm

from .. import MocapLoader, SmplConstants, MocapData

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

SEED = 130297
TRAIN_SET_PERCENTAGE = 90
TEST_SET_PERCENTAGE = 5
VALID_SET_PERCENTAGE = 5
TARGET_FRAMERATE = 25
SOURCE_SEQ_LEN = 50
TARGET_SEQ_LEN = 25
MIN_MOCAP_FRAMES = SOURCE_SEQ_LEN + TARGET_SEQ_LEN
MOCAP_WINDOW_STEP = 50
PCA_COMPONENTS = 50
MIN_CLUSTERS = 30
MAX_CLUSTERS = 60

AMASS_DATASETS = [
    # "ACCAD",
    # "GRAB",
    "DanceDB",
    # "BMLmovi",
    # "BMLrub",
    # "EyesJapanDataset",
    # "HDM05",
    # "MoSh",
    # "TotalCapture",
    # "CMU",
    # "EKUT",
    # "SFU",
    # "Transitions",
]


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
    return mocap_str.endswith(".npz") and "stagei.npz" not in mocap_str and "shape.npz" not in mocap_str


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


def get_mocaps_list(dataset_list: list, mocap_loader: MocapLoader) -> list:
    """
    Returns the list of mocaps in the given dataset .

    ---
    ### Args:
    - `dataset_list` (list): list of lists of subjects in each dataset.
    - `mocap_loader` (MocapLoader): mocap loader.
    ---
    ### Returns:
    - `mocaps_list` (list): list of all mocap files in the given dataset.
    """
    mocaps_list = []
    # Iterate over each dataset in subject_lists_dict["action"]
    for dataset in tqdm(dataset_list, desc=f"Loading datasets"):
        mocap_list_by_subject = []
        # Iterate over each subject in the dataset
        for subject in dataset:
            # Use a generator expression along with os.scandir() to find all .npz files
            mocaps_list = mocaps_list + [
                entry.path
                for entry in os.scandir(subject)
                if entry.is_file() and is_valid_mocap(entry.name) and get_num_frames(entry.path, mocap_loader) >= MIN_MOCAP_FRAMES
            ]
    return mocaps_list


def split_amass_by_clustering(
    amass_directory: str,
    save_mocaps_dir: str,
    save_json_dir: str,
    use_relative_path: bool = True,
    train_set_percentage: float = TRAIN_SET_PERCENTAGE,
    test_set_percentage: float = TEST_SET_PERCENTAGE,
    valid_set_percentage: float = VALID_SET_PERCENTAGE,
) -> dict:
    """
    Returns a dictionary of dictionaries containing the train, test and validation sets
    which are obtained by extracting from the original mocaps different windows. Windows
    are split based on a clustering algorithm.


    ---
    ### Args:
    - `amass_directory` (str): absolute path to the AMASS dataset root directory.
    - `save_mocaps_dir` (str): absolute path where to save the extracted mocap windows.
    - `save_json_dir` (str): absolute path where to save the dictionary as .json files.
    - `train_set_percentage` (float): percentage of samples to be used for the train set (default: 90).
    - `test_set_percentage` (float): percentage of samples to be used for the test set (default: 5).
    - `valid_set_percentage` (float): percentage of samples to be used for the validation set (default: 5).
    ---
    ### Returns:
    - `dataset_dict` (dict): dictionary of dictionaries containing the train,
        test and validation sets.
    """

    subdataset_path_lists = [os.path.join(amass_directory, subdataset) for subdataset in AMASS_DATASETS]

    subject_list_by_dataset = []
    for subdataset in subdataset_path_lists:
        subject_list_by_dataset.append([dirpath for dirpath, _, filenames in os.walk(subdataset) if any(filename.endswith(".npz") for filename in filenames)])
    mocap_loader = MocapLoader(
        body_model_type=SmplConstants.BODY_MODEL_TYPE,
        keypoint_ids_to_load=SmplConstants.KEYPOINTS,
        target_framerate=TARGET_FRAMERATE,
    )

    # get the complete list of mocaps
    mocaps_list = get_mocaps_list(subject_list_by_dataset, mocap_loader)

    # break mocap into windows
    mocap_windows_list = []
    for mocap_path in tqdm(mocaps_list, desc=f"Splitting mocaps into windows"):
        mocap = mocap_loader.load_mocap(mocap_path)
        mocap_name = "_".join(mocap_path.split("/")[-3:])
        mocap_name = mocap_name[: len(mocap_name) - 4]

        mocap_windows_list = mocap_windows_list + split_mocap_into_windows(mocap, mocap_name)

    print("{:30s} {:d}".format("Total number of mocaps:", len(mocaps_list)))
    print("{:30s} {:d}".format("Number of windows extracted:", len(mocap_windows_list)))

    # get the mocap embeddings through PCA
    mocap_embeddings = get_mocap_embeddings(mocap_windows_list)

    # cluster mocaps
    mocap_labels, n_clusters = get_mocap_labels(mocap_embeddings)

    # split mocap windows into train, test, and valid based on clustering labels
    train_split, test_split, valid_split = split_dataset(
        mocap_windows_list,
        mocap_labels,
        n_clusters,
        train_set_percentage,
        test_set_percentage,
        valid_set_percentage,
    )

    print("{:30s} {:d}".format("Training set size:", len(train_split)))
    print("{:30s} {:d}".format("Test set size:", len(test_split)))
    print("{:30s} {:d}".format("Valid set size:", len(valid_split)))

    # save mocap windows and generate the dictionary
    dataset_dict = save_dataset_split(train_split, test_split, valid_split, save_mocaps_dir)

    # convert the dictionary to JSON format and save it
    json_data = json.dumps(dataset_dict)
    os.makedirs(save_json_dir, exist_ok=True)
    save_json_path = os.path.join(save_json_dir, "amass_clustering_split")
    with open(save_json_path + ".json", "w") as file:
        file.write(json_data)

    return dataset_dict


def split_mocap_into_windows(mocap, mocap_name):
    """
    Returns the list of windows extracted from a given mocap .

    ---
    ### Args:
    - `mocap` (MocapData): mocap to split into windows.
    - `mocap_name` (str): mocap id.
    ---
    ### Returns:
    - `mocap_windows_list` (list): list of all windows extracted from the given mocap.
    """
    rots = mocap.rots
    trans = mocap.trans
    mocap_windows_list = []
    # extract all possible windows and create a new mocap for each window
    index = 0
    window_len = SOURCE_SEQ_LEN + TARGET_SEQ_LEN
    while index * MOCAP_WINDOW_STEP + window_len < rots.shape[0]:
        new_rots = rots[index * MOCAP_WINDOW_STEP : index * MOCAP_WINDOW_STEP + window_len]
        new_trans = trans[index * MOCAP_WINDOW_STEP : index * MOCAP_WINDOW_STEP + window_len]
        mocap_windows_list.append(
            [
                MocapData(
                    new_rots,
                    new_trans,
                    mocap.betas,
                    TARGET_FRAMERATE,
                    TARGET_FRAMERATE,
                    gender=mocap.gender,
                ),
                mocap_name + "_" + str(index),
            ]
        )
        index += 1
    return mocap_windows_list


def get_mocap_embeddings(mocaps_list):
    """
    Returns the list of embeddings for each mocap in the given list.

    ---
    ### Args:
    - `mocaps_list` (list): list of mocaps.
    ---
    ### Returns:
    - `mocap_embeddings_list` (list): list of mocap embeddings.
    """
    mocaps_raw_embeddings = []

    for mocap in mocaps_list:
        # create a raw embedding concatenating each frame representation
        rots = mocap[0].rots[:, SmplConstants.MAJOR_JOINTS, :]
        rots = np.reshape(rots, (-1, len(SmplConstants.MAJOR_JOINTS) * 3))
        raw_embedd = np.reshape(
            rots,
            ((SOURCE_SEQ_LEN + TARGET_SEQ_LEN) * len(SmplConstants.MAJOR_JOINTS) * 3),
        )
        mocaps_raw_embeddings.append(raw_embedd)

    # apply PCA to reduce embedding dimensionality
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    pca.fit(mocaps_raw_embeddings)
    mocap_embeddings_list = pca.transform(mocaps_raw_embeddings)

    return mocap_embeddings_list


def get_mocap_labels(mocap_embeddings):
    """
    Returns the list of labels associated to each mocap. Labels are generated
    using a clustering algorithm based on mocap embeddings.

    ---
    ### Args:
    - `mocap_embeddings` (list): list of mocap embeddings.
    ---
    ### Returns:
    - `mocap_labels` (list): list of mocap labels.
    - `best_n_clusters` (int): the optimal number of clusters.
    """
    # find the optimal number of clusters in this range
    clusters_range = range(MIN_CLUSTERS, MAX_CLUSTERS)

    best_score = -1
    best_n_clusters = 0

    for n_clusters in tqdm(clusters_range, desc=f"Clustering mocap windows"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        cluster_labels = kmeans.fit(mocap_embeddings).labels_
        score = silhouette_score(mocap_embeddings, cluster_labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    # get the optimal labels
    print("{:30s} {:d}".format("Optimal number of clusters:", best_n_clusters))
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=0, n_init="auto")
    mocap_labels = kmeans.fit(mocap_embeddings).labels_

    return mocap_labels, best_n_clusters


def split_dataset(
    mocaps_list,
    mocap_labels,
    n_clusters,
    train_set_percentage,
    test_set_percentage,
    valid_set_percentage,
):
    """
    Returns train, test and validation splits based on custom percentages and on mocap labels.

    ---
    ### Args:
    - `mocaps_list` (list): list of mocaps to split.
    - `mocap_labels` (list): list of mocap labels.
    - `n_clusters` (int): total number of different clusters.
    - `train_set_percentage` (float): percentage of samples to be used for the train set (default: 90).
    - `test_set_percentage` (float): percentage of samples to be used for the test set (default: 5).
    - `valid_set_percentage` (float): percentage of samples to be used for the validation set (default: 5).
    ---
    ### Returns:
    - `split_train` (list): list of training set mocaps.
    - `split_test` (list): list of test set mocaps.
    - `split_valid` (list): list of validation set mocaps.
    """

    # compute number of mocaps for each split
    dataset_size = len(mocaps_list)
    size_split_test = math.ceil(test_set_percentage / 100 * dataset_size)
    size_split_valid = math.ceil(valid_set_percentage / 100 * dataset_size)

    if test_set_percentage + valid_set_percentage + train_set_percentage == 100:
        size_split_train = dataset_size - size_split_test - size_split_valid
    else:
        size_split_train = math.ceil(train_set_percentage / 100 * dataset_size)

    if min([size_split_train, size_split_test, size_split_valid]) < n_clusters:
        raise Exception("The split sizes should be greater than the number of classes...")

    # get training set split
    seed_value = SEED
    mocap_labels = check_labels_distribution(mocap_labels, n_clusters)
    split_temp, split_train, labels_temp, labels_train = train_test_split(
        mocaps_list,
        mocap_labels,
        test_size=size_split_train,
        stratify=mocap_labels,
        random_state=seed_value,
    )

    # get test set split
    labels_temp = check_labels_distribution(labels_temp, n_clusters)
    split_temp, split_test, labels_temp, labels_test = train_test_split(
        split_temp,
        labels_temp,
        test_size=size_split_test,
        stratify=labels_temp,
        random_state=seed_value,
    )

    # get validation set split
    labels_temp = check_labels_distribution(labels_temp, n_clusters)
    if len(split_temp) == size_split_valid:
        split_valid = split_temp

    else:
        # if there is not enough data
        if len(set(labels_temp)) <= len(split_temp) - size_split_valid:
            # don't do a stratified split
            split_temp, split_valid, labels_temp, labels_valid = train_test_split(
                split_temp,
                labels_temp,
                test_size=size_split_valid,
                random_state=seed_value,
            )
        else:
            split_temp, split_valid, labels_temp, labels_valid = train_test_split(
                split_temp,
                labels_temp,
                test_size=size_split_valid,
                stratify=labels_temp,
                random_state=seed_value,
            )

    return split_train, split_test, split_valid


def check_labels_distribution(mocap_labels, n_clusters):
    """
    Returns an updated list of mocap labels to ensure there are at least two
    elements for each class.

    ---
    ### Args:
    - `mocap_labels` (list): list of mocap labels.
    - `n_clusters` (int): number of clusters.
    ---
    ### Returns:
    - `mocap_labels` (list): updated list of mocap labels.
    """
    # check there is no class with < 2 elements
    few_mocap_labels = []
    largest_label = -1
    num_elements_largest_label = 0

    for label in range(n_clusters):
        num_elements = np.count_nonzero(mocap_labels == label)
        if num_elements < 2:
            few_mocap_labels.append(label)
        elif num_elements > num_elements_largest_label:
            num_elements_largest_label = num_elements
            largest_label = label

    # re-assign labels with too few elements to the most common label
    for label in few_mocap_labels:
        indices = np.where(mocap_labels == label)
        for index in indices[0]:
            mocap_labels[index] = largest_label

    return mocap_labels


def save_mocap(mocap, mocap_name, save_path):
    """
    Saves a given mocap as an .npz file with the same format as in AMASS.

    ---
    ### Args:
    - `mocap` (MocapData): mocap to save.
    - `mocap_name` (str): mocap id.
    - `save_path` (str): where to save the mocap.
    """
    # create dictionary with AMASS format keys
    data = {}
    data["poses"] = mocap.rots
    data["trans"] = mocap.trans
    data["mocap_framerate"] = TARGET_FRAMERATE
    data["gender"] = mocap.gender

    # save it as .npz file
    save_path = os.path.join(save_path, mocap_name)
    np.savez(save_path, **data)
    return


def save_dataset_split(split_train, split_test, split_valid, save_dir):
    """
    Saves train, test and validation splits and returns the dictionary with the list
    of mocaps for each split.

    ---
    ### Args:
    - `split_train` (list): list of training set mocaps.
    - `split_test` (list): list of test set mocaps.
    - `split_valid` (list): list of validation set mocaps.
    - `save_dir` (str): directory where to save mocap splits.
    ---
    ### Returns:
    - `dataset_dict` (dict): dictionary containing the list of mocaps divided into train, test and validation.
    """
    save_mocap_path = os.path.join(save_dir, "")
    dataset_dict = {"train": {}, "test": {}, "validation": {}}

    # train
    mocap_names_list = []
    os.makedirs(save_mocap_path, exist_ok=True)
    for item in tqdm(split_train, f"Saving training data"):
        mocap = item[0]
        mocap_name = item[1] + ".npz"
        mocap_names_list.append(mocap_name)
        save_mocap(mocap, mocap_name, save_mocap_path)
    dataset_dict["train"]["CLUSTERING"] = mocap_names_list

    # test
    mocap_names_list = []
    os.makedirs(save_mocap_path, exist_ok=True)
    for item in tqdm(split_test, f"Saving test data"):
        mocap = item[0]
        mocap_name = item[1] + ".npz"
        mocap_names_list.append(mocap_name)
        save_mocap(mocap, mocap_name, save_mocap_path)
    dataset_dict["test"]["CLUSTERING"] = mocap_names_list

    # valid
    mocap_names_list = []
    os.makedirs(save_mocap_path, exist_ok=True)
    for item in tqdm(split_valid, f"Saving validation data"):
        mocap = item[0]
        mocap_name = item[1] + ".npz"
        mocap_names_list.append(mocap_name)
        save_mocap(mocap, mocap_name, save_mocap_path)
    dataset_dict["validation"]["CLUSTERING"] = mocap_names_list

    return dataset_dict
