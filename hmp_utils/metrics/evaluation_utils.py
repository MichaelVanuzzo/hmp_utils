import numpy as np
from typing import Union
import math

SRNN_EVALUATION_BACH_SIZE = 8  # as in SRNN
SUBACTIONS = 2

PREFIX = 50  # as in SRNN -> 50 input frames
SUFFIX = 100  # as in SRNN -> 100 output frames

AMASS_EVALUATION_SEED = 130297
AMASS_INPUT_SEQ_LEN = 2  # in seconds
AMASS_OUTPUT_SEQ_LEN = 1  # in seconds
AMASS_SEQ_LEN = AMASS_INPUT_SEQ_LEN + AMASS_OUTPUT_SEQ_LEN  # in seconds
AMASS_MAX_SEQUENCES_BY_MOCAP = 3

MIN_DISTRIBUTION_METRICS_EVALUATION_BACH_SIZE = 20000

AMASS_DATASETS_UTILIZED = [
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "DFaust",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HDM05",
    "MoSh",
    "SFU",
    "SSM",
    "TCDHands",
    "TotalCapture",
    "Transitions",
]


def get_start_indices_srnn(len_subaction_1: int, len_subaction_2: int) -> list:
    """
    Find frames indices for standard evaluation with human3.6m dataset.
    Adapted from https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    to find the same indices as in SRNN.
    """

    # Same seed as https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = len_subaction_1 - PREFIX - SUFFIX
    T2 = len_subaction_2 - PREFIX - SUFFIX

    idxs = []
    for _ in range(SRNN_EVALUATION_BACH_SIZE // SUBACTIONS):
        idxs.append([rng.randint(16, T1), rng.randint(16, T2)])
    return idxs


def get_h36m_evaluation_frames(
    len_subaction_1: int, len_subaction_2: int, input_len: int, output_len: int
) -> np.ndarray:
    """
    Returns a numpy array (subaction, bach, frame) with the frames selected for evaluation as in srnn.
    ---
    ### Args:
    - `len_subaction_1` (int): number of frames of subaction number 1.
    - `len_subaction_2` (int): number of frames of subaction number 2.
    - `input_len` (int): number of input frames.
    - `output_len` (int): number of output frames.
    """

    if input_len > PREFIX:
        raise ValueError("input_len must be smaller than or equal to 50")
    if output_len > SUFFIX:
        raise ValueError("output_len must be smaller than or equal to 100")

    start_idxs = get_start_indices_srnn(len_subaction_1, len_subaction_2)
    frame_idxs = np.zeros(
        (SUBACTIONS, SRNN_EVALUATION_BACH_SIZE // SUBACTIONS, input_len + output_len),
        dtype=int,
    )

    for idx in range(SRNN_EVALUATION_BACH_SIZE):
        t0 = (
            start_idxs[idx // SUBACTIONS][idx % SUBACTIONS] + PREFIX
        )  # last known frame
        frame_idxs[idx % SUBACTIONS][idx // SUBACTIONS] = np.arange(
            t0 - input_len, t0 + output_len
        )

    return frame_idxs


def get_start_indices_amass(mocap_len: int, framerate: int) -> list:
    """
    Find start frames indices for standard evaluation with AMASS dataset.
    ---
    ### Args:
    - `mocap_len` (int): number of frames of the mocap.
    - `framerate` (int): framerate of the mocap.
    """
    rng = np.random.RandomState(AMASS_EVALUATION_SEED)

    max_index = mocap_len - AMASS_SEQ_LEN
    if max_index >= AMASS_MAX_SEQUENCES_BY_MOCAP * AMASS_SEQ_LEN:
        min_index = math.ceil(AMASS_SEQ_LEN / 2)
        max_index -= math.floor(AMASS_SEQ_LEN / 2)
    else:
        min_index = math.ceil((mocap_len % AMASS_SEQ_LEN) / 2)

    frame_idxs = []
    subsections = min(
        (max_index - min_index + 1) // AMASS_SEQ_LEN, AMASS_MAX_SEQUENCES_BY_MOCAP
    )
    for idx in range(subsections):
        frame_idxs.append(
            rng.randint(
                min_index + idx * AMASS_SEQ_LEN, min_index + (idx + 1) * AMASS_SEQ_LEN
            )
        )

    return frame_idxs


def get_amass_evaluation_frames(
    num_frames_list: Union[list, np.ndarray],
    input_len: int,
    output_len: int,
    framerate: int,
) -> np.ndarray:
    """
    Returns a numpy array (num_mocap, num_seq) with the frame indexes selected for evaluation.
    ---
    ### Args:
    - `num_frames_list` (list or np.ndarray): list of number of frames of the mocaps.
    - `input_len` (int): number of input frames.
    - `output_len` (int): number of output frames.
    - `framerate` (int): framerate of the mocaps.
    """
    if input_len > AMASS_INPUT_SEQ_LEN * framerate:
        raise ValueError(
            "input_len must be smaller than or equal to AMASS_INPUT_SEQ_LEN(2s) * framerate"
        )
    if output_len > AMASS_OUTPUT_SEQ_LEN * framerate:
        raise ValueError(
            "output_len must be smaller than or equal to AMASS_OUTPUT_SEQ_LEN(1s) * framerate"
        )

    frame_idxs_list = np.array([])
    for num_frames in num_frames_list:
        start_idxs = get_start_indices_amass(mocap_len=num_frames, framerate=framerate)
        frame_idxs = np.array([])
        for start_idx in start_idxs:
            t0 = start_idx + AMASS_INPUT_SEQ_LEN * framerate
            frame_idxs = np.append(
                frame_idxs, np.arange(t0 - input_len, t0 + output_len)
            )
        frame_idxs_list = np.append(frame_idxs_list, frame_idxs)

    return frame_idxs_list
