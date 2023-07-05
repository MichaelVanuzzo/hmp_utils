import numpy as np

SRNN_EVALUATION_BACH_SIZE = 8  # as in SRNN
SUBACTIONS = 2

PREFIX = 50  # as in SRNN -> 50 input frames
SUFFIX = 100  # as in SRNN -> 100 output frames

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


def get_start_indices_srnn(len_subaction_1: int, len_subaction_2: int):
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


def get_distribution_metrics_evaluation_frames(
    dataset_len: int, seq_len: int
) -> np.ndarray:  # TODO
    """
    Returns a numpy array (bach, frame) with the frames selected for distribution metrics evaluation.

    ---
    ### Args:
    - `dataset_len` (int): number of sequences of the dataset.
    - `seq_len` (int): number of frames of the sequence.
    """

    # SEED = 1234567890
    # rng = np.random.RandomState(SEED)

    # idx_sequences = rng.sample(
    #     0, dataset_len, DISTRIBUTION_METRICS_EVALUATION_BACH_SIZE
    # )

    # rng.randint(16, T1)

    # T1 = len_subaction_1 - PREFIX - SUFFIX
    # T2 = len_subaction_2 - PREFIX - SUFFIX

    # idxs = []
    # for _ in range(SRNN_EVALUATION_BACH_SIZE // SUBACTIONS):
    #     idxs.append([rng.randint(16, T1), rng.randint(16, T2)])
    # return idxs
    # return frame_idxs
