import numpy as np

SRNN_EVALUATION_BACH_SIZE = 8  # as in SRNN
SUBACTIONS = 2

PREFIX = 50  # as in SRNN -> 50 input frames
SUFFIX = 100  # as in SRNN -> 100 output frames


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
    for _ in range(SRNN_EVALUATION_BACH_SIZE / SUBACTIONS):
        idxs.append([rng.randint(16, T1), rng.randint(16, T2)])
    return idxs


def get_h36m_evaluation_frames(
    len_subaction_1: int, len_subaction_2: int, input_len: int, output_len: int
):
    if input_len > PREFIX:
        raise ValueError("input_len must be smaller than or equal to 50")
    if output_len > SUFFIX:
        raise ValueError("output_len must be smaller than or equal to 100")

    start_idxs = get_start_indices_srnn(len_subaction_1, len_subaction_2)

    frame_idxs_0 = []
    frame_idxs_1 = []

    for idx in range(SRNN_EVALUATION_BACH_SIZE / SUBACTIONS):
        t0 = start_idxs[idx][0] + PREFIX  # last known frame of subaction 0
        t1 = start_idxs[idx][1] + PREFIX  # last known frame of subaction 1

        frame_idxs_0.append(range(t0 - input_len, t0 + output_len))
        frame_idxs_1.append(range(t1 - input_len, t1 + output_len))

    return frame_idxs_0, frame_idxs_1
