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

SPECTRUM_EVALUATION_SEQ_LEN = 1  # in seconds
MIN_SPECTRUM_METRICS_EVALUATION_BACH_SIZE = 20000  # as in STT


class SrnnEvaluationFrames:
    @staticmethod
    def get_start_indices(len_subaction_1: int, len_subaction_2: int) -> list:
        """
        Find frames indices for standard evaluation with human3.6m dataset.
        Adapted from https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        to find the same indices as in SRNN.

        ---
        ### Args:
        - `len_subaction_1` (int): number of frames of subaction number 1.
        - `len_subaction_2` (int): number of frames of subaction number 2.
        ---
        ### Returns:
        - `idxs` (list): list of start indices.
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

    @staticmethod
    def get_frames(
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
        ---
        ### Returns:
        - `frame_idxs` (np.ndarray): array of shape (subaction, bach, frame) with the frames selected for evaluation.
        """

        if input_len > PREFIX:
            raise ValueError("input_len must be smaller than or equal to 50")
        if output_len > SUFFIX:
            raise ValueError("output_len must be smaller than or equal to 100")

        start_idxs = SrnnEvaluationFrames.get_start_indices(
            len_subaction_1, len_subaction_2
        )
        frame_idxs = np.zeros(
            (
                SUBACTIONS,
                SRNN_EVALUATION_BACH_SIZE // SUBACTIONS,
                input_len + output_len,
            ),
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


class AmassEvaluationFrames:
    @staticmethod
    def get_start_indices(mocap_len: int, seq_len: int) -> list:
        """
        Return start frames indices for standard evaluation with AMASS dataset.

        ---
        ### Args:
        - `mocap_len` (int): number of frames of the mocap.
        - `seq_len` (int): length of the sequence in frames.
        ---
        ### Returns:
        - `frame_idxs` (list): list of start indices.
        """
        rng = np.random.RandomState(AMASS_EVALUATION_SEED)

        subsections = min(mocap_len // seq_len, AMASS_MAX_SEQUENCES_BY_MOCAP)

        last_index = mocap_len - seq_len + 1  # first not usable index in the mocap
        if last_index > AMASS_MAX_SEQUENCES_BY_MOCAP * seq_len:
            first_index = math.ceil(seq_len / 2)
            last_index -= math.floor(seq_len / 2)
        elif mocap_len % seq_len > math.floor(seq_len / 2):
            first_index = math.floor(seq_len / 2)
        elif subsections > 1:
            first_index = math.ceil(seq_len / 2)
            subsections -= 1
        else:
            first_index = math.ceil((mocap_len % seq_len) / 2)

        if subsections > 0:
            subseq_len = (last_index - first_index) // subsections

        frame_idxs = []
        for idx in range(subsections):
            frame_idxs.append(
                rng.randint(
                    first_index + idx * subseq_len, first_index + (idx + 1) * subseq_len
                )
            )

        return frame_idxs

    @staticmethod
    def get_frames(
        num_frames_list: Union[list, np.ndarray],
        input_len: int,
        output_len: int,
        framerate: int,
    ) -> list:
        """
        Returns a list of lists (num_mocap, num_seq) of frame indices selected for evaluation.

        ---
        ### Args:
        - `num_frames_list` (list or np.ndarray): list of number of frames of the mocaps.
        - `input_len` (int): number of input frames.
        - `output_len` (int): number of output frames.
        - `framerate` (int): framerate of the mocaps.
        ---
        ### Returns:
        - `frame_idxs_list` (list): list of lists (num_mocap, batch, frame) of frame indices selected for evaluation.
        """
        if input_len > AMASS_INPUT_SEQ_LEN * framerate:
            raise ValueError(
                "input_len must be smaller than or equal to AMASS_INPUT_SEQ_LEN(2s) * framerate"
            )
        if output_len > AMASS_OUTPUT_SEQ_LEN * framerate:
            raise ValueError(
                "output_len must be smaller than or equal to AMASS_OUTPUT_SEQ_LEN(1s) * framerate"
            )

        frame_idxs_list = []
        for num_frames in num_frames_list:
            frame_idxs = []
            for start_idx in AmassEvaluationFrames.get_start_indices(
                mocap_len=num_frames, seq_len=AMASS_SEQ_LEN * framerate
            ):
                t0 = start_idx + AMASS_INPUT_SEQ_LEN * framerate
                frame_idxs.append(range(t0 - input_len, t0 + output_len))
            frame_idxs_list.append(frame_idxs)

        return frame_idxs_list

    @staticmethod
    def get_spectrum_frames(
        num_frames_list: Union[list, np.ndarray],
        framerate: int,
        seq_len: float = SPECTRUM_EVALUATION_SEQ_LEN,
    ) -> list:
        """
        Returns a list of lists (num_mocap, num_seq) of frame indices selected for the spectrum comparison.

        ---
        ### Args:
        - `num_frames_list` (list or np.ndarray): list of number of frames of the mocaps.
        - `framerate` (int): framerate of the mocaps.
        - `seq_len` (float): length of the sequence in seconds.
        ---
        ### Returns:
        - `frame_idxs_list` (list): list of lists (num_mocap, batch, frame) of frame indices selected for the spectrum comparison.
        """
        if seq_len > AMASS_SEQ_LEN:
            raise ValueError(
                "seq_len must be smaller than or equal to AMASS_SEQ_LEN(3s)"
            )

        frame_idxs_list = []
        for num_frames in num_frames_list:
            frame_idxs = []
            for start_idx in AmassEvaluationFrames.get_start_indices(
                mocap_len=num_frames, seq_len=seq_len * framerate
            ):
                frame_idxs.append(range(start_idx, start_idx + seq_len * framerate))
            frame_idxs_list.append(frame_idxs)

        if (
            sum(len(frame_idxs) for frame_idxs in frame_idxs_list)
            < MIN_SPECTRUM_METRICS_EVALUATION_BACH_SIZE
        ):
            raise ValueError(
                f"the number of frames selected for evaluation is less than {MIN_SPECTRUM_METRICS_EVALUATION_BACH_SIZE}"
            )
        return frame_idxs_list
