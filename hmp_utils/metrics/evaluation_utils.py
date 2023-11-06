import pkg_resources
import numpy as np
from typing import Union, Optional
import math
import os

from .. import (
    MetricsProcessor,
    SpectrumMetricsProcessor,
    MetricsPostProcessor,
    ForwardKinematics,
    BodyConstants,
    BodyModelType,
    TableManager,
)

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

EVALUATION_TIMESTAMPS_25FPS = [120, 200, 400, 600, 800, 1000]
EVALUATION_WINDOW_LEN_25FPS = 3
EVALUATION_TIMESTAMPS_60FPS = [
    100,
    200,
    400,
    600,
    800,
    1000,
]
EVALUATION_WINDOW_LEN_60FPS = 6
EVALUATION_WINDOW_LEN_DICT = {
    25: EVALUATION_WINDOW_LEN_25FPS,
    30: EVALUATION_WINDOW_LEN_60FPS / 2,
    60: EVALUATION_WINDOW_LEN_60FPS,
}

EVALUATION_T_25 = {int(t / (1000 / 25) - 1): t for t in EVALUATION_TIMESTAMPS_25FPS}
EVALUATION_T_30 = {int(t / (1000 / 30) - 1): t for t in EVALUATION_TIMESTAMPS_60FPS}
EVALUATION_T_60 = {int(t / (1000 / 60) - 1): t for t in EVALUATION_TIMESTAMPS_60FPS}
EVALUATION_T_DICT = {25: EVALUATION_T_25, 30: EVALUATION_T_30, 60: EVALUATION_T_60}

H36M_PSD = {
    "train": pkg_resources.resource_filename(__name__, "../pds_files/h36m_train_psd.npy"),
    "test": pkg_resources.resource_filename(__name__, "../pds_files/h36m_test_psd.npy"),
    "validation": pkg_resources.resource_filename(__name__, "../pds_files/h36m_validation_psd.npy"),
}

AMASS_25_PSD = {
    "train": pkg_resources.resource_filename(__name__, "../pds_files/amass_25_train_psd.npy"),
    "test": pkg_resources.resource_filename(__name__, "../pds_files/amass_25_test_psd.npy"),
    "validation": pkg_resources.resource_filename(__name__, "../pds_files/amass_25_validation_psd.npy"),
}
AMASS_30_PSD = {
    "train": pkg_resources.resource_filename(__name__, "../pds_files/amass_30_train_psd.npy"),
    "test": pkg_resources.resource_filename(__name__, "../pds_files/amass_30_test_psd.npy"),
    "validation": pkg_resources.resource_filename(__name__, "../pds_files/amass_30_validation_psd.npy"),
}
AMASS_60_PSD = {
    "train": pkg_resources.resource_filename(__name__, "../pds_files/amass_60_train_psd.npy"),
    "test": pkg_resources.resource_filename(__name__, "../pds_files/amass_60_test_psd.npy"),
    "validation": pkg_resources.resource_filename(__name__, "../pds_files/amass_60_validation_psd.npy"),
}
AMASS_PSD_BY_FRAMERATE = {25: AMASS_25_PSD, 30: AMASS_30_PSD, 60: AMASS_60_PSD}


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
    def get_frames(len_subaction_1: int, len_subaction_2: int, input_len: int, output_len: int) -> np.ndarray:
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

        start_idxs = SrnnEvaluationFrames.get_start_indices(len_subaction_1, len_subaction_2)
        frame_idxs = np.zeros(
            (
                SUBACTIONS,
                SRNN_EVALUATION_BACH_SIZE // SUBACTIONS,
                input_len + output_len,
            ),
            dtype=int,
        )

        for idx in range(SRNN_EVALUATION_BACH_SIZE):
            t0 = start_idxs[idx // SUBACTIONS][idx % SUBACTIONS] + PREFIX  # last known frame
            frame_idxs[idx % SUBACTIONS][idx // SUBACTIONS] = np.arange(t0 - input_len, t0 + output_len)

        return frame_idxs

    @staticmethod
    def get_spectrum_frames(
        num_frames_list: Union[list, np.ndarray],
        seq_len: float = SPECTRUM_EVALUATION_SEQ_LEN,
    ) -> list:
        """
        Returns a list of lists (num_mocap, num_seq) of frame indices selected for the spectrum comparison.
        This is an adaptation of the protocol used with the AMASS dataset.

        ---
        ### Args:
        - `num_frames_list` (list or np.ndarray): list of number of frames of the mocaps.
        - `seq_len` (float): length of the sequence in seconds.
        ---
        ### Returns:
        - `frame_idxs_list` (list): list of lists (num_mocap, batch, frame) of frame indices selected for the spectrum comparison.
        """
        framerate = 25

        frame_idxs_list = []
        for num_frames in num_frames_list:
            frame_idxs = []
            for start_idx in AmassEvaluationFrames.get_start_indices(mocap_len=num_frames, seq_len=seq_len * framerate):
                frame_idxs.append(range(start_idx, start_idx + seq_len * framerate))
            frame_idxs_list.append(frame_idxs)

        return frame_idxs_list


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
            frame_idxs.append(rng.randint(first_index + idx * subseq_len, first_index + (idx + 1) * subseq_len))

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
            raise ValueError("input_len must be smaller than or equal to AMASS_INPUT_SEQ_LEN(2s) * framerate")
        if output_len > AMASS_OUTPUT_SEQ_LEN * framerate:
            raise ValueError("output_len must be smaller than or equal to AMASS_OUTPUT_SEQ_LEN(1s) * framerate")

        frame_idxs_list = []
        for num_frames in num_frames_list:
            frame_idxs = []
            for start_idx in AmassEvaluationFrames.get_start_indices(mocap_len=num_frames, seq_len=AMASS_SEQ_LEN * framerate):
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
            raise ValueError("seq_len must be smaller than or equal to AMASS_SEQ_LEN(3s)")

        frame_idxs_list = []
        for num_frames in num_frames_list:
            frame_idxs = []
            for start_idx in AmassEvaluationFrames.get_start_indices(mocap_len=num_frames, seq_len=seq_len * framerate):
                frame_idxs.append(range(start_idx, start_idx + seq_len * framerate))
            frame_idxs_list.append(frame_idxs)

        if sum(len(frame_idxs) for frame_idxs in frame_idxs_list) < MIN_SPECTRUM_METRICS_EVALUATION_BACH_SIZE:
            raise ValueError(f"the number of frames selected for evaluation is less than {MIN_SPECTRUM_METRICS_EVALUATION_BACH_SIZE}")
        return frame_idxs_list


class EvaluationProtocol:
    @staticmethod
    def initialize_metrics_dicts(fk_processor: ForwardKinematics) -> tuple([dict, dict]):
        """
        Initialize the metrics dictionaries.

        ---
        ### Args:
        - `fk_processor` (ForwardKinematics): forward kinematics processor.
        ---
        ### Returns:
        - `metrics_dict` (dict): dictionary of metrics names.
        - `metric_processor_dict` (dict): dictionary of metrics processors.
        """
        standard_metrics_dict = {
            "default_error": "Default Error",
            "euler_err_xyz": "Euler Error XYZ",
            "euler_err_zxy": "Euler Error ZXY",
            "joint_ang_diff_local": "Joint Angle Diff Local",
            "joint_ang_diff_global": "Joint Angle Diff Global",
            "positonal_err": "Positional Error",
            "w_distance": "Wasserstein Distance",
        }
        cumulative_metrics_dict = {
            "cumulative_default_error": "Cumulative Default Error",
            "cumulative_euler_err_xyz": "Cumulative Euler Error XYZ",
            "cumulative_euler_err_zxy": "Cumulative Euler Error ZXY",
            "cumulative_joint_ang_diff_local": "Cumulative Joint Angle Diff Local",
            "cumulative_joint_ang_diff_global": "Cumulative Joint Angle Diff Global",
            "cumulative_positonal_err": "Cumulative Positional Error",
        }
        frame_metrics_dict = {
            "frame_default_error": "Frame Default",
            "frame_euler_err_xyz": "Frame Euler Error XYZ",
            "frame_euler_err_zxy": "Frame Euler Error ZXY",
            "frame_joint_ang_diff_local": "Frame Joint Angle Diff Local",
            "frame_joint_ang_diff_global": "Frame Joint Angle Diff Global",
            "frame_positonal_err": "Frame Positional Error",
        }
        metrics_dict = {
            **standard_metrics_dict,
            **cumulative_metrics_dict,
            **frame_metrics_dict,
        }

        metric_processor_dict = {
            "euler_err_xyz": lambda m_proc: m_proc.euler_error(euler_seq="xyz"),
            "euler_err_zxy": lambda m_proc: m_proc.euler_error(euler_seq="zxy"),
            "joint_ang_diff_local": lambda m_proc: m_proc.joint_angle_difference(),
            "joint_ang_diff_global": lambda m_proc: m_proc.joint_angle_difference(use_global_rotations=True, f_kinematics_processor=fk_processor),
            "positonal_err": lambda m_proc: m_proc.positional_error(fk_processor),
        }
        return metrics_dict, metric_processor_dict

    @staticmethod
    def load_power_spectral_density_dict(
        body_model_type: BodyModelType,
        framerate: int,
    ) -> tuple([dict, dict]):
        """
        Load the power spectral density dictionaries.

        ---
        ### Args:
        - `body_model_type` (BodyModelType): body model type.
        - `framerate` (int): framerate of the mocaps.
        ---
        ### Returns:
        - `power_spectral_density_dict` (dict): dictionary of power spectral density.
        - `removed_joints_dict` (dict): dictionary of removed joints.
        """
        if body_model_type == BodyModelType.H36M:
            power_spectral_density_files = H36M_PSD
        elif body_model_type == BodyModelType.AMASS:
            power_spectral_density_files = AMASS_PSD_BY_FRAMERATE[framerate]
        power_spectral_density_dict = {}
        removed_joints_dict = {}
        for key, file in power_spectral_density_files.items():
            power_spectral_density_dict[key] = np.load(file)
            (
                power_spectral_density_dict[key],
                removed_joints_dict[key],
            ) = SpectrumMetricsProcessor.remove_static_joints(power_spectral_density_dict[key])
        return power_spectral_density_dict, removed_joints_dict

    @staticmethod
    def compute_wasserstein_distance_list(
        predictions: Union[list, np.ndarray],
        framerate: int,
        fk_processor: ForwardKinematics,
        bodyConstants: BodyConstants,
        set_of_psd_to_use: str = "train",
        zero_velocity: bool = False,
    ) -> list:
        """
        Compute the wasserstein distance list.

        ---
        ### Args:
        - `predictions` (list or np.ndarray): list of predictions.
        - `framerate` (int): framerate of the mocaps.
        - `fk_processor` (ForwardKinematics): forward kinematics processor.
        - `bodyConstants` (BodyConstants): body constants.
        - `set_of_psd_to_use` (str): computed power spectral density to use.
        - `zero_velocity` (bool): if True the Zero-Velocity model is used.
        ---
        ### Returns:
        - `error_list` (list): list of wasserstein distance.
        """
        power_spectral_density_dict, removed_joints_dict = EvaluationProtocol.load_power_spectral_density_dict(bodyConstants.BODY_MODEL_TYPE, framerate)

        pos_predictions = fk_processor.compute_fk(predictions, apply_root_rotation=False)  # (batch, seq_len, num_joints, 3)
        error_list = []
        seq_len = pos_predictions.shape[-3]
        window_len = SPECTRUM_EVALUATION_SEQ_LEN * framerate
        for idx in range(seq_len - window_len + 1):
            psd_pred_target = SpectrumMetricsProcessor.power_spectral_density(
                seq_t=pos_predictions[:, idx : window_len + idx],
                framerate=framerate,
                remove_zero_freq=True,
                normalize=True,
                joints_to_compute=bodyConstants.MAJOR_JOINTS,
            )  # (ps_len, num_joints, 3)

            if not zero_velocity:
                (
                    psd_pred_target,
                    removed_joints,
                ) = SpectrumMetricsProcessor.remove_static_joints(psd_pred_target)
                if removed_joints != removed_joints_dict[set_of_psd_to_use]:
                    raise Exception("Removed joints are not the same")
            else:
                psd_pred_target = np.delete(psd_pred_target, removed_joints_dict[set_of_psd_to_use], axis=1)

            error_list.append(SpectrumMetricsProcessor.wasserstein_distance(power_spectral_density_dict[set_of_psd_to_use], psd_pred_target))
        return error_list  # (seq_len - window_len + 1)

    @staticmethod
    def compute_all_metrics(
        predictions: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
        framerate: int,
        target_seq_len: int,
        default_error: float,
        bodyConstants: BodyConstants,
        fk_processor: ForwardKinematics,
        metrics_res_by_action_dict: dict,
        action: str,
        metrics_dict: dict,
        metric_processor_dict: dict,
        set_of_psd_to_use: str = "train",
        zero_velocity=False,
        use_global_rotations=False,
    ) -> None:
        """
        Compute all the metrics and update the `metrics_res_by_action_dict` dictionary.

        ---
        ### Args:
        - `predictions` (list or np.ndarray): list of predictions.
        - `targets` (list or np.ndarray): list of targets.
        - `framerate` (int): framerate of the mocaps.
        - `target_seq_len` (int): number of frames of the target sequence.
        - `default_error` (float): default error.
        - `bodyConstants` (BodyConstants): body constants.
        - `fk_processor` (ForwardKinematics): forward kinematics processor.
        - `metrics_res_by_action_dict` (dict): dictionary of metrics results by action.
        - `action` (str): action name.
        - `metrics_dict` (dict): dictionary of metrics names.
        - `metric_processor_dict` (dict): dictionary of metrics processors.
        - `set_of_psd_to_use` (str): computed power spectral density to use.
        - `zero_velocity` (bool): if True the Zero-Velocity model is used.
        - `use_global_rotations` (bool): if True the global rotations are used.
        """
        metric_proc_res = {}
        metric_proc_res["default_error"] = default_error

        if use_global_rotations:
            joints_to_compute = bodyConstants.MAJOR_JOINTS
        else:
            joints_to_compute = [i for i in bodyConstants.MAJOR_JOINTS if i != 0]

        metrics_proc = MetricsProcessor(
            predictions=predictions,
            targets=targets,
            joints_to_compute=joints_to_compute,
        )

        for key, metric in metric_processor_dict.items():
            metric_proc_res[key] = metric(metrics_proc)

        metric_proc_res["w_distance"] = EvaluationProtocol.compute_wasserstein_distance_list(
            predictions=predictions,
            framerate=framerate,
            fk_processor=fk_processor,
            bodyConstants=bodyConstants,
            set_of_psd_to_use=set_of_psd_to_use,
            zero_velocity=zero_velocity,
        )

        for key in metrics_dict.keys():
            metric_post_proc = MetricsPostProcessor(metric_proc_res[key.replace("cumulative_", "").replace("frame_", "")])
            metrics_value = []
            if "w_distance" == key:
                for frame in range(len(metric_proc_res[key])):
                    metrics_value.append(metric_post_proc.instant(at_frame=frame))
            else:
                if "cumulative_" in key:
                    for frame, _ in EVALUATION_T_DICT[framerate].items():
                        metrics_value.append(metric_post_proc.cumulative(end_frame=frame, normalize=True))
                elif "frame_" in key:
                    for frame in range(target_seq_len * 25):
                        metrics_value.append(metric_post_proc.instant(at_frame=frame))
                else:
                    for frame, _ in EVALUATION_T_DICT[framerate].items():
                        metrics_value.append(
                            metric_post_proc.windowed_mean(
                                at_frame=frame,
                                window_len_frames=EVALUATION_WINDOW_LEN_DICT[framerate],
                            )
                        )
            metrics_res_by_action_dict[action][key] = metrics_value

    @staticmethod
    def save_all_metrics(
        metrics_res_by_action_dict: dict,
        metrics_dict: dict,
        framerate: int,
        csv_path: Optional[str] = None,
        print_table=False,
    ) -> None:
        """
        Save all the metrics in a csv file and/or print them.

        ---
        ### Args:
        - `metrics_res_by_action_dict` (dict): dictionary of metrics results by action.
        - `metrics_dict` (dict): dictionary of metrics names.
        - `framerate` (int): framerate of the mocaps.
        - `csv_path` (str): path of the csv file. If None the table is not saved.
        - `print_table` (bool): if True the table is printed.
        """
        frame_span_ms = 1000 / framerate
        index = [action for action in metrics_res_by_action_dict.keys()]
        if len(index) > 1:
            add_to_index = ["mean"]
        else:
            add_to_index = []
        for metric_key, table_title in metrics_dict.items():
            table = np.array([metrics_res_by_action_dict[action][metric_key] for action in index])
            if len(index) > 1:
                table = np.vstack([table, np.mean(table, axis=0)])
            if "w_distance" == metric_key:
                times_to_print = [SPECTRUM_EVALUATION_SEQ_LEN * 1000 + round(idx * frame_span_ms) for idx in range(table.shape[1])]
            elif "frame_" in metric_key:
                times_to_print = [round((idx + 1) * frame_span_ms) for idx in range(table.shape[1])]
            else:
                times_to_print = [round((idx + 1) * frame_span_ms) for idx in EVALUATION_T_DICT[framerate].keys()]
            header = [f"{ms: >6} ms" for ms in times_to_print]

            t_manager = TableManager(
                data=table,
                table_name=table_title,
                header=header,
                index=index + add_to_index,
            )
            if print_table:
                t_manager.print()
            if csv_path is not None:
                t_manager.to_csv(path=csv_path)
