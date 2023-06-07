import os
import logging
import numpy as np
from scipy import signal
from typing import Union
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Union
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
from hmp_utils import RotConvert
from .. import BodyModelType, SmplConstants, H36MConstants


class DownsampleMethod(Enum):
    RESAMPLE = auto()
    DECIMATE = auto()
    SLERP = auto()
    NONE = auto()


class MocapData:
    def __init__(
        self,
        rots: np.ndarray,
        trans: np.ndarray,
        original_framerate: float,
        framerate: float,
        gender: str,
    ):
        # rots: relative rotations of joints with shape (num_frames, num_joints, 3)
        self.rots = rots
        # trans: translation of root joint (num_frames, 1, 3)
        self.trans = trans
        self.original_framerate = original_framerate
        self.framerate = framerate
        self.gender = gender


def downsample_motion(
    source_array: Union[np.ndarray, list],
    source_framerate: float,
    target_framerate: float,
    method: "DownsampleMethod" = DownsampleMethod.SLERP,
) -> Union[np.ndarray, None]:
    """
    Downsamples the given array to a target framerate using the specified method

    ---
    ### Args:
    - `source_array` (np.ndarray): array containing the sequence to be downsampled. If downsampling
        method is `slerp`, then the rotations must be in axis-angle representation in which case
        the array has shape: (num_frames, ..., 3)
    - `source_framerate` (float): framerate at which the original sequence is sampled
    - `target_framerate` (float): desired framerate of the returned sequence
    - `method` (DownsampleMethod): either `decimate` or `resample` have general validity
        If the method is `slerp`, then the rotations must be in axis-angle representation

    ---
    ### Returns:
    (np.ndarray): the downsampled array
    """

    if not isinstance(source_array, np.ndarray):
        source_array = np.array(source_array)

    if method == DownsampleMethod.DECIMATE:
        try:
            downsampling_factor = int(round(source_framerate / target_framerate))
            original_len = int(np.floor(source_array.shape[0] / downsampling_factor))
            downsampled_array = signal.decimate(
                source_array, downsampling_factor, zero_phase=True, axis=0
            )[:original_len]
            return np.array(downsampled_array)
        except ValueError:
            # "The length of the input vector x must be greater than padlen"
            logging.error("Downsampling decimation failed")
            return None

    elif method == DownsampleMethod.RESAMPLE:
        source_num_frames = source_array.shape[0]
        downsampled_len_frames = int(
            source_num_frames * target_framerate / source_framerate
        )

        downsampled_array = signal.resample_poly(
            source_array, downsampled_len_frames, source_num_frames
        )
        return np.array(downsampled_array)

    elif method == DownsampleMethod.SLERP:  # TODO: check if this is correct
        if source_array.shape[-1] != 3:
            raise Exception(
                "When using slerp downsampling method, the provided source array "
                "should be in axis-angle representation with shape (..., 3). "
                f"Input array shape is {source_array.shape}"
            )

        # compute at which times the sampling should occur
        original_shape = source_array.shape
        original_len_frames = original_shape[0]
        # this is not the actual duration of the sequence, but the last frame sample time
        original_len_sec = original_len_frames / source_framerate

        original_key_times = np.arange(
            start=0, stop=original_len_sec, step=1 / source_framerate
        )
        original_key_times = original_key_times[:original_len_frames]
        target_times = np.arange(
            start=0, stop=original_len_sec, step=1 / target_framerate
        )

        # remove the last frame if it exceeds the time of the last frame in the key times
        target_times = target_times[target_times < original_key_times[-1]]

        # the shape is (num_frames, ..., 3) but we need (..., num_frames, 3)
        source_array = np.swapaxes(source_array, 0, -2)
        # reshape to process single rotation vectors
        source_array = np.reshape(source_array, (-1, original_len_frames, 3))
        # iterate over all sequences
        interpolated_rotations = []
        for i in range(source_array.shape[0]):
            sequence = source_array[i]
            slerp_worker = Slerp(original_key_times, Rotation.from_rotvec(sequence))
            interpolated_sequence = slerp_worker(target_times)
            interpolated_rotations.append(interpolated_sequence.as_rotvec())

        interpolated_rotations = np.array(interpolated_rotations)
        downsampled_len_frames = interpolated_rotations.shape[1]

        # now from shape (N, downsampled_len_frames, 3) to (downsampled_len_frames, N, 3)
        interpolated_rotations = np.swapaxes(interpolated_rotations, 0, 1)
        return_shape = (downsampled_len_frames, *original_shape[1:])
        return np.reshape(interpolated_rotations, return_shape)

    elif method == DownsampleMethod.NONE:
        return source_array

    raise Exception(f"Invalid downsampling method {method}")


class MocapLoader:
    def __init__(
        self,
        body_model_type: BodyModelType = BodyModelType.SMPL,
        keypoint_ids_to_load: Optional[Union[list, np.ndarray]] = None,
        target_framerate: Optional[float] = None,
        downsampling_method: DownsampleMethod = DownsampleMethod.SLERP,
        downsampling_tolerance=1e-3,
    ):
        """
        The body model type is used to determine the dataset (human3.6m or AMASS).

        The loaded trajectory will be downsampled to the target framerate, using slerp by default.
        If no target_framerate is provided, the trajectories will not be downsampled.

        ---
        ### Args:
        - `body_model_type` (BodyModelType): body model type to use for loading the mocap data
        - `keypoint_ids_to_load` (list or np.ndarray): list of indices of the keypoints to load
        - `target_framerate` (float): target framerate of the loaded mocap data
        - `downsampling_method` (DownsampleMethod): downsampling method to use
        - `downsampling_tolerance` (float): tolerance for downsampling
        """
        self.body_model_type = body_model_type

        if not isinstance(keypoint_ids_to_load, np.ndarray):
            keypoint_ids_to_load = np.array(keypoint_ids_to_load)
        self.keypoint_ids_to_load = keypoint_ids_to_load

        if target_framerate == None:
            self.target_framerate = target_framerate
        else:
            if target_framerate > 0:
                self.target_framerate = target_framerate
            else:
                raise Exception(
                    f"Provide a valid downsampling framerate (was given {target_framerate})"
                )

        self.downsampling_method = downsampling_method
        self.downsampling_tolerance = downsampling_tolerance

    def load_mocap(
        self,
        file_path,
    ) -> Union[MocapData, None]:
        """
        Assuming trajectories in axis-angle representation with 3 values per joint,
        flattened with shape like (num_frames, num_keypoints*3).

        ---
        ### Args:
        - `filepath` (str): filepath to the mocap file

        ---
        ### Returns:
        - The downsampled mocap trajectory as the MocapData object.
        - None if the target framerate is higher than the mocap framerate

        ---
        ### Raises:
        Exception if the mocap file couldn't be loaded
        """

        if self.body_model_type == BodyModelType.SMPL:
            mocap = self.__load_smpl_mocap(file_path)
        elif self.body_model_type == BodyModelType.H36M:
            mocap = self.__load_h36m_mocap(file_path)

        return mocap

    @staticmethod
    def __can_decimate(source_framerate, target_framerate, tolerance=1e-3):
        """
        If the source framerate is a multiple of the target downsampled framerate
        within the specified tolerance, the timeseries can be decimated (which needs an integer downsampling factor).
        Otherwise we would need to resample the signal.
        """
        ratio = source_framerate / target_framerate
        return abs(ratio - round(ratio)) < tolerance

    def __load_h36m_mocap(self, file_path):
        data = np.loadtxt(file_path, delimiter=",")

        if data is None:
            raise Exception(f"Could not load mocap file {file_path}")

        if self.target_framerate is None:
            target_framerate = float(50)
        else:
            target_framerate = self.target_framerate

        rot_trajectory = np.array(data[:, 3:])
        trans_trajectory = np.array(data[:, :3])

        original_framerate = float(50)
        original_num_frames = rot_trajectory.shape[0]

        # check if in the downsampled traj there will be at least one frame
        if int(round(original_num_frames * target_framerate / original_framerate) <= 1):
            return None

        framerate_difference = abs(original_framerate - target_framerate)
        if (
            original_framerate < target_framerate
            and framerate_difference > self.downsampling_tolerance
        ):
            raise Exception(
                f"Mocap sequence could not be loaded because target framerate {target_framerate} "
                f"is higher than the mocap framerate {original_framerate} \n"
                f"Mocap file: {os.path.basename(file_path)}"
            )

        # reshape to (num_frames, num_joints, 3) or (num_frames, 1, 3) for translation
        final_shape = (original_num_frames, -1, 3)
        trans_trajectory = np.reshape(trans_trajectory, final_shape)
        rot_trajectory = np.reshape(rot_trajectory, final_shape)

        # select the components of the angle-axis representation to extract from the 'pose' array
        rot_trajectory = rot_trajectory[:, self.keypoint_ids_to_load, :]

        # don't downsample
        if framerate_difference <= self.downsampling_tolerance:
            return MocapData(
                rot_trajectory,
                trans_trajectory,
                original_framerate,
                original_framerate,
            )

        # downsample to the target framerate
        if (
            self.downsampling_method == DownsampleMethod.DECIMATE
            and not self.__can_decimate(original_framerate, target_framerate)
        ):
            downsample_method = DownsampleMethod.SLERP
        else:
            downsample_method = self.downsampling_method

        rot_trajectory = downsample_motion(
            rot_trajectory, original_framerate, target_framerate, downsample_method
        )
        trans_trajectory = downsample_motion(
            trans_trajectory, original_framerate, target_framerate, downsample_method
        )

        if rot_trajectory is None or trans_trajectory is None:
            return None

        downsampled_framerate = (
            original_framerate * len(trans_trajectory) / original_num_frames
        )

        return MocapData(
            rot_trajectory,
            trans_trajectory,
            original_framerate,
            downsampled_framerate,
        )

    def __load_smpl_mocap(self, file_path):
        data = np.load(file_path)

        if data is None:
            raise Exception(f"Could not load mocap file {file_path}")

        smpl_keys = ["poses", "trans", "mocap_framerate", "gender"]
        if any([key not in data.files for key in smpl_keys]):
            raise Exception(
                f"KeyError: {smpl_keys} smpl key was not found in mocap {file_path}"
            )

        if self.target_framerate is None:
            target_framerate = float(data["mocap_framerate"])
        else:
            target_framerate = self.target_framerate

        rot_trajectory = np.array(data["poses"])
        trans_trajectory = np.array(data["trans"])

        original_framerate = float(data["mocap_framerate"])
        original_num_frames = rot_trajectory.shape[0]

        gender = str(data["gender"])

        # check if in the downsampled traj there will be at least one frame
        if int(round(original_num_frames * target_framerate / original_framerate) <= 1):
            return None

        framerate_difference = abs(original_framerate - target_framerate)
        if (
            original_framerate < target_framerate
            and framerate_difference > self.downsampling_tolerance
        ):
            raise Exception(
                f"Mocap sequence could not be loaded because target framerate {target_framerate} "
                f"is higher than the mocap framerate {original_framerate} \n"
                f"Mocap file: {os.path.basename(file_path)}"
            )

        # reshape to (num_frames, num_joints, 3) or (num_frames, 1, 3) for translation
        final_shape = (original_num_frames, -1, 3)
        trans_trajectory = np.reshape(trans_trajectory, final_shape)
        rot_trajectory = np.reshape(rot_trajectory, final_shape)

        # select the components of the angle-axis representation to extract from the 'pose' array
        rot_trajectory = rot_trajectory[:, self.keypoint_ids_to_load, :]

        # don't downsample
        if framerate_difference <= self.downsampling_tolerance:
            return MocapData(
                rot_trajectory,
                trans_trajectory,
                original_framerate,
                original_framerate,
                gender,
            )

        # downsample to the target framerate
        if (
            self.downsampling_method == DownsampleMethod.DECIMATE
            and not self.__can_decimate(original_framerate, target_framerate)
        ):
            downsample_method = DownsampleMethod.SLERP
        else:
            downsample_method = self.downsampling_method

        rot_trajectory = downsample_motion(
            rot_trajectory, original_framerate, target_framerate, downsample_method
        )
        trans_trajectory = downsample_motion(
            trans_trajectory, original_framerate, target_framerate, downsample_method
        )

        if rot_trajectory is None or trans_trajectory is None:
            return None

        downsampled_framerate = (
            original_framerate * len(trans_trajectory) / original_num_frames
        )

        return MocapData(
            rot_trajectory,
            trans_trajectory,
            original_framerate,
            downsampled_framerate,
            gender,
        )


class Amass2HumanConvert:
    def __init__(
        self,
    ):
        # human-smpl keypoints correspondence
        self.human_to_amass_kp = {
            0: 0,
            1: 2,
            2: 5,
            3: 8,
            4: 11,
            6: 1,
            7: 4,
            8: 7,
            9: 10,
            11: 3,
            12: 6,
            13: 9,
            14: 12,
            15: 15,
            16: 13,
            17: 16,
            18: 18,
            19: 20,
            22: 22,
            24: 14,
            25: 17,
            26: 19,
            27: 21,
            30: 23,
        }

        # no rotations for some joints
        self.human_to_skip = [4, 5, 9, 10, 15, 20, 21, 22, 23, 28, 29, 30, 31]

        # compute rotation matricex
        self.aligning_rotmats = self.compute_alining_rotmats()

    def compute_alining_rotmats(self):
        # amass relative offsets
        amass_offsets = np.reshape(np.array(SmplConstants.OFFSETS), (-1, 3))
        amass_relative_offsets = amass_offsets
        amass_relative_offsets[1:] = (
            amass_offsets[1:] - amass_offsets[SmplConstants.PARENTS[1:]]
        )

        # human relative offsets
        human_relative_offsets = np.reshape(np.array(H36MConstants.OFFSETS), (-1, 3))
        human_relative_offsets_ = np.reshape(np.array(H36MConstants.OFFSETS), (-1, 3))

        rotations_chain = [np.eye(3) for _ in range(len(human_relative_offsets))]
        aligning_rotmats = [np.eye(3) for _ in range(len(human_relative_offsets))]

        # compute aligning rotation matrices
        for kp in range(1, len(human_relative_offsets)):
            if kp in self.human_to_skip:
                pass
            else:
                kp_parent = H36MConstants.PARENTS[kp]
                # in most of the cases the child kp is kp+1 (does not work for hands)
                kp_child = kp + 1
                if kp == 19:
                    kp_child = 22
                elif kp == 27:
                    kp_child = 30

                # apply parent rotations also to the current link
                human_relative_offsets_[kp_child] = np.matmul(
                    rotations_chain[kp_parent], human_relative_offsets[kp_child]
                )

                # compute aligning rotmat
                amass_link = amass_relative_offsets[self.human_to_amass_kp[kp_child]]
                human_link = human_relative_offsets_[kp_child]
                # if human link 12, align the spine with amass link 0-6
                if kp == 12:
                    amass_link = amass_offsets[6] - amass_offsets[0]

                aligning_rotmats[kp] = self.rotation_matrix_from_vectors(
                    human_link, amass_link
                )

                # update rotations chain
                rotations_chain[kp] = np.matmul(
                    aligning_rotmats[kp], rotations_chain[kp_parent]
                )

        return aligning_rotmats

    def amass_to_h36m_frame(self, amass_rots, amass_trans=None):
        if amass_trans is None:
            amass_trans = np.zeros((1, 3))

        amass_motion_rotmat = RotConvert.aa2rotmat(amass_rots)
        h36m_frame = np.zeros((32, 3))

        for kp in range(1, len(h36m_frame)):
            if kp in self.human_to_skip:
                pass
            else:
                h36m_frame[kp] = -RotConvert.rotmat2aa(
                    np.matmul(
                        amass_motion_rotmat[self.human_to_amass_kp[kp]],
                        self.aligning_rotmats[kp],
                    )
                )

                # need to swap axes
                if kp in [16, 17, 18, 19]:
                    temp = -RotConvert.rotmat2aa(
                        np.matmul(
                            amass_motion_rotmat[self.human_to_amass_kp[kp]],
                            self.aligning_rotmats[kp],
                        )
                    )
                    h36m_frame[kp][2] = temp[2]
                    h36m_frame[kp][0] = -temp[1]
                    h36m_frame[kp][1] = temp[0]

                if kp in [24, 25, 26, 27]:
                    temp = -RotConvert.rotmat2aa(
                        np.matmul(
                            amass_motion_rotmat[self.human_to_amass_kp[kp]],
                            self.aligning_rotmats[kp],
                        )
                    )
                    h36m_frame[kp][2] = temp[2]
                    h36m_frame[kp][0] = temp[1]
                    h36m_frame[kp][1] = -temp[0]
        h36m_frame = np.concatenate([amass_trans, h36m_frame])
        return h36m_frame

    def amass_to_h36m_mocap(self, amass_mocap_rots, amass_mocap_trans=None):
        # return np.apply_along_axis(self.amass_to_h36m_frame, axis=0, arr=amass_mocap)
        num_frames = amass_mocap_rots.shape[0]

        if amass_mocap_trans is None:
            amass_mocap_trans = np.zeros((num_frames, 1, 3))

        human_motion_full = np.zeros((num_frames, 33, 3))
        for frame in range(num_frames):
            human_motion_full[frame] = self.amass_to_h36m_frame(
                amass_rots=amass_mocap_rots[frame], amass_trans=amass_mocap_trans[frame]
            )

        return human_motion_full

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
            vec2 / np.linalg.norm(vec2)
        ).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

        return rotation_matrix
