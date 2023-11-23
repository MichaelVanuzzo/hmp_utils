import os
import logging
import numpy as np
import math
from scipy import signal
from enum import Enum, auto
from typing import Optional, Union
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation
from .. import RotConverter
from .. import BodyModelType, SmplConstants, H36MConstants


class DownsampleMethod(Enum):
    RESAMPLE = auto()
    DECIMATE = auto()
    SLERP = auto()  # not applicable to translation and don't have anti-aliasing
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
        """
        Mocap data container.

        ---
        ### Args:
        - `rots` (np.ndarray): rotations of joints with shape (num_frames, num_joints, 3)
        - `trans` (np.ndarray): translation of root joint (num_frames, 1, 3)
        - `original_framerate` (float): original framerate of the mocap
        - `framerate` (float): framerate of the mocap
        - `gender` (str): gender of the mocap
        """
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
    method: DownsampleMethod = DownsampleMethod.RESAMPLE,
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
    - `downsampled_array` (np.ndarray): downsampled array with shape (num_frames, ..., 3)
    """

    if not isinstance(source_array, np.ndarray):
        source_array = np.array(source_array)

    if method == DownsampleMethod.DECIMATE:
        try:
            downsampling_factor = int(round(source_framerate / target_framerate))
            original_len = int(np.floor(source_array.shape[0] / downsampling_factor))
            downsampled_array = signal.decimate(source_array, downsampling_factor, zero_phase=True, axis=0)
            return np.array(downsampled_array)
        except ValueError:
            # "The length of the input vector x must be greater than padlen"
            logging.error("Downsampling decimation failed")
            return None

    elif method == DownsampleMethod.RESAMPLE:
        downsampling_factor = source_framerate / target_framerate
        downsampled_array = signal.resample_poly(source_array, up=target_framerate, down=source_framerate)
        return np.array(downsampled_array)

    elif method == DownsampleMethod.SLERP:
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
        original_len_sec = (original_len_frames - 1) / source_framerate
        original_key_times = np.arange(original_len_frames) / source_framerate

        target_key_times = np.arange(start=0, stop=original_len_sec + 1e-15, step=1 / target_framerate)

        # the shape is (num_frames, ..., 3) but we need (..., num_frames, 3)
        source_array = np.swapaxes(source_array, 0, -2)
        # reshape to process single rotation vectors
        source_array = np.reshape(source_array, (-1, original_len_frames, 3))
        # iterate over all sequences
        interpolated_rotations = []
        for i in range(source_array.shape[0]):
            sequence = source_array[i]
            slerp_worker = Slerp(original_key_times, Rotation.from_rotvec(sequence))
            interpolated_sequence = slerp_worker(target_key_times)
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
        body_model_type: BodyModelType,
        keypoint_ids_to_load: Union[list, np.ndarray],
        target_framerate: Optional[float] = None,
        downsampling_method: DownsampleMethod = DownsampleMethod.RESAMPLE,
        downsampling_tolerance=1e-3,
    ) -> None:
        """
        The body model type is used to determine the dataset (human3.6m or AMASS).

        The loaded trajectory will be downsampled to the target framerate, using slerp by default.
        If no target_framerate is provided, the trajectories will not be downsampled.

        ---
        ### Args:
        - `body_model_type` (BodyModelType): body model type to use for loading the mocap data
        - `keypoint_ids_to_load` (list, ndarray): list of indices of the keypoints to load
        - `target_framerate` (float)(optional): target framerate of the loaded mocap data
        - `downsampling_method` (DownsampleMethod): downsampling method to use
        - `downsampling_tolerance` (float): tolerance for downsampling
        """
        self.body_model_type = body_model_type
        self.keypoint_ids_to_load = keypoint_ids_to_load

        if target_framerate == None:
            self.target_framerate = target_framerate
        else:
            if target_framerate > 0:
                self.target_framerate = target_framerate
            else:
                raise Exception(f"Provide a valid downsampling framerate (was given {target_framerate})")

        self.downsampling_method = downsampling_method
        self.downsampling_tolerance = downsampling_tolerance

    def load_mocap(
        self,
        file_path,
    ) -> Union[MocapData, None]:
        """
        Assuming trajectories in axis-angle representation with 3 values per joint,
        flattened with shape (num_frames, num_keypoints*3).

        ---
        ### Args:
        - `filepath` (str): filepath to the mocap file
        ---
        ### Returns:
        - The downsampled mocap trajectory as the MocapData object.
        - None if the target framerate is higher than the mocap framerate
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
        """
        Load a human3.6m mocap file.

        ---
        ### Args:
        - `file_path` (str): path to the mocap file to load
        ---
        ### Returns:
        - The downsampled mocap trajectory as the MocapData object.
        - None if the target framerate is higher than the mocap framerate
        """
        data = np.loadtxt(file_path, delimiter=",")

        if data is None:
            raise Exception(f"Could not load mocap file {file_path}")

        if self.target_framerate is None:
            target_framerate = 50.0
        else:
            target_framerate = self.target_framerate

        rot_trajectory = np.array(data[:, 3:])
        trans_trajectory = np.array(data[:, :3])

        original_framerate = 50.0
        original_num_frames = rot_trajectory.shape[0]

        # check if in the downsampled traj there will be at least one frame
        if math.floor(original_num_frames * target_framerate / original_framerate) < 1:
            return None

        framerate_difference = abs(original_framerate - target_framerate)
        if original_framerate < target_framerate and framerate_difference > self.downsampling_tolerance:
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
                gender="neutral",
            )

        # downsample to the target framerate
        if self.downsampling_method == DownsampleMethod.DECIMATE and not self.__can_decimate(original_framerate, target_framerate):
            downsample_method = DownsampleMethod.RESAMPLE
        else:
            downsample_method = self.downsampling_method

        rot_trajectory = downsample_motion(rot_trajectory, original_framerate, target_framerate, downsample_method)
        trans_trajectory = downsample_motion(trans_trajectory, original_framerate, target_framerate, downsample_method)

        if rot_trajectory is None or trans_trajectory is None:
            return None

        if len(trans_trajectory) == int(np.ceil(original_num_frames * target_framerate / original_framerate)):
            downsampled_framerate = target_framerate
        else:
            raise Exception("Downsampled mocap has a different length than expected")

        return MocapData(
            rot_trajectory,
            trans_trajectory,
            original_framerate,
            downsampled_framerate,
            gender="neutral",
        )

    def __load_smpl_mocap(self, file_path):
        """
        Load a smpl mocap file.

        ---
        ### Args:
        - `file_path` (str): path to the mocap file to load
        ---
        ### Returns:
        - The downsampled mocap trajectory as the MocapData object.
        - None if the target framerate is higher than the mocap framerate
        """
        data = np.load(file_path)

        if data is None:
            raise Exception(f"Could not load mocap file {file_path}")

        smpl_keys = ["poses", "trans", "gender"]
        smpl_framerate_keys = ["mocap_frame_rate", "mocap_framerate"]
        if any(key not in data.files for key in smpl_keys) or all(key not in data.files for key in smpl_framerate_keys):
            raise Exception(f"KeyError: {smpl_keys} smpl key was not found in mocap {file_path}")

        if self.target_framerate is None:
            if "mocap_frame_rate" in data.files:
                target_framerate = float(data["mocap_frame_rate"])
            else:
                target_framerate = float(data["mocap_framerate"])
        else:
            target_framerate = self.target_framerate

        rot_trajectory = np.array(data["poses"])
        trans_trajectory = np.array(data["trans"])

        if "mocap_frame_rate" in data.files:
            original_framerate = float(data["mocap_frame_rate"])
        else:
            original_framerate = float(data["mocap_framerate"])
        original_num_frames = rot_trajectory.shape[0]

        gender = str(data["gender"])

        # check if in the downsampled traj there will be at least one frame
        if math.floor(original_num_frames * target_framerate / original_framerate) < 1:
            return None

        framerate_difference = abs(original_framerate - target_framerate)
        if original_framerate < target_framerate and framerate_difference > self.downsampling_tolerance:
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
        if self.downsampling_method == DownsampleMethod.DECIMATE and not self.__can_decimate(original_framerate, target_framerate):
            downsample_method = DownsampleMethod.RESAMPLE
        else:
            downsample_method = self.downsampling_method

        rot_trajectory = downsample_motion(rot_trajectory, original_framerate, target_framerate, downsample_method)
        trans_trajectory = downsample_motion(trans_trajectory, original_framerate, target_framerate, downsample_method)

        if rot_trajectory is None or trans_trajectory is None:
            return None

        if len(trans_trajectory) == int(np.ceil(original_num_frames * target_framerate / original_framerate)):
            downsampled_framerate = target_framerate
        else:
            raise Exception("Downsampled mocap has a different length than expected")

        return MocapData(
            rot_trajectory,
            trans_trajectory,
            original_framerate,
            downsampled_framerate,
            gender,
        )


class Amass2HumanConverter:
    """
    Class to convert an amass mocap to the human3.6m format.

    """

    def __init__(
        self,
    ):
        # human-smpl keypoints correspondence
        self.human2amass_keypoints = {
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
        self.human_keypoint_to_skip = [4, 5, 9, 10, 15, 20, 21, 22, 23, 28, 29, 30, 31]

        # compute rotation matrices
        self.aligning_rotmats = self.__compute_alining_rotmats()

    def __compute_alining_rotmats(self):
        """
        Compute the rotation matrices to align the human3.6m skeleton with the amass skeleton.
        """

        # amass absolute offsets
        amass_absolute_offsets = np.reshape(np.array(SmplConstants.OFFSETS), (-1, 3))

        # amass relative offsets
        amass_relative_offsets = np.reshape(np.array(SmplConstants.OFFSETS), (-1, 3))
        amass_relative_offsets[1:] = amass_absolute_offsets[1:] - amass_absolute_offsets[SmplConstants.PARENTS[1:]]

        # human relative offsets
        H36MRelativeOffsets = np.reshape(np.array(H36MConstants.OFFSETS), (-1, 3))
        H36MRelativeOffsets.flags.writeable = False
        human_relative_offsets = np.reshape(np.array(H36MConstants.OFFSETS), (-1, 3))

        rotations_chain = [np.eye(3) for _ in range(H36MConstants.NUM_KEYPOINTS)]
        aligning_rotmats = [np.eye(3) for _ in range(H36MConstants.NUM_KEYPOINTS)]

        # compute aligning rotation matrices
        for keypoint in range(1, H36MConstants.NUM_KEYPOINTS):
            if keypoint in self.human_keypoint_to_skip:
                pass
            else:
                keypoint_parent = H36MConstants.PARENTS[keypoint]
                # keypoint_child = keypoint+1 in all cases except for the hands
                keypoint_child = keypoint + 1
                if keypoint == 19:  # LeftHand
                    keypoint_child = 22  # L_Wrist_End
                elif keypoint == 27:  # RightHand
                    keypoint_child = 30  # R_Wrist_End

                # apply parent rotations also to the current link
                human_relative_offsets[keypoint_child] = np.matmul(
                    rotations_chain[keypoint_parent],
                    H36MRelativeOffsets[keypoint_child],
                )

                # compute aligning rotmat
                amass_link = amass_relative_offsets[self.human2amass_keypoints[keypoint_child]]
                human_link = human_relative_offsets[keypoint_child]

                # if human link 12, align the spine with amass link 0-6
                if keypoint == 12:  # Spine1
                    amass_link = amass_absolute_offsets[6] - amass_absolute_offsets[0]  # spine2 - pelvis

                aligning_rotmats[keypoint] = self.rotation_matrix_from_vectors(human_link, amass_link)

                # update rotations chain
                rotations_chain[keypoint] = np.matmul(aligning_rotmats[keypoint], rotations_chain[keypoint_parent])

        return aligning_rotmats

    @staticmethod
    def rotation_matrix_from_vectors(vec1: Union[list, np.ndarray], vec2: Union[list, np.ndarray]) -> np.ndarray:
        """
        Compute the rotation matrix to align vec1 to vec2.

        ---
        ### Args:
        - `vec1` (list, ndarray): 3d "source" vector
        - `vec2` (list, ndarray): 3d "destination" vector
        ---
        ### Returns:
        - `mat` (np.ndarray): transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """

        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Compute cross product and dot product
        cross_product = np.cross(vec1, vec2)
        dot_product = np.dot(vec1, vec2)

        # Compute skew-symmetric matrix
        skew_symmetric_matrix = np.array(
            [
                [0, -cross_product[2], cross_product[1]],
                [cross_product[2], 0, -cross_product[0]],
                [-cross_product[1], cross_product[0], 0],
            ]
        )

        # Compute rotation matrix
        rotation_matrix = (
            np.eye(3) + skew_symmetric_matrix + skew_symmetric_matrix.dot(skew_symmetric_matrix) * ((1 - dot_product) / (np.linalg.norm(cross_product) ** 2))
        )
        return rotation_matrix

    def amass2h36m_mocap(
        self,
        amass_rots: np.ndarray,
        amass_trans: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert an amass mocap to the human3.6m format.

        ---
        ### Args:
        `amass_rots` (np.ndarray): array of amass rotations in angle-axis representation with shape (num_frame,num_joints,3)
        `amass_trans` (np.ndarray)(optional): array of amass root traslations with shape (num_frame,1,3)
        ---
        ### Returns:
        The converted mocap with shape (num_frame,33,3)
        """
        num_frames = amass_rots.shape[0]
        num_joints = H36MConstants.NUM_KEYPOINTS

        if amass_trans is None:
            amass_trans = np.zeros((num_frames, 1, 3))

        human_rots = np.zeros((num_frames, num_joints, 3))
        amass_rotmat = RotConverter.aa2rotmat(amass_rots)

        for keypoint in range(1, num_joints):  # TODO: and keypoint 0???
            if keypoint in self.human_keypoint_to_skip:
                pass
            else:
                human_rots[:, keypoint] = -RotConverter.rotmat2aa(
                    np.matmul(
                        amass_rotmat[:, self.human2amass_keypoints[keypoint]],
                        self.aligning_rotmats[keypoint],
                    )
                )
                # need to swap components
                if keypoint in [16, 17, 18, 19]:  # TODO: WTF?
                    human_rots[:, keypoint] = human_rots[:, keypoint, [1, 0, 2]]
                    human_rots[:, keypoint] *= (-1, 1, 1)

                if keypoint in [24, 25, 26, 27]:  # TODO: WTF?
                    human_rots[:, keypoint] = human_rots[:, keypoint, [1, 0, 2]]
                    human_rots[:, keypoint] *= (1, -1, 1)

        return np.concatenate([amass_trans, human_rots], axis=1)


def save_h36m_mocap_np2txt(
    mocap: np.ndarray,
    dir_path: str,
    filename: str,
    trans: Optional[np.ndarray] = None,
) -> None:
    """
    Function to save a h36m mocap in the same format as the original dataset

    ---
    ### Args:
    - `mocap` (np.ndarray): translation(optional) + rotations of joints in angle-axis representation with shape (num_frames, num_joints, 3)
    - `dir_path` (str): path of the saved file
    - `filename` (str): name of the saved file (without ".txt")
    - `trans` (np.ndarray)(optional): translation of root joint (num_frames, 1, 3)
    """

    original_shape = mocap.shape

    mocap = np.reshape(mocap, (-1, *original_shape[-2:]))  # (num_frame,joints,3)
    num_frames = mocap.shape[0]
    num_joints = mocap.shape[1]
    assert num_joints in [32, 33]

    # add root position if it's missing in the mocap
    if num_joints == 32:
        if trans is None:
            trans = np.zeros([num_frames, 1, 3])
        else:
            trans = np.reshape(trans, (-1, 1, 3))
        mocap = np.concatenate([trans, mocap], axis=1)

    mocap_txt = np.reshape(mocap, [-1, 99])  # (num_frame,99)

    # save to txt
    file_path = os.path.join(dir_path, f"{filename}.txt")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savetxt(file_path, mocap_txt, fmt="%.7f", delimiter=",")
