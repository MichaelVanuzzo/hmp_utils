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


class DownsampleMethod(Enum):
    resample = auto()
    decimate = auto()
    slerp = auto()
    none = auto()


@dataclass
class MocapData:
    # rots: relative rotations of joints with shape (num_frames, num_joints, 3)
    rots: np.ndarray
    # trans: translation of root joint (num_frames, 1, 3)
    trans: np.ndarray
    original_framerate: float
    framerate: float
    gender: str


def downsample_motion(source_array: np.ndarray, source_framerate: float,
                      target_framerate: float,
                      method: 'DownsampleMethod' = DownsampleMethod.decimate
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
    if method == DownsampleMethod.decimate:
        try:
            downsampling_factor = int(round(source_framerate/target_framerate))
            original_len = int(np.floor(source_array.shape[0]/downsampling_factor))  # type: ignore
            downsampled_array = signal.decimate(source_array, downsampling_factor,
                                                zero_phase=True, axis=0)[:original_len]
            return np.array(downsampled_array)
        except ValueError:
            # "The length of the input vector x must be greater than padlen"
            logging.error('Downsampling decimation failed')
            return None

    elif method == DownsampleMethod.resample:
        source_num_frames = source_array.shape[0]
        downsampled_len_frames = int(source_num_frames*target_framerate/source_framerate)

        downsampled_array = signal.resample_poly(source_array, downsampled_len_frames, source_num_frames)
        return np.array(downsampled_array)

    elif method == DownsampleMethod.slerp:
        if source_array.shape[-1] != 3:
            raise Exception('When using slerp downsampling method, the provided source array '
                            'should be in axis-angle representation with shape (..., 3). '
                            f'Input array shape is {source_array.shape}')

        # compute at which times the sampling should occur
        original_shape = source_array.shape
        original_len_frames = original_shape[0]
        # this is not the actual duration of the sequence, but the last frame sample time
        original_len_sec = original_len_frames/source_framerate
        
        original_key_times = np.arange(start=0, stop=original_len_sec, step=1/source_framerate)
        original_key_times = original_key_times[:original_len_frames]
        target_times = np.arange(start=0, stop=original_len_sec, step=1/target_framerate)
        
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
            slerp_worker = Slerp(original_key_times, Rotation.from_rotvec(sequence))  # type:ignore
            interpolated_sequence = slerp_worker(target_times)
            interpolated_rotations.append(interpolated_sequence.as_rotvec())

        interpolated_rotations = np.array(interpolated_rotations)
        downsampled_len_frames = interpolated_rotations.shape[1]

        # now from shape (N, downsampled_len_frames, 3) to (downsampled_len_frames, N, 3)
        interpolated_rotations = np.swapaxes(interpolated_rotations, 0, 1)
        return_shape = (downsampled_len_frames, *original_shape[1:])
        return np.reshape(interpolated_rotations, return_shape)

    elif method == DownsampleMethod.none:
        return source_array

    raise Exception(f'Invalid downsampling method {method}')


def can_decimate(source_framerate, target_framerate, tolerance=1e-3):
    """
    If the source framerate is a multiple of the target downsampled framerate
    within the specified tolerance, the timeseries can be decimated (which needs an integer downsampling factor).
    Otherwise we would need to resample the signal.
    """
    ratio = source_framerate / target_framerate
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False


def load_mocap(filepath,
               keypoint_ids_to_load: Union['list[int]', np.ndarray],
               target_framerate: Optional[float] = None,
               downsampling_tolerance=1e-3,
               target_downsampling_method: DownsampleMethod = DownsampleMethod.slerp
               ) -> Union[MocapData, None]:
    """
    Assuming AMASS trajectories in axis-angle representation with 3 values per joint,
    flattened with shape like (num_frames, num_keypoints*3).

    The loaded trajectory will be downsampled to the target framerate, using slerp by default.
    If no target_framerate is provided, the trajectories will not be downsampled.

    ---
    ### Args:
    - `filepath` (str): filepath to the AMASS mocap .npz file
    - `joint_ids_list` (list[int]): joint ids for which to return the trajectory
    - `target_framerate` (float): the framerate to which the mocap trajectories should be downsampled to

    ---
    ### Returns: 
    - The downsampled mocap trajectory as the MocapData object.
    - None if the target framerate is higher than the mocap framerate

    ---
    ### Raises:
    Exception if the mocap file couldn't be loaded
    """
    data = np.load(filepath)

    if data is None:
        raise Exception(f'Could not load mocap file {filepath}')

    if target_framerate is None:
        target_framerate = float(data['mocap_framerate'])
    elif target_framerate <= 0:
        raise Exception(f'Provide a valid downsampling framerate (was given {target_framerate})')

    mocap_file_keys = ['poses', 'trans', 'mocap_framerate']
    if any([key not in data.files for key in mocap_file_keys]):
        logging.debug(
            f"KeyError: one of the mocap file keys {mocap_file_keys} was not found in mocap {filepath}")
        return None

    rot_trajectory = np.array(data['poses'])
    trans_trajectory = np.array(data['trans'])
    original_num_frames = rot_trajectory.shape[0]
    original_framerate = float(data['mocap_framerate'])
    gender = str(data['gender'])

    # check if in the downsampled traj there will be at least one frame
    num_frames_downsampled_traj = int(original_num_frames*target_framerate/original_framerate)
    if num_frames_downsampled_traj < 1:
        return None

    if original_framerate < target_framerate and abs(original_framerate-target_framerate) > downsampling_tolerance:
        logging.warning(f'Mocap sequence could not be loaded because target framerate {target_framerate} '
                        f'is higher than the mocap framerate {original_framerate} \n'
                        f'Mocap file: {os.path.basename(filepath)}')
        return None

    # select the components of the angle-axis representation to extract from the 'pose' array
    rot_trajectory = np.reshape(rot_trajectory, newshape=(original_num_frames, -1, 3))
    rot_trajectory = rot_trajectory[:, keypoint_ids_to_load]

    # reshape to (num_frames, num_joints, 3) or (num_frames, 1, 3) for translation
    final_shape = (original_num_frames, -1, 3)
    rot_trajectory = np.reshape(rot_trajectory, final_shape)
    trans_trajectory = np.reshape(trans_trajectory, final_shape)

    # don't downsample
    if abs(target_framerate - original_framerate) <= downsampling_tolerance:
        return MocapData(rot_trajectory, trans_trajectory, original_framerate, original_framerate, gender)

    # downsample to the target framerate
    if target_downsampling_method == DownsampleMethod.decimate:
        if can_decimate(original_framerate, target_framerate):
            downsample_method = DownsampleMethod.decimate
        else:
            downsample_method = DownsampleMethod.resample
    else:
        downsample_method = target_downsampling_method

    rot_trajectory = downsample_motion(rot_trajectory, original_framerate,
                                       target_framerate, downsample_method)
    trans_trajectory = downsample_motion(trans_trajectory, original_framerate,
                                         target_framerate, downsample_method)

    if rot_trajectory is None or trans_trajectory is None:
        return None

    downsampled_framerate = original_framerate*len(trans_trajectory)/original_num_frames

    return MocapData(rot_trajectory, trans_trajectory, original_framerate, downsampled_framerate, gender)
