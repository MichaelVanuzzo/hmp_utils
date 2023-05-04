import os
import logging
import numpy as np
from scipy import signal
from typing import Union
from enum import Enum, auto
from dataclasses import dataclass


class DownsampleMethod(Enum):
    resample = auto()
    decimate = auto()


def downsample_array(source_array: np.ndarray, source_framerate: float,
                     target_framerate: float,
                     method: 'DownsampleMethod' = DownsampleMethod.decimate
                     ) -> Union[np.ndarray, None]:
    """
    Downsamples the given array to a target framerate using the specified method

    ---
    ### Args:
    - `source_array` (np.ndarray): array containing the whole mocap
    - `source_framerate` (float): framerate at which the original mocap is sampled
    - `target_framerate` (float): desired framerate of the returned mocap trajectory
    - `method` (DownsampleMethod): either `decimate` or `resample`

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
            return None

    elif method == DownsampleMethod.resample:
        source_num_frames = source_array.shape[0]
        num_downsampled_frames = int(source_num_frames*target_framerate/source_framerate)
        
        downsampled_array = signal.resample_poly(source_array, num_downsampled_frames, source_num_frames)
        return np.array(downsampled_array)

    else:
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


@dataclass
class MocapData:
    # rots: relative rotations of joints with shape (num_frames, num_joints, 3)
    rots: np.ndarray
    # trans: translation of root joint (num_frames, 1, 3)
    trans: np.ndarray
    original_framerate: float
    framerate:float
    gender: str


def load_mocap(filepath, joint_ids_list:'list[int]', target_framerate: float, num_axis=3,
               downsampling_tolerance=1e-3) -> Union[MocapData, None]:
    """
    Assuming AMASS trajectories are in axis-angle representation with 3 values per joint,
    flattened with shape like (num_frames, num_keypoints*3).

    The loaded trajectory will be downsampled to the target framerate, using decimation if possible,
    resampling otherwise. If the downsamle_framerate is set to -1, the trajectories will not be downsampled.
    
    ---
    ### Args:
    - `filepath` (str): filepath to the AMASS mocap .npz file
    - `joint_ids_list` (list[int]): joint ids for which to return the trajectory
    - `target_framerate` (float): the framerate to which the mocap trajectories should be downsampled to
    - `num_axis` (int):
    
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

    mocap_file_keys = ['poses', 'trans', 'mocap_framerate']
    if any([key not in data.files for key in mocap_file_keys]):
        logging.debug(f"KeyError: one of the mocap file keys {mocap_file_keys} was not found in mocap {filepath}")
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
    rot_trajectory = rot_trajectory[:, joint_ids_list]

    # reshape to (num_frames, num_joints, 3) or (num_frames, 1, 3) for translation
    final_shape = (original_num_frames, -1, 3)
    rot_trajectory = np.reshape(rot_trajectory, final_shape)
    trans_trajectory = np.reshape(trans_trajectory, final_shape)

    # don't downsample
    if target_framerate <= 0:
        return MocapData(rot_trajectory, trans_trajectory, original_framerate, target_framerate, gender)

    # downsample to the target framerate
    if can_decimate(original_framerate, target_framerate):
        downsample_method = DownsampleMethod.decimate
    else:
        downsample_method = DownsampleMethod.resample

    rot_trajectory = downsample_array(rot_trajectory, original_framerate,
                                      target_framerate, downsample_method)
    trans_trajectory = downsample_array(trans_trajectory, original_framerate,
                                        target_framerate, downsample_method)

    if rot_trajectory is None or trans_trajectory is None:
        return None

    return MocapData(rot_trajectory, trans_trajectory, original_framerate, target_framerate, gender)
