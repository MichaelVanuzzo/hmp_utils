import numpy as np
from typing import Union, Optional

from hmp_utils.motion.pose_conversions import aa2euler, aa2rotmat, rotmat2aa, DEFAULT_EULER_SEQ
from hmp_utils.motion.kinematics import ForwardKinematics

"""
## Example usage:
- predictions and targets are tensors with shape (..., num_frames, num_joints, 3) where the joints
    are for example the 24 SMPL joints.
- joints_to_metric: since one may want to obtain the metric for a specific subset of joints, it is
    possible to define a list of joints_to_metric that define which joints indexes should be used in the error
    calculation out of the given ones.
    So if prediction has shape (..., frames, 10, 3) and joints_to_metric = [1, 2, 4] then the joints with those
    indexes in the predictions and targets tensors will be used to compute the error.
"""


class MetricsProcessor:
    """
    Some of the metrics were inspired by [SPL - Structured Prediction Layer Helps 3D Human Motion \
    Modelling](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
    """

    def __init__(self, errors_tensor: Union[list, np.ndarray]) -> None:
        """
        Expects an array of euler errors of shape: (..., num_frames), if more than one dimension,
        performs an averaging over all the first dimensions, so that a single vector of shape
        (num_frames,) remains

        ---
        ### Args:
        - `euler_errors_arr` (list, ndarray): array of euler angles as a vector of shape (num_frames) or a tensor of shape
            (..., num_frames) in which case the mean over all the dimensions (except for the last) is computed.
        """
        # Convert to numpy array
        self.errors: np.ndarray = np.array(errors_tensor)

        # Average over the dimensions other than last if necessary
        if len(self.errors.shape) > 1:
            # compute a vector of euler errors for each frame
            axes_to_mean = tuple(i for i in range(len(self.errors.shape[:-1])))
            self.errors = np.mean(errors_tensor, axes_to_mean)

    def cumulative(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """
        Compute cumulative error from the start of the sequence. If `end_frame` is provided, returns
        the cumulative error summed up until the `at_frame`, excluded. If `end_frame` is not specified,
        returns the cumulative error including last frame. 

        ---
        ### Args:
        - `end_frame` (int): the frame at which to sum up to (including the specified frame).
        """
        if start_frame is None:
            # cumulate from the start
            start_frame = 0
        if end_frame is None:
            # cumulate including the last frame
            end_frame = len(self.errors)-1

        end_frame = self.is_valid_frame(end_frame)
        start_frame = self.is_valid_frame(start_frame)
        return np.sum(self.errors[start_frame:end_frame+1])

    def instant(self, at_frame: int):
        """
        Return the Euler error at a specified frame.
        """
        self.is_valid_frame(at_frame)
        return self.errors[at_frame]

    def mean_window(self, at_frame: int, window_len_frames: Optional[int] = None):
        """
        Compute the mean Euler error in a window with the specified number of frames prior to
        the specified frame (included in the window).

        ---
        ### Args:
        - `at_frame`: the frame at which the averaging window ends

        If `window_len_frames` is none, return the average over all the frames.
        """

        if window_len_frames is None:
            return np.mean(self.errors)

        at_frame = self.is_valid_frame(at_frame)
        window_start_frame = at_frame + 1 - window_len_frames
        return np.mean(self.errors[window_start_frame:at_frame+1])

    def is_valid_frame(self, frame: int) -> int:
        """
        Check whether the frame is a valid index for the Euler errors vector.

        Always returns the equivalent positive index
        """
        num_frames = self.errors.shape[-1]
        if frame not in range(-num_frames, num_frames):
            raise Exception(f'Provided frame index is not valid: {frame}. '
                            f'Number of frames in euler errors array: {num_frames}')
        if frame < 0:
            return len(self.errors + frame)
        return frame


def euler_error(predictions: np.ndarray, targets: np.ndarray, euler_seq: Optional[str] = None,
                joints_to_metric: Optional[Union[list, np.ndarray]] = None
                ) -> np.ndarray:
    """
    Computes the Euler angle error following:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    and: 
    https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py

    - predictions and targets must be in angle-axis representation
    - predictions and targets shapes: (..., n_joints, 3)
    - the Euler angle is computed as the Euclidean distance between flattened body pose (vectors 
        with shape (n_joints*3)) so that the return shape looks like (..., )
    - if the joints to metric are not specified, only the joints with some degree of variance will
        be used to calculate the error.
    """
    # save the first part of the shape in order to return (..., )
    return_shape = targets.shape[:-2]
    n_joints = targets.shape[-2]

    # convert into Euler representation
    if euler_seq is None:
        euler_seq = DEFAULT_EULER_SEQ
    euler_preds = aa2euler(predictions, euler_seq)
    euler_targs = aa2euler(targets, euler_seq)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints*3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints*3])

    if joints_to_metric is None:
        # Keep only the joints that have significant movement in the trajectories (in order to remove
        # end effectors from the metric computation)
        idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]
    else:
        j_ids = np.array(joints_to_metric)
        idx_to_use = np.hstack([j_ids*3+i for i in range(3)])

    # Compute euclidean distance (L2 norm) on the body poses (vectors like (num_joints*3))
    # The resulting error tensor will have 1 Euler distance per time step per sequence ecc
    euc_error = np.power(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)  # type: ignore
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # shape (-1,) # type: ignore

    # reshape to original (..., 1)
    return np.reshape(euc_error, return_shape)


def joint_angle_diff(predictions: np.ndarray, targets: np.ndarray, joints_to_metric:list, 
                     single_joints_errors:bool=False, global_rotations:bool=False):
    """
    Compute the angle needed to align each joint in the predictions to the targets.
    Expects the predictions and targets to be in angle-axis representation, as tensors of 
    shape (..., num_joints, 3)
    
    Credits to: SPL https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py
    
    ---
    ### Args:
    - `predictions` (ndarray): must be in angle-axis representation, with shape (..., num_joints, 3)
    - `targets` (ndarray): same representation and shape of predictions
    - `single_joints_errors` (bool): return shape is (..., n_joints)
    - `joints_to_metric` joints for which to return the joint angle differences (as single joints or
        averaged over the joints depending on the `single_joints_errors` parameter)
    - `global_rotations` (bool): defaults to false, meaning that the rotations are expected to be
        local (or relative). If True, the provided predictions and targets have to be in global
        (or absolute) representation

    ---
    ### Global / Local Rotations:
    - global angles means that the angle difference is calculated for each joint using the joint
        global orientation (with respect to absolute system)
    - local angles means that the metric is computed using the relative angles of the joints 
        (relative to the parent joint, that is the standard representation)

    ---
    ### Returns:
    Returns by default the mean geodesic distance for each joint as an np array of shape (..., ),
    if single_joints is True, it returns a tensor with the single joint angle difference errors
    with shape (..., num_joints)
    """

    if global_rotations:
        # here we would need to convert the joint rotations to global (absolute)
        raise NotImplementedError

    # Check shape
    if not predictions.shape[-1] == targets.shape[-1] == 3:
        raise Exception('predictions and targets have to be in angle-axis representation with 3 '
                        'components (3d vector which mangitude is the rotation angle)')
    if predictions.shape != targets.shape:
        raise Exception('Targets and predictions must have same shapes')
        
    # save the first part of the shape in order to return (..., ) or (..., nun_joints)
    return_shape = targets.shape[:-1]
    
    predictions_rotmat = aa2rotmat(predictions)
    targets_rotmat = aa2rotmat(targets)
    
    predictions_rotmat = np.reshape(predictions_rotmat, (-1, 3, 3))
    targets_rotmat = np.reshape(targets_rotmat, (-1, 3, 3))
    
    # rotation difference represented as rotation matrix: R1 * R2.T
    # due to the reshaping the first dimension (at 0) contains all the dimensions other than the
    # matrices ones; the shape at 1,2 is the 3x3 matrix which need to be transposed
    rot_diff = np.matmul(predictions_rotmat, np.transpose(targets_rotmat, (0, 2, 1)))
    
    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    aa = rotmat2aa(rot_diff)
    angles = np.linalg.norm(aa, axis=1)
        
    angles = np.array(angles)
    angles = np.reshape(angles, return_shape)
    
    if single_joints_errors:
        return angles
    else:
        # average over the joints
        return np.mean(angles[..., joints_to_metric], axis=-1)


def positional_error(predictions: np.ndarray, targets: np.ndarray, select_joints:list,
                     offsets, parents, keypoints_mean:bool,
                     joint_to_normalize:Optional[int]=None) -> np.ndarray:
    """
    Given the angle-axis representation of the joints, compute the forward kinematics on the
    skeleton and caculate the euclidean distance between each joint (among the joints selected
    for metrics).
    
    The global translation is not accounted in the positional error.
    
    ---
    ### Args:
    - predictions (ndarray): array of predicted joint relative angles in the angle-axis representation
        with shape (batch_size, num_frames, num_keypoints, 3)
    - targets (ndarray): same shape of predictions
    - keypoints_mean (bool): whether to return the distances averaged over the keypoints
    - joint_to_normalize (int, optional): chose a joint which length should be normalized to 1, which
        scale factor is used to scale all the other body links
    
    ---
    ### Return:
    The euclidean distances between predictions and targets. Rerturn shape: (batch_size, num_frames, num_keypoints) or 
        (batch_size, num_frames) if keypoints_mean is true
    """
    
    # convert angles into positions
    kine = ForwardKinematics(offsets, parents)
    
    if joint_to_normalize is not None:
        kine.normalize_joints(reference_joint=joint_to_normalize)
    
    predictions_pos = kine.compute_fk(predictions)
    targets_pos = kine.compute_fk(targets)

    print('targets_pos', targets_pos.shape)
    
    # return the euclidean distance between the joint positions
    distances = np.linalg.norm((predictions_pos[:, :, select_joints] - targets_pos[:, :, select_joints]), axis=-1)
    
    print('distances', distances.shape)
    
    if keypoints_mean:
        return np.mean(distances, axis=-1)
    else:
        return distances
    
