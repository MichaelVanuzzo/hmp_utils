import numpy as np
from typing import Union, Optional
from enum import Enum, auto

from .. import RotConvert
from .. import ForwardKinematics, FkReturnType

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


class MetricsPostProcessor:
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
        if not isinstance(errors_tensor, np.ndarray):
            self.errors = np.array(errors_tensor)
        else:
            self.errors = errors_tensor

        # Average over the dimensions other than last if necessary
        if len(self.errors.shape) > 1:
            # compute a vector of euler errors for each frame
            axes_to_mean = tuple(i for i in range(len(self.errors.shape[:-1])))
            self.errors = np.mean(errors_tensor, axes_to_mean)

    def cumulative(
        self, start_frame: int = 0, end_frame: Optional[int] = None, normalize=False
    ):
        """
        Compute cumulative error from the start of the sequence. If `end_frame` is provided, returns
        the cumulative error summed up until the `end_frame`, excluded. If `end_frame` is not specified,
        returns the cumulative error including last frame. If `normalize` is True, the cumulative error
        is normalized by the number of frames.

        ---
        ### Args:
        - `start_frame` (int): the frame at which to start the cumulation (including the specified frame).
        - `end_frame` (int): the frame at which to sum up to (including the specified frame).
        """
        if end_frame is None:
            # cumulate including the last frame
            end_frame = len(self.errors) - 1

        end_frame = self.__is_valid_frame(end_frame)
        start_frame = self.__is_valid_frame(start_frame)

        error_sum = np.sum(self.errors[start_frame : end_frame + 1])
        if normalize:
            error_sum = error_sum / (end_frame - start_frame + 1)

        return error_sum

    def instant(self, at_frame: int):
        """
        Return the Euler error at a specified frame.
        """
        self.__is_valid_frame(at_frame)
        return self.errors[at_frame]

    def windowed_mean(self, at_frame: int, window_len_frames: Optional[int] = None):
        """
        Compute the mean error in a window with the specified number of frames prior to
        the specified frame (included in the window).

        ---
        ### Args:
        - `at_frame`: the frame at which the averaging window ends

        If `window_len_frames` is none, return the average over all the previous frames.
        """

        at_frame = self.__is_valid_frame(at_frame)
        if window_len_frames is None:
            window_start_frame = 0
        else:
            window_start_frame = at_frame + 1 - window_len_frames

        return np.mean(self.errors[window_start_frame : at_frame + 1])

    def __is_valid_frame(self, frame: int) -> int:
        """
        Check whether the frame is a valid index for the Euler errors vector.

        Always returns the equivalent positive index
        """
        num_frames = self.errors.shape[-1]
        if frame not in range(-num_frames, num_frames):
            raise Exception(
                f"Provided frame index is not valid: {frame}. "
                f"Number of frames in euler errors array: {num_frames}"
            )
        if frame < 0:
            return len(self.errors + frame)
        return frame


class MetricsProcessor:
    def __init__(
        self,
        predictions: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
        joints_to_metric: Optional[Union[list, np.ndarray]] = None,
    ) -> None:
        """
        - predictions and targets must be in angle-axis representation
        - predictions and targets shapes: (..., n_joints, 3)

        ---
        ### Args:
        - predictions (ndarray): array of predicted joint relative angles in
            the angle-axis representation with shape (batch_size, num_frames,
            num_keypoints, 3)
        - targets (ndarray): same shape and representation of predictions
        - joints_to_metric (array-like, optional): joints for which to return the joint angle differences (as single joints or
            averaged over the joints depending on the `single_joints_errors` parameter)
        """
        if not isinstance(predictions, np.ndarray):
            self.predictions_aa = np.array(predictions)
        else:
            self.predictions_aa = predictions
        if not isinstance(targets, np.ndarray):
            self.targets_aa = np.array(targets)
        else:
            self.targets_aa = targets
        if not isinstance(joints_to_metric, np.ndarray):
            self.joints_to_metric = np.array(joints_to_metric)
        else:
            self.joints_to_metric = joints_to_metric

        if not self.predictions_aa.shape[-1] == self.targets_aa.shape[-1] == 3:
            raise Exception(
                "predictions and targets have to be in angle-axis representation with 3 "
                "components (3d vector which mangitude is the rotation angle)"
            )
        if self.predictions_aa.shape != self.targets_aa.shape:
            raise Exception("Targets and predictions must have same shapes")

    # NOTE: add ability to return single euler errors for each joint
    def euler_error(
        self,
        euler_seq: str = RotConvert.DEFAULT_EULER_SEQ,
        single_joints_errors: bool = False,
    ) -> np.ndarray:
        """
        Computes the Euler angle error following:
        https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
        and:
        https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py

        - the Euler angle is computed as the Euclidean distance between flattened body pose (vectors
            with shape (n_joints*3)) so that the return shape looks like (..., )
        - if the joints to metric are not specified, only the joints with some degree of variance will
            be used to calculate the error.
        """
        # save the first part of the shape in order to return (..., )
        return_shape = self.targets_aa.shape[:-2]
        n_joints = self.targets_aa.shape[-2]

        # convert into Euler representation
        euler_preds = RotConvert.aa2euler(self.predictions_aa, euler_seq)
        euler_targs = RotConvert.aa2euler(self.targets_aa, euler_seq)

        # reshape to (-1, n_joints*3) to be consistent with previous work
        euler_preds = np.reshape(euler_preds, [-1, n_joints * 3])
        euler_targs = np.reshape(euler_targs, [-1, n_joints * 3])

        if self.joints_to_metric is None:
            idx_to_use = [i for i in range(n_joints)]
        else:
            j_ids = np.array(self.joints_to_metric)
            idx_to_use = np.hstack([j_ids * 3 + i for i in range(3)])

        # Compute euclidean distance (L2 norm) on the body poses (vectors like (num_joints*3))
        # The resulting error tensor will have 1 Euler distance per time step per sequence ecc
        euc_error = np.linalg.norm(
            euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], axis=1
        )

        # reshape to original (..., 1)
        return np.reshape(euc_error, return_shape)

    def joint_angle_diff(
        self,
        single_joints_errors: bool = False,
        f_kinematics_processor: Optional[ForwardKinematics] = None,
        use_global_rotations: bool = False,
    ):
        """
        Compute the angle needed to align each joint in the predictions to the targets.
        Expects the predictions and targets to be in angle-axis representation, as tensors of
        shape (..., num_joints, 3)

        Credits to: [SPL](https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py)

        ---
        ### Args:
        - `single_joints_errors` (bool): return shape is (..., n_joints)
        - `f_kinematics_processor` (ForwardKinematics, optional): the kinematics object used to
            compute the forward kinematics on the given predictions/targets which always
            represent relative rotations. This is done to compute the global rotations
        - `use_global_rotations` (bool): compute global rotations given the fk processor object

        ---
        ### Returns:
        By default the mean geodesic distance for each joint as an np array of shape (..., ); but
        if single_joints is True, it returns a tensor with the single joint angle difference errors
        with shape (..., num_joints)
        """
        # save the first part of the shape in order to return (..., ) or (..., nun_joints)
        return_shape = self.targets_aa.shape[:-1]

        if use_global_rotations:
            if f_kinematics_processor is None:
                raise Exception(
                    "When use_global_rotations is true, the f_kinematics_processor "
                    "must be provided (ForwardKinematics class)"
                )

            predictions = f_kinematics_processor.compute_fk(
                self.predictions_aa, return_type=FkReturnType.rotations
            )
            targets = f_kinematics_processor.compute_fk(
                self.targets_aa, return_type=FkReturnType.rotations
            )
        else:
            # use local rotations
            predictions = RotConvert.aa2rotmat(self.predictions_aa)
            targets = RotConvert.aa2rotmat(self.targets_aa)

        predictions_rotmat = np.reshape(predictions, (-1, 3, 3))
        targets_rotmat = np.reshape(targets, (-1, 3, 3))

        # rotation difference represented as rotation matrix: R1 * R2.T
        # due to the reshaping the first dimension (at 0) contains all the dimensions other than the
        # matrices ones; the shape at 1,2 is the 3x3 matrix which need to be transposed
        rot_diff = np.matmul(
            predictions_rotmat, np.transpose(targets_rotmat, (0, 2, 1))
        )

        # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
        # the predicted and target orientations
        aa = RotConvert.rotmat2aa(rot_diff)
        angles = np.linalg.norm(aa, axis=1)  # NOTE: check if correct

        angles = np.reshape(angles, return_shape)

        if single_joints_errors:
            return angles
        else:
            # average over the joints
            return np.mean(angles[..., self.joints_to_metric], axis=-1)

    # NOTE: change compute_keypoints_mean to single_joints_errors like other funcitons
    def positional_error(
        self,
        f_kinematics_processor: ForwardKinematics,
        compute_keypoints_mean: bool,
        joint_to_normalize: Optional[int] = None,
    ) -> np.ndarray:
        """
        Given the angle-axis representation of the joints, compute the forward kinematics on the
        skeleton and caculate the euclidean distance between each joint (among the joints selected
        for metrics).

        The global translation is not accounted in the positional error.

        ---
        ### Args:
        - keypoints_mean (bool): whether to return the distances averaged over the keypoints
        - joint_to_normalize (int, optional): chose a joint which length should be normalized to 1, which
            scale factor is used to scale all the other body links

        ---
        ### Return:
        The euclidean distances between predictions and targets. Rerturn shape:
            (batch_size, num_frames, num_keypoints) or (batch_size, num_frames)
            if keypoints_mean is true
        """

        if joint_to_normalize is not None:
            f_kinematics_processor.normalize_joints(reference_joint=joint_to_normalize)

        # convert angles into positions
        predictions_pos = f_kinematics_processor.compute_fk(
            self.predictions_aa, return_type=FkReturnType.positions
        )
        targets_pos = f_kinematics_processor.compute_fk(
            self.targets_aa, return_type=FkReturnType.positions
        )

        # return the euclidean distance between the joint positions
        distances = np.linalg.norm(
            (
                predictions_pos[:, :, self.joints_to_metric]
                - targets_pos[:, :, self.joints_to_metric]
            ),
            axis=-1,
        )

        if compute_keypoints_mean:
            return np.mean(distances, axis=-1)
        else:
            return distances
