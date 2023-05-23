import numpy as np
from typing import Union, Optional
from enum import Enum, auto

from .. import RotConvert
from .. import ForwardKinematics, FkReturnType

"""
## Example usage:
- predictions and targets are tensors with shape (..., num_frames, n_joints, 3) where the joints
    are for example the 24 SMPL joints.
- joints_to_compute: since one may want to obtain the metric for a specific subset of joints, it is
    possible to define a list of joints_to_compute that define which joints indexes should be used in the error
    calculation out of the given ones.
    So if prediction has shape (..., frames, 10, 3) and joints_to_compute = [1, 2, 4] then the joints with those
    indexes in the predictions and targets tensors will be used to compute the error.
"""


class MetricsProcessor:
    def __init__(
        self,
        predictions: Union[list, np.ndarray],
        targets: Union[list, np.ndarray],
        joints_to_compute: Optional[Union[list, np.ndarray]] = None,
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
        - joints_to_compute (array-like, optional): joints for which to return the joint angle differences (as single joints or
            averaged over the joints depending on the `single_joint_errors` parameter)
        """
        if not isinstance(predictions, np.ndarray):
            self.predictions_aa = np.array(predictions)
        else:
            self.predictions_aa = predictions
        if not isinstance(targets, np.ndarray):
            self.targets_aa = np.array(targets)
        else:
            self.targets_aa = targets
        if not isinstance(joints_to_compute, np.ndarray):
            self.joints_to_compute = np.array(joints_to_compute)
        else:
            self.joints_to_compute = joints_to_compute

        if not self.predictions_aa.shape[-1] == self.targets_aa.shape[-1] == 3:
            raise Exception(
                "predictions and targets have to be in angle-axis representation with 3 "
                "components (3d vector which mangitude is the rotation angle)"
            )
        if self.predictions_aa.shape != self.targets_aa.shape:
            raise Exception("Targets and predictions must have same shapes")

        if self.joints_to_compute is None:
            joints_to_compute = [i for i in np.arange(self.targets_aa.shape[-2])]

    def euler_error(
        self,
        euler_seq: str = RotConvert.DEFAULT_EULER_SEQ,
        single_joint_errors: bool = False,
        joint_weights: Optional[Union[list, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Computes the Euler angle error as the Euclidean distance between flattened body pose (vectors
            with shape (n_joints*3)) or as the mean of the single joint distances (mean over n_joints
            and vectors with shape (3)).

        The the return shape looks like (..., ).

        If the joints to metric are not specified all the joints will be used to calculate the error.
        """
        # save the first part of the shape in order to return (..., )
        return_shape = self.targets_aa.shape[:-2]
        n_joints = self.targets_aa.shape[-2]

        # convert into Euler representation
        euler_preds = RotConvert.aa2euler(self.predictions_aa, euler_seq)
        euler_targs = RotConvert.aa2euler(self.targets_aa, euler_seq)

        # reshape to (-1, len(self.joints_to_compute), 3)
        euler_preds = np.reshape(euler_preds, [-1, n_joints, 3])[
            :, self.joints_to_compute, :
        ]
        euler_targs = np.reshape(euler_targs, [-1, n_joints, 3])[
            :, self.joints_to_compute, :
        ]

        n_joints = len(self.joints_to_compute)

        if joint_weights is None:
            joint_weights = np.ones(n_joints)
        if n_joints != len(joint_weights):
            raise Exception(
                "The length of joints_to_compute and joint_weights must be the same"
            )

        if single_joint_errors:
            # Compute euclidean distance (L2 norm) on the body joints (vectors like (3))
            # The resulting error tensor will have n_joints Euler distance per time step per sequence
            euc_distance = np.linalg.norm(euler_targs - euler_preds, axis=-1)
            euc_distance = np.average(
                euc_distance, axis=-1, weights=joint_weights
            )  # (..., )

        else:
            # reshape to (-1, n_joints*3)
            euler_preds = np.reshape(euler_preds, [-1, n_joints * 3])
            euler_targs = np.reshape(euler_targs, [-1, n_joints * 3])

            # Compute euclidean distance (L2 norm) on the body poses (vectors like (n_joints*3))
            # The resulting error tensor will have 1 Euler distance per time step per sequence
            euc_distance = np.linalg.norm(euler_targs - euler_preds, axis=-1)
            euc_distance = np.reshape(euc_distance, return_shape)  # (..., )

        return euc_distance

    def joint_angle_difference(
        self,
        use_global_rotations: bool = False,
        single_joint_errors: bool = False,
        f_kinematics_processor: Optional[ForwardKinematics] = None,
    ) -> np.ndarray:
        """
        Compute the angle needed to align each joint in the predictions to the targets.
        Expects the predictions and targets to be in angle-axis representation, as tensors of
        shape (..., n_joints, 3)

        Credits to: [SPL](https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py)

        ---
        ### Args:
        - `use_global_rotations` (bool): compute global rotations given the fk processor object
        - `single_joint_errors` (bool): return shape is (..., n_joints)
        - `f_kinematics_processor` (ForwardKinematics, optional): the kinematics object used to
            compute the forward kinematics on the given predictions/targets which always
            represent relative rotations. This is done to compute the global rotations

        ---
        ### Returns:
        By default the mean geodesic distance for each joint as an np array of shape (..., ); but
        if single_joints is True, it returns a tensor with the single joint angle difference errors
        with shape (..., n_joints)
        """
        # save the first part of the shape in order to return (..., ) or (..., n_joints)
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
        aa_diff = RotConvert.rotmat2aa(rot_diff)
        angles = np.linalg.norm(aa_diff, axis=1)

        angles = np.reshape(angles, return_shape)  # (..., n_joints)
        if not single_joint_errors:
            # average over the joints
            angles = np.mean(angles[..., self.joints_to_compute], axis=-1)

        return angles

    def positional_error(
        self,
        f_kinematics_processor: ForwardKinematics,
        joint_to_normalize: Optional[int] = None,
        single_joint_errors: bool = False,
    ) -> np.ndarray:
        """
        Given the angle-axis representation of the joints, compute the forward kinematics on the
        skeleton and caculate the euclidean distance between each joint (among the joints selected
        for metrics).

        The global translation is not accounted in the positional error.

        ---
        ### Args:
        - `f_kinematics_processor` (ForwardKinematics): the kinematics object used to
            compute the forward kinematics on the given predictions/targets which always
            represent relative rotations.
        - single_joint_errors (bool): whether to return the distances averaged over the keypoints
        - joint_to_normalize (int, optional): chose a joint which length should be normalized to 1, which
            scale factor is used to scale all the other body links

        ---
        ### Return:
        The euclidean distances between predictions and targets. Return shape:
            (..., num_keypoints) if single_joint_errors is true
            or (..., ) if not
        """

        # save the original shape in order to return (..., ) or (..., n_joints)
        original_shape = self.targets_aa.shape
        n_joints = self.targets_aa.shape[-2]

        if joint_to_normalize is not None:
            f_kinematics_processor.normalize_joints(reference_joint=joint_to_normalize)

        # convert angles into positions
        predictions_pos = f_kinematics_processor.compute_fk(
            self.predictions_aa, return_type=FkReturnType.positions
        )
        targets_pos = f_kinematics_processor.compute_fk(
            self.targets_aa, return_type=FkReturnType.positions
        )

        # remove extra dimensions if present
        predictions_pos = np.reshape(predictions_pos, [-1, n_joints, 3])[
            :, self.joints_to_compute, :
        ]
        targets_pos = np.reshape(targets_pos, [-1, n_joints, 3])[
            :, self.joints_to_compute, :
        ]

        # calculate the euclidean distance between the joint positions
        distances = np.linalg.norm(
            (predictions_pos - targets_pos),
            axis=-1,
        )

        # restore the original shape
        distances = np.reshape(distances, (*original_shape[:-2], -1))  # (..., n_joints)

        if not single_joint_errors:
            # average over the joints
            distances = np.mean(distances, axis=-1)  # (..., )

        return distances


class MetricsPostProcessor:
    """
    Some of the metrics were inspired by [SPL - Structured Prediction Layer Helps 3D Human Motion \
    Modelling](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
    """

    def __init__(self, errors_tensor: Union[list, np.ndarray]) -> None:
        """
        Expects an array of errors of shape: (..., num_frames), if more than one dimension,
        performs an averaging over all the first dimensions, so that a single vector of shape
        (num_frames,) remains

        ---
        ### Args:
        - `errors_tensor` (list, ndarray): array of errors as a vector of shape (num_frames) or a tensor of shape
            (..., num_frames) in which case the mean over all the dimensions (except for the last) is computed.
        """

        self.errors = self.mean_error_over_frames(errors_tensor)

    @staticmethod
    def mean_error_over_frames(errors_tensor: Union[list, np.ndarray]) -> np.ndarray:
        # Convert to numpy array
        if not isinstance(errors_tensor, np.ndarray):
            errors = np.array(errors_tensor)
        else:
            errors = errors_tensor

        # Average over the dimensions other than last if necessary
        if len(errors.shape) > 1:
            # compute a vector of euler errors for each frame
            axes_to_mean = tuple(i for i in range(len(errors.shape[:-1])))
            errors = np.mean(errors_tensor, axes_to_mean)
        return errors

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
