import numpy as np
from scipy import stats
from typing import Union, Optional

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
    """
    Compute errors between predictions and targets.

    """

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
        - predictions (list, ndarray): array of predicted joint relative angles in
            the angle-axis representation with shape (batch_size, num_frames,
            num_keypoints, 3)
        - targets (list, ndarray): same shape and representation of predictions
        - joints_to_compute (list, ndarray)(optional): joints for which to return the joint angle differences (as single joints or
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
        errors_computed_per_joint: bool = False,
        joint_weights: Optional[Union[list, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Computes the Euler angle error as the Euclidean distance between flattened body pose (vectors
        with shape (n_joints*3)) or as the mean of the single joint distances (mean over n_joints
        and vectors with shape (3)).

        The the return shape looks like (..., ).

        If the joints to metric are not specified all the joints will be used to calculate the error.

        ---
        ### Args:
        - `euler_seq` (str): the euler sequence to use for the conversion
        - `errors_computed_per_joint` (bool): whether to compute the euclidean distance for each
            joint separately or to compute the euclidean distance on the whole body pose vector
        - `joint_weights` (list, ndarray)(optional): the weights to use to average the errors over the
            joints. Must have the same length as the joints_to_compute list
        ---
        ### Returns:
        The euclidean distances between predictions and targets. Return shape:(..., )
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

        if errors_computed_per_joint:
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

        return euc_distance  # (..., )

    def joint_angle_difference(
        self,
        use_global_rotations: bool = False,
        single_joint_errors: bool = False,
        f_kinematics_processor: ForwardKinematics = None,
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
        - `f_kinematics_processor` (ForwardKinematics): the kinematics object used to
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
                self.predictions_aa, return_type=FkReturnType.ROTATIONS
            )
            targets = f_kinematics_processor.compute_fk(
                self.targets_aa, return_type=FkReturnType.ROTATIONS
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

        angles = np.reshape(angles, return_shape)[
            ..., self.joints_to_compute
        ]  # (..., n_joints)

        if not single_joint_errors:
            # average over the joints
            angles = np.mean(angles, axis=-1)  # (..., )

        return angles

    def positional_error(
        self,
        f_kinematics_processor: ForwardKinematics,
        joint_to_normalize: int = None,
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
        - joint_to_normalize (int): chose a joint which length should be normalized to 1, which
            scale factor is used to scale all the other body links
        - single_joint_errors (bool): whether to return the distances averaged over the keypoints
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
            self.predictions_aa, return_type=FkReturnType.POSITIONS
        )
        targets_pos = f_kinematics_processor.compute_fk(
            self.targets_aa, return_type=FkReturnType.POSITIONS
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
    Compute metrics on the errors tensor.

    Some of the metrics were inspired by [SPL - Structured Prediction Layer Helps 3D Human Motion \
    Modelling](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
    """

    def __init__(self, errors_tensor: Union[list, np.ndarray]) -> None:
        """
        Expects an array of errors of shape: (..., num_frames), if more than one dimension,
        performs an averaging over all the first dimensions, so that a single vector of shape
        (num_frames) remains.

        ---
        ### Args:
        - `errors_tensor` (list, ndarray): array of errors as a vector of shape (num_frames) or a tensor of shape
            (..., num_frames) in which case the mean over all the dimensions (except for the last) is computed.
        """

        self.errors = self.__mean_error_over_frames(errors_tensor)

    @staticmethod
    def __mean_error_over_frames(errors_tensor: Union[list, np.ndarray]) -> np.ndarray:
        """
        Compute the mean error over the frames of the given errors tensor, that is to say that
        the mean over all the dimensions (except for the last) is computed.

        ---
        ### Args:
        - `errors_tensor` (list, ndarray): array of errors as a tensor of shape (..., num_frames)
            or (num_frames)
        ---
        ### Returns:
        The mean error over the frames of the given errors tensor.
        """
        # Convert to numpy array
        if not isinstance(errors_tensor, np.ndarray):
            errors = np.array(errors_tensor)
        else:
            errors = errors_tensor

        # Average over the dimensions other than last if necessary
        if len(errors.shape) > 1:
            # compute a vector of errors for each frame
            axes_to_mean = tuple(i for i in range(len(errors.shape[:-1])))
            errors = np.mean(errors_tensor, axes_to_mean)
        return errors

    def cumulative(
        self, start_frame: int = 0, end_frame: int = None, normalize=False
    ) -> float:
        """
        Compute cumulative error from the start of the sequence.

        If `end_frame` is provided, returns the cumulative error summed up until the `end_frame`,
        excluded. If `end_frame` is not specified, returns the cumulative error including last frame.

        If `normalize` is True, the cumulative error is normalized by the number of frames.

        ---
        ### Args:
        - `start_frame` (int): the frame at which to start the cumulation (including the specified frame).
        - `end_frame` (int): the frame at which to sum up to (including the specified frame).
        - `normalize` (bool): whether to normalize the cumulative error by the number of frames.
        ---
        ### Returns:
        The cumulative error from the start of the sequence.
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

    def instant(self, at_frame: int) -> float:
        """
        Return the error at a specified frame.

        ---
        ### Args:
        - `at_frame` (int): the frame at which to return the error.
        ---
        ### Returns:
        The error at the specified frame.
        """
        self.__is_valid_frame(at_frame)
        return self.errors[at_frame]

    def windowed_mean(self, at_frame: int, window_len_frames: int = None) -> float:
        """
        Compute the mean error in a window with the specified number of frames prior to
        the specified frame (included in the window).

        If `window_len_frames` is none, return the average over all the previous frames.

        ---
        ### Args:
        - `at_frame` (int): the frame at which the averaging window ends
        - `window_len_frames` (int): the length of the averaging window in frames. If not specified,
            the average over all the previous frames is returned.
        ---
        ### Returns:
        The mean error in the window with the specified number of frames prior to the specified
        """

        at_frame = self.__is_valid_frame(at_frame)
        if window_len_frames is None:
            window_start_frame = 0
        else:
            window_start_frame = at_frame + 1 - window_len_frames

        return np.mean(self.errors[window_start_frame : at_frame + 1])

    def __is_valid_frame(self, frame: int) -> int:
        """
        Check whether the frame is a valid index for the errors vector.

        ---
        ### Args:
        - `frame` (int): the frame index to check
        ---
        ### Returns:
        The frame index if valid (always the equivalent positive index),
        otherwise raises an exception.
        """
        num_frames = self.errors.shape[-1]
        if frame not in range(-num_frames, num_frames):
            raise Exception(
                f"Provided frame index is not valid: {frame}. "
                f"Number of frames in errors array: {num_frames}"
            )
        if frame < 0:
            return len(self.errors + frame)
        return frame


class SpectrumMetricsProcessor:
    """
    Compute metrics on the power spectrum.
    """

    @staticmethod
    def power_spectral_density(
        seq_t: Union[list, np.ndarray],
        framerate: float,
        remove_zero_freq: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute the power spectral density of the given sequence of joint angles.

        ---
        ### Args:
        - `seq_t` (list, ndarray): sequence of joint angles in time domain with shape
            (..., seq_len, n_joints, 3)
        - `framerate` (float): the framerate of the sequence
        - `remove_zero_freq` (bool): whether to remove the zero frequency component or not
        - `normalize` (bool): whether to normalize the power spectral density
        ---
        ### Returns:
        The power spectral density of the given sequence of joint angles with shape
            (ps_len, n_joints, 3)
        """
        if not isinstance(seq_t, np.ndarray):
            seq_t = np.array(seq_t)

        feature_size = seq_t.shape[-1]
        n_joints = seq_t.shape[-2]
        seq_len = seq_t.shape[-3]

        seq_t = np.reshape(
            seq_t, [-1, seq_len, n_joints * feature_size]
        )  # (batch, seq_len, n_joints*3)
        batch_size = seq_t.shape[0]

        # calculate power spectrum
        seq_fft = np.fft.fft(seq_t, axis=1)  # (batch, seq_len, n_joints*3)

        # remove the zero frequency component or divide by 2 due to the symmetry of the fft
        if remove_zero_freq:
            seq_fft[:, 0, :] = 0
        else:
            seq_fft[:, 0, :] /= 2

        # remove the second half of the sequence (due to the symmetry of the fft)
        seq_fft = seq_fft[:, 0 : seq_len // 2, :]

        seq_ps = np.abs(seq_fft) ** 2  # (batch, seq_len, n_joints*3)

        # scale to obtain the power spectral density
        seq_ps = seq_ps * 2 / (framerate * seq_len)  # (batch, ps_len, n_joints*3)

        seq_ps_global = seq_ps.sum(axis=0)  # (ps_len, n_joints*3)

        if normalize:
            norm_values = seq_ps_global.sum(axis=0, keepdims=True)  # (1, n_joints*3)
            norm_values = np.where(
                norm_values == 0, 1, norm_values
            )  # avoid division by 0
            seq_ps_global /= norm_values  # (ps_len, n_joints*3)
        else:
            seq_ps_global /= batch_size  # (ps_len, n_joints*3)

        return np.reshape(
            seq_ps_global, (-1, n_joints, feature_size)
        )  # (ps_len, n_joints, 3)

    @staticmethod  # TODO all here
    def ps_entropy(
        seq: Union[list, np.ndarray], is_t_seq: bool = False, get_mean: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Compute the entropy of the power spectrum.

        Args:
            seq -> (..., seq_len, n_joints, 3)
            is_t_seq -> whether the sequence is in time domain or not
            get_mean -> whether to return the mean of the entropies or all the entropies

        Returns:
            float or (n_joints, 3) if get_mean is False
        """
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)

        if is_t_seq:
            seq_ps = SpectrumMetricsProcessor.power_spectrum(
                seq
            )  # (seq_len, n_joints, 3)
        else:
            seq_ps = seq  # (seq_len, n_joints, 3)

        entropy = stats.entropy(seq_ps, base=2, axis=0)  # (n_joints, 3)
        if get_mean:
            return np.mean(entropy)
        else:
            return entropy

    @staticmethod
    def kl_divergence(
        seq_from: Union[list, np.ndarray],
        seq_to: np.ndarray,
        is_t_seq: bool = False,
        get_mean: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Compute the KL(Kullback–Leibler) divergence between two power spectrums.

        Args:
            seq_from -> (..., seq_len, n_joints, 3)
            seq_to -> (..., seq_len, n_joints, 3)
            is_t_seq -> whether the sequences are in time domain or not
            get_mean -> whether to return the mean of the kl divergences or all the divergences

        Returns:
            float or (n_joints, 3) if get_mean is False
        """
        if not isinstance(seq_from, np.ndarray):
            seq_from = np.array(seq_from)
        if not isinstance(seq_to, np.ndarray):
            seq_to = np.array(seq_to)

        if is_t_seq:
            seq_ps_from = SpectrumMetricsProcessor.power_spectrum(
                seq_from
            )  # (seq_len, n_joints, 3)
            seq_ps_to = SpectrumMetricsProcessor.power_spectrum(
                seq_to
            )  # (seq_len, n_joints, 3)
        else:
            seq_ps_from = seq_from  # (seq_len, n_joints, 3)
            seq_ps_to = seq_to  # (seq_len, n_joints, 3)

        kl_divergence = stats.entropy(
            pk=seq_ps_from, qk=seq_ps_to, base=2, axis=0
        )  # (n_joints, 3)

        if get_mean:
            return np.mean(kl_divergence)  # float
        else:
            return kl_divergence  # (n_joints, 3)

    @staticmethod
    def js_divergence(
        seq_g: Union[list, np.ndarray],
        seq_h: Union[list, np.ndarray],
        is_t_seq: bool = False,
        get_mean: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Compute the JS (Jensen-Shannon) divergence between two power spectrums.

        Args:
            seq_from -> (..., seq_len, n_joints, 3)
            seq_to -> (..., seq_len, n_joints, 3)
            is_t_seq -> whether the sequences are in time domain or not
            get_mean -> whether to return the mean of the js divergences or all the divergences

        Returns:
            float or (n_joints, 3) if get_mean is False
        """
        if not isinstance(seq_g, np.ndarray):
            seq_g = np.array(seq_g)
        if not isinstance(seq_h, np.ndarray):
            seq_h = np.array(seq_h)

        if is_t_seq:
            seq_ps_g = SpectrumMetricsProcessor.power_spectrum(
                seq_g
            )  # (seq_len, n_joints, 3)
            seq_ps_h = SpectrumMetricsProcessor.power_spectrum(
                seq_h
            )  # (seq_len, n_joints, 3)
        else:
            seq_ps_g = seq_g  # (seq_len, n_joints, 3)
            seq_ps_h = seq_h  # (seq_len, n_joints, 3)

        jd_divergence = (
            SpectrumMetricsProcessor.kl_divergence(seq_ps_g, (seq_ps_g + seq_ps_h) / 2)
            / 2
            + SpectrumMetricsProcessor.kl_divergence(
                seq_ps_h, (seq_ps_g + seq_ps_h) / 2
            )
            / 2
        )

        if get_mean:
            return np.mean(jd_divergence)  # float
        else:
            return jd_divergence  # (n_joints, 3)

    @staticmethod
    def wasserstein_distance(
        seq_u: Union[list, np.ndarray],
        seq_v: Union[list, np.ndarray],
        is_t_seq: bool = False,
        get_mean: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Compute the Wasserstein distance between two power spectrums.

        Args:
            seq_u -> (..., seq_len, n_joints, 3)
            seq_v -> (..., seq_len, n_joints, 3)
            is_t_seq -> whether the sequences are in time domain or not
            get_mean -> whether to return the mean of the wasserstein distances or all the distances

        Returns:
            float or (n_joints, 3) if get_mean is False
        """
        if not isinstance(seq_u, np.ndarray):
            seq_u = np.array(seq_u)
        if not isinstance(seq_v, np.ndarray):
            seq_v = np.array(seq_v)

        feature_size = seq_u.shape[-1]
        n_joints = seq_u.shape[-2]

        if is_t_seq:
            seq_ps_u = SpectrumMetricsProcessor.power_spectrum(
                seq_u, normalize=True
            )  # (seq_len, n_joints, 3)
            seq_ps_v = SpectrumMetricsProcessor.power_spectrum(
                seq_v, normalize=True
            )  # (seq_len, n_joints, 3)
        else:
            seq_ps_u = seq_u  # (seq_len, n_joints, 3)
            seq_ps_v = seq_v  # (seq_len, n_joints, 3)

        w_distance = np.zeros((n_joints, feature_size))  # (n_joints, 3)
        for i in range(n_joints):
            for j in range(feature_size):
                w_distance[i][j] = stats.wasserstein_distance(
                    seq_ps_u[:, i, j], seq_ps_v[:, i, j]
                )

        if get_mean:
            return np.mean(w_distance)  # float
        else:
            return w_distance  # (n_joints, 3)
