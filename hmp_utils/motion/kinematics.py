import numpy as np
from enum import Enum, auto
from typing import Optional, Union
from .. import RotConvert, BodyModelType


"""
Credits to [SPL paper](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
"""


class FkReturnType(Enum):
    POSITIONS = auto()
    ROTATIONS = auto()


class ForwardKinematics:
    """
    Forward kinematics for SMPL-like body pose.
    """

    def __init__(
        self,
        offsets: Union[np.ndarray, list],
        parents: Union[np.ndarray, list],
        offsets_are_relative: Optional[bool] = None,
        left_mul: Optional[bool] = None,
    ) -> None:
        """
        Body offsets refer to the position of keypoints. The offsets may be
        provided as absolute vectors (like SMPL) connecting the root joint to
        the keypoint; or as relative offsets (like Human3.6M) which are vectors
        that connect the keypoint to its parent.

        ---
        ### Args:
        - offsets (list):
        - parents (list):
        - offsets_are_relative (bool): whether the given keypoints position offsets are absolute
            (an array of absolute positions of the body keypoints like AMASS), or relative like
            Human3.6M where the list contains vectors that displace the position of the keypoint
            in relation to the position of the parent keypoint. Automatically set with the
            set_body_model_type() function
        - left_mul (bool): whether to concatenate rotations like R3*R2*R1 (AMASS way) or vice versa
            (Human3.6M way). Automatically set with the set_body_model_type() function
        """

        if not isinstance(offsets, np.ndarray):
            self.body_link_offsets = np.array(offsets)
        if not isinstance(parents, np.ndarray):
            self.parents = np.array(parents)

        if len(self.body_link_offsets) != len(self.parents):
            raise Exception(
                "Length of keypoint offsets should have same length of keypoint parents list"
            )

        self.num_keypoints = len(offsets)
        self.offsets_are_relative = offsets_are_relative
        self.left_mul = left_mul

        if offsets_are_relative is False:
            self.__compute_relative_offsets()

    def __compute_relative_offsets(self):
        """
        Make the offsets relative to the parent joint (not needed for the root joint).
        """
        if self.offsets_are_relative:
            raise Exception("Offsets are already relative")
        self.body_link_offsets[1:] = (
            self.body_link_offsets[1:] - self.body_link_offsets[self.parents[1:]]
        )
        self.offsets_are_relative = True

    def set_body_model_type(self, model_type: BodyModelType):
        """
        The forward kinematics is computed in a different way between SMPL
        poses and Human3.6M poses.

        See `BODY_MODEL_TYPE` in `body_model_constants.py` the classes
        SmplConstants H36MConstants.
        - For SMPL / AMASS body models pass BodyModelType.SMPL
        - For Human3.6M body model pass BodyModelType.H36M
        """
        if model_type == BodyModelType.SMPL:
            if self.offsets_are_relative is None:
                self.__compute_relative_offsets()
            self.left_mul = False
        elif model_type == BodyModelType.H36M:
            self.offsets_are_relative = True
            self.left_mul = True
        else:
            raise Exception(f"Model type not recognized {model_type}")

    def normalize_joints(self, reference_joint):
        """
        Select a joint which length should be used as reference to normalize all the other joints,
        so that the reference joint has length 1.
        The body link length refers to the offset between `reference_joint` and its parent
        """
        ref_joint_len = np.linalg.norm(self.body_link_offsets[reference_joint])
        self.body_link_offsets = self.body_link_offsets / ref_joint_len

    def compute_fk(
        self,
        joint_rots: np.ndarray,
        apply_root_rotation: bool = False,
        root_translation: Optional[np.ndarray] = None,
        return_type: FkReturnType = FkReturnType.POSITIONS,
    ) -> np.ndarray:  # TODO: clean up
        """
        Compute forward kinematics for the object initialized with the body offsets (global or local
        positions). It computes only local positions, meaning that the positions are referenced to
        the root keypoint.

        ---
        ### Args:
        - joint_rots (ndarray): relative joint rotations in angle-axis representation.
            Shape (..., num_joints, 3). The number of joints must be equal to the total
            number of keypoints which position needs to be computed.
        - apply_root_rotation (bool): if False, the root joint will not rotate with respect to world
        - root_translation (Optional, np.ndarray): If root_translation is None, the keypoint positions are not rotat by
            a global transaltion. Otherwise, if a vector is given, all the joint positions will be
            translated by that amount. The translation is generally referred to the root joint (pelvis)
            translation with respect to world origin
        - return_type (FkReturnType): return keypoints positions by default. If rotations is chose, then rotation
            matrices representing keypoint global orientations are returned

        ---
        ### Returns:
        A numpy array containing the computed keypoint positions or rotations matrices.
        """

        if None in [self.offsets_are_relative, self.left_mul]:
            raise Exception(
                'Please indicate both values for "offsets are relative" and '
                'whether "left multiplication" should be performed'
            )

        orig_shape = joint_rots.shape

        # return 3d points or 3x3 rotmats
        batch_shape = orig_shape[:-2]
        frames_num = int(np.prod(batch_shape))

        if return_type == FkReturnType.POSITIONS:
            # return 3d positions
            return_shape = (*batch_shape, self.num_keypoints, 3)
        elif FkReturnType.POSITIONS:
            # return rotation matrices
            return_shape = (*batch_shape, self.num_keypoints, 3, 3)
        else:
            raise NotImplementedError

        # convert angle-axis notations to rotation matrices
        keyp_rotmats = RotConvert.aa2rotmat(joint_rots)
        # forward kinematic is performed on the set of keypoints
        keyp_rotmats = np.reshape(keyp_rotmats, (-1, self.num_keypoints, 3, 3))

        if not apply_root_rotation:
            keyp_rotmats[:, 0] = np.eye(3)

        # create transformation matrices 4x4: [[R, t], [[0,0,0], [1]]]
        # where R is the rotation matrix 3x3 ad t is the offset vector 3x1
        # shape: (N, num_keypoints, 4, 4)
        transf_mats = np.full(
            shape=(*keyp_rotmats.shape[:-2], 4, 4), fill_value=np.eye(4)
        )
        # translation vectors: relative link offsets
        transf_mats[:, :, :3, 3] = self.body_link_offsets
        # rotation matrices
        transf_mats[:, :, :3, :3] = keyp_rotmats

        # save forward kinematics results
        global_transf_mats = np.zeros(transf_mats.shape)

        # Initialize the world transformation matrix for the root joint
        global_transf_mats[:, 0] = transf_mats[:, 0]

        # Iterate over all child joints
        for keyp_idx in range(1, self.num_keypoints):
            # Retrieve the world transformation matrix of the parent joint
            parent_idx = self.parents[keyp_idx]
            parent_transform = global_transf_mats[:, parent_idx]

            # Compute the world transformation matrix for the current keypoint
            if self.left_mul:
                # For Human3.6M
                offset = np.expand_dims(transf_mats[:, keyp_idx, :3, 3], axis=1)
                rel_angle = transf_mats[:, keyp_idx, :3, :3]
                parent_position = np.expand_dims(parent_transform[:, :3, 3], axis=1)
                parent_rotation = parent_transform[:, :3, :3]

                # global position of this keypoint
                this_position = np.matmul(offset, parent_rotation) + parent_position
                this_rotation = np.matmul(rel_angle, parent_rotation)

                curr_transform_world = np.full(
                    shape=(frames_num, 4, 4), fill_value=np.eye(4)
                )
                curr_transform_world[:, :3, :3] = this_rotation
                curr_transform_world[:, :3, 3] = np.reshape(
                    this_position, (frames_num, 3)
                )
            else:
                # For SMPL
                curr_transform_world = np.matmul(
                    parent_transform, transf_mats[:, keyp_idx]
                )

            # Store the world transformation matrix of the current joint
            global_transf_mats[:, keyp_idx] = curr_transform_world

        # return 3d positions or 3x3 rotation matrices
        if return_type == FkReturnType.POSITIONS:
            # The last column of the transformations contains keypoints positions
            keyp_positions = global_transf_mats[:, :, :3, 3]
            keyp_positions = np.reshape(keyp_positions, return_shape)

            # apply root joint translation by translation of every point
            if root_translation is not None:
                keyp_positions += root_translation

            return keyp_positions

        elif return_type == FkReturnType.ROTATIONS:
            # the north-west 3x3 matrix in the transformation matrix contains global rotations
            keyp_glob_rotations = global_transf_mats[:, :, :3, :3]
            keyp_glob_rotations = np.reshape(keyp_glob_rotations, return_shape)
            return keyp_glob_rotations

        else:
            raise NotImplementedError
