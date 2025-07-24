import numpy as np
from enum import Enum, auto
from typing import Optional, Union
from .. import RotConverter, BodyModelType


class FkReturnType(Enum):
    POSITIONS = auto()
    ROTATIONS = auto()


class ForwardKinematics:
    """
    Compute forward kinematics for a mocap with a given set of body offsets.
    """

    def __init__(
        self,
        offsets: Union[np.ndarray, list],
        parents: Union[np.ndarray, list],
        offsets_are_relative: Optional[bool] = None,
        left_mul: Optional[bool] = None,
    ) -> None:
        """
        Initialize the forward kinematics object with the body offsets and parents list.

        ---
        ### Args:
        - `offsets` (list, ndarray): keypoint offsets, the first element is the root joint offset.
        - `parents` (list, ndarray): keypoint parents, the first element is the root joint parent.
        - `offsets_are_relative` (bool)(optional): whether the given keypoints position offsets are absolute
            (an array of absolute positions of the body keypoints like AMASS), or relative like
            Human3.6M where the list contains vectors that displace the position of the keypoint
            in relation to the position of the parent keypoint. Automatically set with the
            set_body_model_type() function.
        - `left_mul` (bool)(optional): whether to concatenate rotations like R3*R2*R1 (AMASS way) or vice versa
            (Human3.6M way). Automatically set with the set_body_model_type() function.
        """
        if not isinstance(offsets, np.ndarray):
            self.body_link_offsets = np.array(offsets)
        else:
            self.body_link_offsets = offsets

        if not isinstance(parents, np.ndarray):
            self.parents = np.array(parents)
        else:
            self.parents = parents

        if len(self.body_link_offsets) != len(self.parents):
            raise Exception("Length of keypoint offsets should have same length of keypoint parents list")

        self.num_keypoints = len(offsets)
        self.offsets_are_relative = offsets_are_relative
        self.left_mul = left_mul

        if offsets_are_relative is False:
            self.__compute_relative_offsets()

    def __compute_relative_offsets(self) -> None:
        """
        Make the offsets relative to the parent joint (not needed for the root joint).
        """
        if self.offsets_are_relative:
            raise Exception("Offsets are already relative")
        self.body_link_offsets[1:] = self.body_link_offsets[1:] - self.body_link_offsets[self.parents[1:]]
        self.offsets_are_relative = True

    def set_body_model_type(self, model_type: BodyModelType) -> None:
        """
        The forward kinematics is computed in a different way depending on the body model type.

        - For SMPL / AMASS mocap use BodyModelType.SMPL
        - For Human3.6M body mocap use BodyModelType.H36M
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

    def normalize_joints(self, reference_joint: int) -> None:
        """
        Select a joint which length should be used as reference to normalize all the other joints,
        so that the reference joint has length 1.
        The body link length refers to the offset between `reference_joint` and its parent.

        ---
        ### Args:
        - `reference_joint` (int): index of the joint to use as reference
        """
        ref_joint_len = np.linalg.norm(self.body_link_offsets[reference_joint])
        self.body_link_offsets = self.body_link_offsets / ref_joint_len

    def compute_fk(
        self,
        joint_rots: np.ndarray,
        apply_root_rotation: bool = False,
        root_translation: Optional[np.ndarray] = None,
        return_type: FkReturnType = FkReturnType.POSITIONS,
    ) -> np.ndarray:
        """
        Compute forward kinematics for the object initialized with the body offsets (global or local
        positions). It computes only local positions, meaning that the positions are referenced to
        the root keypoint.

        ---
        ### Args:
        - `joint_rots` (ndarray): relative joint rotations in angle-axis representation with shape (..., num_joints, 3).
            The number of joints must be equal to the total number of keypoints which position needs to be computed.
        - `apply_root_rotation` (bool): if set to True, the root joint will rotate with respect to world
        - `root_translation` (np.ndarray)(optional): if root_translation is used, the keypoint positions are translated by
            a global transaltion.
        - `return_type` (FkReturnType): return keypoints positions by default. If rotations is chose, then rotation
            matrices representing keypoint global orientations are returned.
        ---
        ### Returns:
        A numpy array containing the computed keypoint positions or rotations matrices.
        """
        if None in [self.offsets_are_relative, self.left_mul]:
            raise Exception('Please indicate both values for "offsets are relative" and ' 'whether "left multiplication" should be performed')

        orig_shape = joint_rots.shape
        batch_shape = orig_shape[:-2]
        frames_num = int(np.prod(batch_shape))

        # return 3d points or 3x3 rotmats
        if return_type == FkReturnType.POSITIONS:
            # return 3d positions
            return_shape = (*batch_shape, self.num_keypoints, 3)
        elif return_type == FkReturnType.ROTATIONS:
            # return rotation matrices
            return_shape = (*batch_shape, self.num_keypoints, 3, 3)
        else:
            raise NotImplementedError

        # convert angle-axis notations to rotation matrices
        joint_rotmat = RotConverter.aa2rotmat(joint_rots)
        # forward kinematic is performed on the set of keypoints
        joint_rotmat = np.reshape(joint_rotmat, (-1, self.num_keypoints, 3, 3))

        if not apply_root_rotation:
            joint_rotmat[:, 0] = np.eye(3)

        # create transformation matrices 4x4: [[R, t], [[0,0,0], [1]]]
        # where R is the rotation matrix 3x3 ad t is the offset vector 3x1
        # shape: (N, num_keypoints, 4, 4)
        relative_transf_mat = np.full(shape=(*joint_rotmat.shape[:-2], 4, 4), fill_value=np.eye(4))
        # translation vectors: relative link offsets
        relative_transf_mat[:, :, :3, 3] = self.body_link_offsets
        # rotation matrices
        relative_transf_mat[:, :, :3, :3] = joint_rotmat

        # save forward kinematics results
        global_transf_mat = np.zeros(relative_transf_mat.shape)

        # Initialize the world transformation matrix for the root joint
        global_transf_mat[:, 0] = relative_transf_mat[:, 0]

        # Iterate over all child joints
        for keypoint_idx in range(1, self.num_keypoints):
            # Retrieve the world transformation matrix of the parent joint
            parent_idx = self.parents[keypoint_idx]
            parent_transform = global_transf_mat[:, parent_idx]

            # Compute the world transformation matrix for the current keypoint
            if self.left_mul:
                # for Human3.6M
                relative_offset = np.expand_dims(relative_transf_mat[:, keypoint_idx, :3, 3], axis=1)
                relative_rotmat = relative_transf_mat[:, keypoint_idx, :3, :3]
                parent_position = np.expand_dims(parent_transform[:, :3, 3], axis=1)
                parent_rotmat = parent_transform[:, :3, :3]

                # global position of this keypoint
                keypoint_global_position = np.matmul(relative_offset, parent_rotmat) + parent_position
                keypoint_global_rotmat = np.matmul(relative_rotmat, parent_rotmat)

                keypont_global_transf_mat = np.full(shape=(frames_num, 4, 4), fill_value=np.eye(4))
                keypont_global_transf_mat[:, :3, :3] = keypoint_global_rotmat
                keypont_global_transf_mat[:, :3, 3] = np.reshape(keypoint_global_position, (frames_num, 3))
            else:
                # for SMPL
                keypont_global_transf_mat = np.matmul(parent_transform, relative_transf_mat[:, keypoint_idx])

            # Store the world transformation matrix of the current joint
            global_transf_mat[:, keypoint_idx] = keypont_global_transf_mat

        # return 3d positions or 3x3 rotation matrices
        if return_type == FkReturnType.POSITIONS:
            # The last column of the transformations contains keypoints positions
            global_positions = global_transf_mat[:, :, :3, 3]
            global_positions = np.reshape(global_positions, return_shape)

            # apply root joint translation by translation of every point
            if root_translation is not None:
                global_positions += root_translation

            return global_positions

        elif return_type == FkReturnType.ROTATIONS:
            # the north-west 3x3 matrix in the transformation matrix contains global rotations
            global_rotmat = global_transf_mat[:, :, :3, :3]
            return np.reshape(global_rotmat, return_shape)

        else:
            raise NotImplementedError
