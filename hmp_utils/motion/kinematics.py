import numpy as np
from enum import Enum, auto
from typing import Optional, Union
from .. import RotConvert


"""
Credits to [SPL paper](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
"""


class fk_return(Enum):
    positions = auto()
    rotations = auto()


class ForwardKinematics:
    """
    Forward kinematics for SMPL-like body pose.
    """

    def __init__(self, offsets: list, parents: list,
                 offsets_are_relative: bool = False,
                 left_mul: bool = False) -> None:
        """
        - number of keypoints:
            - defaults to the number of keypoints in SMPL (24)
            - it is the number of point positions that will be computed by the forward kinematics
        - body offsets represent the vector that starts at the root joint (pelvis) and ends to a 
            keypoint in the standard SMPL T-pose

        ---
        ### Args:
        - offsets (list):
        - parents (list):
        - offsets_are_relative (bool): whether the given keypoints position offsets are absolute
            (an array of absolute positions of the body keypoints like AMASS), or relative like
            Human3.6M where the list contains vectors that displace the position of the keypoint
            in relation to the position of the parent keypoint
        - left_mul (bool): whether to concatenate rotations like R3*R2*R1 (AMASS way) or vice versa
            (Human3.6M way)
        """

        if len(offsets) != len(parents):
            raise Exception(
                'Length of keypoint offsets should have same length of keypoint parents list')

        self.num_keypoints = len(offsets)
        # compute body_links vectors: joint to parent displacement vectors
        self.body_links_offsets = np.zeros([self.num_keypoints, 3])
        self.body_links_offsets[0] = offsets[0]
        # compute the vectors that connect a body keypoint to its parent (not needed for the root joint)
        offsets_ = np.array(offsets)
        if not offsets_are_relative:
            self.body_links_offsets[1:] = offsets_[1:] - offsets_[parents[1:]]
        else:
            self.body_links_offsets = offsets_
        self.parents = parents
        self.left_mul = left_mul

    def normalize_joints(self, reference_joint):
        """
        Select a joint which length should be used as reference to normalize all the other joints, 
        so that the reference joint has length 1
        """
        # since the root joint was removed from the body links offsets
        reference_joint = reference_joint - 1

        ref_joint_len = np.linalg.norm(self.body_links_offsets[reference_joint])
        self.body_links_offsets = self.body_links_offsets / ref_joint_len

    def compute_fk(self,
                   joint_rots: Union[list, np.ndarray],
                   apply_root_rotation: bool = False,
                   root_translation: Optional[Union[list, np.ndarray]] = None,
                   joint_keyp_ids: Optional[Union[list, np.ndarray]] = None,
                   return_type: fk_return = fk_return.positions
                   ) -> np.ndarray:
        """
        Compute forward kinematics for the object initialized with the body offsets (global or local
        positions). It computes only local positions, meaning that the positions are referenced to 
        the root keypoint.

        ---
        ### Args:
        - joint_rots (list, ndarray): relative joint rotations in angle-axis representation.
            Shape (..., num_joints, 3)
        - apply_root_rotation (bool): if False, the root joint will not rotate with respect to world
        - root_translation (): If root_translation is None, the keypoint positions are not rotat by 
            a global transaltion. Otherwise, if a vector is given, all the joint positions will be 
            translated by that amount. The translation is generally referred to the root joint (pelvis)
            translation with respect to world origin
        - joint_keyp_ids (): used to index the joint rotations. If keypoint_ids is None, it is assumed that
            the given joint rotations are in ascending order starting from 0
        - return_type (bool): return keypoints positions by default. If rotations is chose, then rotation
            matrices representing keypoint global orientations are returned

        ---
        ### Returns:
        A numpy array containing the computed keypoint positions or rotations matrices.
        """

        joint_rots_arr: np.ndarray = np.array(joint_rots)
        orig_shape = joint_rots_arr.shape

        if joint_keyp_ids is None:
            joint_keyp_ids = np.arange(len(joint_rots_arr))

        # return 3d points or 3x3 rotmats
        N = orig_shape[:-2]
        if return_type == fk_return.positions:
            # return 3d positions
            return_shape = (*N, self.num_keypoints, 3)
        elif fk_return.positions:
            # return rotation matrices
            return_shape = (*N, self.num_keypoints, 3, 3)
        else:
            raise NotImplementedError

        # the keypoints that do not represent joints are zeros, while the ones that represent joints
        # are filled with the joint rotations
        keyp_rotations = np.zeros((*N, self.num_keypoints, 3))
        keyp_rotations[joint_keyp_ids] = joint_rots_arr

        # convert angle-axis notations to rotation matrices
        keyp_rotmats = RotConvert.aa2rotmat(keyp_rotations)
        # forward kinematic is performed on the set of keypoints
        keyp_rotmats = np.reshape(keyp_rotmats, (-1, self.num_keypoints, 3, 3))

        if not apply_root_rotation:
            keyp_rotmats[:, 0] = np.eye(3)

        # create transformation matrices 4x4: [[R, t], [[0,0,0], [1]]]
        # where R is the rotation matrix 3x3 ad t is the offset vector 3x1
        # shape: (N, num_keypoints, 4, 4)
        transf_mats = np.full(shape=(*keyp_rotmats.shape[:-2], 4, 4), fill_value=np.eye(4))
        # translation vectors: relative link offsets
        transf_mats[:, :, :3, 3] = self.body_links_offsets
        # rotation matrices
        transf_mats[:, :, :3, :3] = keyp_rotmats

        # save forward kinematics results
        global_transf_mats = np.zeros(transf_mats.shape)

        # get the keypoint position for each pose (that is constituted by num_keyponts transformation
        # matrices)
        for pose_idx, pose_transf_mats in enumerate(transf_mats):
            # Initialize the world transformation matrix for the root joint
            transform_chain = np.full(shape=(self.num_keypoints, 4, 4), fill_value=np.eye(4))
            transform_chain[0] = pose_transf_mats[0]
            # Iterate over all child joints
            for keyp_idx in range(1, len(self.parents)):
                # Retrieve the world transformation matrix of the parent joint
                parent_idx = self.parents[keyp_idx]
                parent_transform = transform_chain[parent_idx]

                # Compute the world transformation matrix for the current keypoint
                if self.left_mul:
                    # For Human3.6M
                    offset = pose_transf_mats[keyp_idx, :3, 3]
                    rel_angle = pose_transf_mats[keyp_idx, :3, :3]
                    parent_rotation = parent_transform[:3, :3]
                    parent_position = parent_transform[:3, 3]

                    # global position of this keypoint
                    this_position = np.matmul(offset, parent_rotation) + parent_position
                    this_rotation = np.matmul(rel_angle, parent_rotation)

                    curr_transform_world = np.eye(4)
                    curr_transform_world[:3, :3] = this_rotation
                    curr_transform_world[:3, 3] = this_position
                else:
                    # For AMASS
                    curr_transform_world = np.matmul(parent_transform, pose_transf_mats[keyp_idx])

                # Store the world transformation matrix of the current joint
                transform_chain[keyp_idx] = curr_transform_world

            # save transform_chain, which has shape: (num_keypoints, 4, 4)
            global_transf_mats[pose_idx] = transform_chain

        # return 3d positions or 3x3 rotation matrices
        if return_type == fk_return.positions:
            # The last column of the transformations contains keypoints positions
            keyp_positions = global_transf_mats[:, :, :3, 3]
            keyp_positions = np.reshape(keyp_positions, return_shape)
            # apply root joint translation by translation of every point
            if root_translation is not None:
                root_translation = np.array(root_translation)
                keyp_positions = keyp_positions + root_translation
            return keyp_positions

        elif return_type == fk_return.rotations:
            # the north-west 3x3 matrix in the transformation matrix contains global rotations
            keyp_glob_rotations = global_transf_mats[:, :, :3, :3]
            keyp_glob_rotations = np.reshape(keyp_glob_rotations, return_shape)
            return keyp_glob_rotations

        else:
            raise NotImplementedError
