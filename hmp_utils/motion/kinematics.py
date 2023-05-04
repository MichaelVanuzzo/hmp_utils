import numpy as np
from typing import Optional, Union
from hmp_utils.motion.pose_conversions import aa2rotmat
from hmp_utils.motion.body_model_data import SMPL_OFFSETS, SMPL_PARENTS, SMPL_NR_KEYPOINTS

# body offsets represent the vector that starts at the root joint (pelvis) and
# ends to a keypoint in the standard SMPL T-pose


# ? NOTE: SPL: normalize so that right thigh has length 1


class ForwardKinematics:
    """
    Forward kinematics for SMPL-like body pose.
    
    Inspiration credits to [SPL paper](https://ait.ethz.ch/projects/2019/spl/downloads/spl_iccv19.pdf)
    """
    def __init__(self, offsets:list, parents:list) -> None:
        """
        - number of keypoints:
            - defaults to the number of keypoints in SMPL (24)
            - it is the number of point positions that will be computed by the forward kinematics
        """
        
        if len(offsets) != len(parents):
            raise Exception('Length of keypoint offsets should have same length of keypoint parents list')
        
        self.num_keypoints = len(offsets)
        # compute body_links vectors: joint to parent displacement vectors
        self.body_links_offsets = np.zeros([self.num_keypoints, 3])
        self.body_links_offsets[0] = offsets[0]
        # compute the vectors that connect a body keypoint to its parent (not needed for the root joint)
        offsets = np.array(offsets)
        self.body_links_offsets[1:] = offsets[1:] - offsets[parents[1:]]
        self.parents = parents

        

    def compute_fk(self, joint_rots:Union[list, np.ndarray], 
                   apply_root_rotation:bool = False,
                   root_translation: Optional[Union[list, np.ndarray]] = None,
                   joint_keyp_ids: Optional[Union[list, np.ndarray]] = None):
        """
        - apply_root_rotation: if False, the root joint will not rotate with respect to world
        - joint_rots is a tensor of shape (..., num_joints, 3)
        - If root_translation is None, the keypoint positions are not translated by a global transaltion. Otherwise, 
        if a vector is given, all the joint positions will be translated by that amount. The translation is generally
        referred to the root joint (pelvis) translation with respect to world origin
        - The joint_keyp_ids are used to index the joint rotations. If keypoint_ids is None, it is assumed that
        the given joint rotations are in ascending order starting from 0
        """
        joint_rots_arr:np.ndarray = np.array(joint_rots)
        orig_shape = joint_rots_arr.shape
        
        if joint_keyp_ids is None:
            joint_keyp_ids = np.arange(len(joint_rots_arr))
        
        # return: (..., num_keypoints, 3)
        N = orig_shape[:-2]
        return_shape = (*N, self.num_keypoints, 3)
            
        # the keypoints that do not represent joints are zeros, while the ones that represent joints
        # are filled with the joint rotations
        keyp_rotations = np.zeros((*N, self.num_keypoints, 3))
        keyp_rotations[joint_keyp_ids] = joint_rots_arr
        
        # convert angle-axis notations to rotation matrices
        keyp_rotmats = aa2rotmat(keyp_rotations)
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
        
        # prepare return array with shape: (N, num_keypoints, 3)
        keyp_positions = np.zeros((*transf_mats.shape[:2], 3))
        
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
                curr_transform_world = np.matmul(parent_transform, pose_transf_mats[keyp_idx])
                
                # Store the world transformation matrix of the current joint
                transform_chain[keyp_idx] = curr_transform_world
            
            # The last column of the transformations contains keypoints positions
            # transform_chain shape: (num_keypoints, 4, 4)
            keyp_positions[pose_idx] = transform_chain[:, :3, 3]
        
        # apply root joint translation by translation of every point
        if root_translation is not None:
            root_translation = np.array(root_translation)
            keyp_positions = keyp_positions + root_translation
        
        return np.reshape(keyp_positions, return_shape)
        