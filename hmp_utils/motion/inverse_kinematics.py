import numpy as np
from typing import Union, Optional

from .. import RotConverter, SmplConstants, H36MConstants, BodyConstants, BodyModelType


class InverseKinematics:
    """
    Compute inverse kinematics for a mocap with a given set of body offsets.
    """

    def __init__(
        self,
        offsets: Union[np.ndarray, list],
        parents: Union[np.ndarray, list],
        offsets_are_relative: Optional[bool] = None,
        left_mul: Optional[bool] = None,
    ) -> None:
        """
        Initialize the inverse kinematics object with the body offsets and parents list.

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

        if isinstance(parents, np.ndarray):
            self.parents = parents.tolist()
        else:
            self.parents = parents

        self.num_keypoints = len(offsets)

        # Find the first child of every joint
        self.childs = [-1] * self.num_keypoints
        for joint_idx in range(self.num_keypoints):
            if joint_idx in self.parents:
                self.childs[joint_idx] = self.parents.index(joint_idx)

        if len(self.body_link_offsets) != len(self.parents):
            raise Exception("Length of keypoint offsets should have same length of keypoint parents list")

        self.offsets_are_relative = offsets_are_relative
        self.left_mul = left_mul

    def set_body_model_type(self, model_type: BodyModelType) -> None:
        """
        The inverse kinematics is computed in a different way depending on the body model type.

        - For SMPL / AMASS mocap use BodyModelType.SMPL
        - For Human3.6M body mocap use BodyModelType.H36M
        """
        if model_type == BodyModelType.SMPL:
            self.offsets_are_relative = False
            self.left_mul = False
            self.model_type = BodyModelType.SMPL
        elif model_type == BodyModelType.H36M:
            self.offsets_are_relative = True
            self.left_mul = True
            self.model_type = BodyModelType.H36M
        else:
            raise Exception(f"Model type not recognized {model_type}")

    def compute_ik(
        self,
        joint_pos: np.ndarray,
        joint_indices: Union[np.ndarray, list],
        ignore_root_joint: bool = True,
    ) -> np.ndarray:
        """
        Compute inverse kinematics.

        ---
        ### Args:
        - `joint_pos` (ndarray): keypoint positions with shape (..., num_joints, 3).
        - `joint_indices` (list, ndarray): indices of the given joint_pos.
        - `ignore_root_joint` (bool): if set to True, the root joint is ignored and the positions of the
            root joint is set to the offsets of the root joint.
        ---
        ### Returns:
        A numpy array containing the computed keypoint rotations in angle-axis (expmap) representation.
        """
        if None in [self.offsets_are_relative, self.left_mul]:
            raise Exception('Please indicate both values for "offsets are relative" and ' 'whether "left multiplication" should be performed')

        if True in [self.offsets_are_relative, self.left_mul]:
            raise Exception("offsets_are_relative and left_mul set to true is not yet tested (H3.6M)")

        # Assuming `self.num_keypoints` and `self.childs` are already defined within the class this method is part of.
        not_provided_joints = set(range(self.num_keypoints)).difference(joint_indices)
        if ignore_root_joint:
            root_childs = [joint_idx for joint_idx, parent_joint in enumerate(self.parents) if parent_joint == 0]
            not_provided_joints = not_provided_joints.difference([0] + root_childs)
        missing_joint_with_childs = [joint for joint in not_provided_joints if self.childs[joint] != -1]

        if len(missing_joint_with_childs) != 0:
            raise Exception(f"according to joint_indices, {missing_joint_with_childs} is missing")

        input_shape = list(joint_pos.shape)
        # Check if input has the right shape
        if len(input_shape) < 2 or not input_shape[-1] == 3:
            raise Exception(
                f"""joint_pos must have at least two dimensions in the form 
                            (..., num_keypoints, 3), but the input shape is {input_shape}"""
            )

        # Reshape to (frames, num of input keypoints, 3)
        joint_pos = np.reshape(joint_pos, (-1, input_shape[-2], 3))
        tmp_shape = list(joint_pos.shape)
        tot_frames = tmp_shape[0]

        # Update new_shape and return_shape(because of missing joints)
        tmp_shape[1] = self.num_keypoints
        output_shape = input_shape.copy()
        output_shape[-2] = self.num_keypoints

        # Add zeros in the position of the missing joints to get the correct shape
        full_joint_pos = np.zeros(tmp_shape)
        full_joint_pos[:, joint_indices] = joint_pos

        if ignore_root_joint:
            if self.offsets_are_relative:
                # here it should compute the forward kinematic up to the joint, using [0,0,0]
                # for the current joint expmap, to find the keypoint absolute position
                raise Exception(f"offsets_are_relative with ignore_root_joints not implemented yet")
            else:
                full_joint_pos[:, [0] + root_childs] = self.body_link_offsets[[0] + root_childs]

        # Initialize tensors
        all_aa = np.zeros(tmp_shape)
        all_rot_mat = np.full(shape=(tot_frames, self.num_keypoints, 3, 3), fill_value=np.eye(3))

        for joint_idx in range(self.num_keypoints):
            # skip joints with no childs
            if self.childs[joint_idx] == -1:
                continue
            # skip root joints if ignore_root_joints is True
            if joint_idx == 0 and ignore_root_joint:
                continue

            # find parent and child joint
            parent_joint = self.parents[joint_idx]
            child_joint = self.childs[joint_idx]

            # calculate reference link vector and norm
            ref_link_vector = np.array(self.body_link_offsets[child_joint])
            if not self.offsets_are_relative:
                ref_link_vector -= np.array(self.body_link_offsets[joint_idx])
            ref_link_norm = np.linalg.norm(ref_link_vector)

            # calculate new link vector and norm
            new_link_vector = full_joint_pos[:, child_joint] - full_joint_pos[:, joint_idx]
            if self.left_mul:
                new_link_vector = np.expand_dims(new_link_vector, axis=-1)
                if parent_joint != -1:
                    new_link_vector = np.matmul(all_rot_mat[:, parent_joint], new_link_vector)
            else:
                new_link_vector = np.expand_dims(new_link_vector, axis=-2)
                if parent_joint != -1:
                    new_link_vector = np.matmul(new_link_vector, all_rot_mat[:, parent_joint])
            new_link_vector = np.reshape(new_link_vector, (tot_frames, 3))
            new_link_norm = np.linalg.norm(new_link_vector, axis=-1)

            # calculate angle between offset and real segment
            dot_prod = np.dot(new_link_vector, ref_link_vector)
            norm_prod = new_link_norm * ref_link_norm
            angle = np.expand_dims(np.arccos(dot_prod / norm_prod), axis=-1)

            # calculate axis between offset and real segment
            cross_prod = np.cross(new_link_vector, ref_link_vector)
            if not self.left_mul:
                cross_prod = -cross_prod  # adjust sign for Smpl
            cross_prod_norm = np.expand_dims(np.linalg.norm(cross_prod, axis=-1), axis=-1)
            axis = cross_prod / cross_prod_norm

            # angle-axis (expmap) notation for the joint
            all_aa[:, joint_idx] = axis * angle

            # for joint 0 align link1, link2, and link3
            if joint_idx == 0 and self.model_type == BodyModelType.SMPL:

                ref_link_vector_2 = np.array(self.body_link_offsets[2])
                if not self.offsets_are_relative:
                    ref_link_vector_2 -= np.array(self.body_link_offsets[joint_idx])
                
                # compute rotmat to align link1
                align_link1_rotmat = RotConverter.aa2rotmat(all_aa[:, joint_idx])

                # replicate the offset vector for all frames
                ref_link_vector_1 = (np.tile(ref_link_vector, (tot_frames, 1)))
                ref_link_vector_2 = (np.tile(ref_link_vector_2, (tot_frames, 1)))

                # target links
                new_link_vector_2 = full_joint_pos[:, 2] - full_joint_pos[:, joint_idx]
                new_link_vector_2 = np.expand_dims(new_link_vector_2, axis=-2)
                new_link_vector_2 = np.reshape(new_link_vector_2, (tot_frames, 3))

                # align the link vectors for link1 and link2
                first_rot_link1 = np.reshape(np.matmul(align_link1_rotmat, ref_link_vector_1[:,:,np.newaxis]), (tot_frames, 3))
                first_rot_link2 = np.reshape(np.matmul(align_link1_rotmat, ref_link_vector_2[:,:,np.newaxis]), (tot_frames, 3))

                # compute new rotation axis around link1
                new_axis = first_rot_link1 / np.linalg.norm(first_rot_link1, axis=1, keepdims=True)

                # get normalized links 2 to compute aligning rotation around new axis
                L = first_rot_link2 / np.linalg.norm(first_rot_link2, axis=1, keepdims=True)
                T = new_link_vector_2 / np.linalg.norm(new_link_vector_2, axis=1, keepdims=True)

                # Project L and T onto the plane perpendicular to new_axis
                a = L * new_axis
                dot_prod_L = np.sum(L * new_axis, axis=1)
                L_perp = L - dot_prod_L[:, np.newaxis] * new_axis
                dot_prod_T = np.sum(T * new_axis, axis=1)
                T_perp = T - dot_prod_T[:, np.newaxis] * new_axis

                # Normalize the projections
                L_perp /= np.linalg.norm(L_perp, axis=1, keepdims=True)
                T_perp /= np.linalg.norm(T_perp, axis=1, keepdims=True)

                # Calculate the angle between the projections
                cos_theta = np.sum(L_perp * T_perp, axis=1)
                theta = np.arccos(np.clip(cos_theta[:, np.newaxis], -1.0, 1.0))

                # Determine if we need to reverse the direction of the angle
                cross_product = np.cross(L_perp, T_perp) # N,3
                for i in range(cross_product.shape[0]):
                    if np.dot(cross_product[i], new_axis[i]) < 0:
                        theta[i] = -theta[i]

                # compute second rotmat
                align_link2_rotmat = RotConverter.aa2rotmat((new_axis * theta))

                # compute full rotation rotmat and aa
                full_rotmat = np.matmul(align_link2_rotmat, align_link1_rotmat)
                full_aa = RotConverter.rotmat2aa(full_rotmat)

                # update aa for joint 0
                all_aa[:, joint_idx] = full_aa

            # save cumulative rotation up to the joint
            rot_mat_joint = RotConverter.aa2rotmat(all_aa[:, joint_idx])
            if parent_joint != -1:
                if self.left_mul:
                    all_rot_mat[:, joint_idx] = np.matmul(rot_mat_joint, all_rot_mat[:, parent_joint])
                else:
                    all_rot_mat[:, joint_idx] = np.matmul(all_rot_mat[:, parent_joint], rot_mat_joint)
            else:
                all_rot_mat[:, joint_idx] = rot_mat_joint

        return np.reshape(all_aa, output_shape)
