import numpy as np
from typing import Union as U
from typing import Optional as O
from hmp_utils.common.rot_convert import RotConvert
from hmp_utils.motion.body_models_constants import SmplConstants, H36MConstants, BodyConstants

class IK:

    @staticmethod
    def xyz_to_aa(xyz_positions: np.ndarray,
                  body_constants: O[BodyConstants] = None,
                  parents: O[U[np.ndarray, list]] = None,
                  childs: O[U[np.ndarray, list]] = None,
                  offsets: O[U[np.ndarray, list]] = None,
                  joints_to_ignore: O[U[np.ndarray, list]] = None,
                  offsets_are_relative: O[bool] = None,
                  left_mul: O[bool] = None
                  ) -> np.ndarray:
        """
        Calculate the angle-axis notation for a given body pose starting from the absolute 
        positions of the keypoints.

        ---
        ### Args:
        - xyz_positions (ndarray): 3d absolute positions of the keypoints. Shape=(...,num_keypoints,3)

        ### Optional Args: 
        (If not specified they are automatically defined based on body_model_type)
        - body_model_type (BodyModelType): used to define missing parameters. Actually supported
        body models are 'BodyModelType.smpl' and 'BodyModelType.h36m'
        - parents (list): list of the parents of the joints
        - childs (list): used to find the child of a joint, -1 if there is no child or more then one
        - offsets (list): lenght of the bones in the body
        - joints_to_ignore (list): joints that must not be calculated, because they are not
        present or because they are end effectors or for other reasons. The return value for these
        joints is [0,0,0]
        - offsets_are_relative (bool): True if the offset of a joint is defined based on his 
        parent joint, False if it's global
        - left_mul (bool): True if the order of matrix multiplication is the same as the one used on
        Human3.6m

        ---
        ### Returns:
        ndarray of the orinal shape containing the angle-axis (expmap) representation 
        of the input positions
        """

        # Save original shape
        original_shape=xyz_positions.shape

        # Check if input has the right shape
        if len(original_shape)<2 or not original_shape[-1]==3:
            raise Exception(f"""xyz_positions must have at least two dimensions in the form 
                            (num_keypoints,3), but the input shape is {original_shape}""")

        # Define missing costants based on body_model_type
        parents = parents or body_constants.PARENTS
        childs = childs or body_constants.CHILDS
        offsets = offsets or body_constants.OFFSETS
        joints_to_ignore = joints_to_ignore or body_constants.JOINTS_TO_IGNORE
        offsets_are_relative = offsets_are_relative or body_constants.OFFSETS_ARE_RELATIVE
        left_mul = left_mul or body_constants.LEFT_MUL

        if body_constants==None:
            if None in [parents,childs,offsets,
                        joints_to_ignore,offsets_are_relative,left_mul]:
                raise Exception(f"Please specify a body model type or all the other parameters")
        elif body_constants not in [H36MConstants, SmplConstants]:
            raise Exception(f"Body constants type not supported")
        
        # Calculate shape related parameters and reshape to (frame,num_keypoints,3)
        num_keypoints=original_shape[-2]
        xyz_positions=np.reshape(xyz_positions,(-1,num_keypoints,3))
        frames=xyz_positions.shape[-3]

        # Initialize tensors
        aa=np.zeros((frames,num_keypoints,3))
        rot_mat=np.full(shape=(frames,num_keypoints,3,3), fill_value=np.eye(3))

        # Joints > 0
        for joint in range (1,num_keypoints):
            if joint in joints_to_ignore:
                continue

            # find parent and child joint, exit if there is no child
            parent_joint=parents[joint]
            child_joint=childs[joint]
            if child_joint==-1:
                raise Exception(f"Joint {joint} has no childs or more than one")
            
            # calculate offset and real segment and their norm
            base=np.array(offsets[child_joint])
            if not offsets_are_relative:
                base-=np.array(offsets[joint])
            base_=np.linalg.norm(base)
            real=xyz_positions[:,child_joint]-xyz_positions[:,joint]
            if left_mul:
                real=np.expand_dims(real,axis=-1)
                real=np.matmul(rot_mat[:,parent_joint],real)
            else:
                real=np.expand_dims(real,axis=-2)
                real=np.matmul(real,rot_mat[:,parent_joint])
            real=np.reshape(real,(frames,3))
            real_=np.linalg.norm(real,axis=-1)

            # calculate angle between offset and real segment
            dot_prod=real @ base
            norm_prod=real_ * base_
            angle=np.expand_dims(np.arccos(dot_prod/norm_prod),axis=-1)

            # calculate axis between offset and real segment
            cross_prod=np.cross(real,base)
            if not left_mul:
                # adjust sign for Smpl
                cross_prod=-cross_prod
            cross_prod_=np.expand_dims(np.linalg.norm(cross_prod,axis=-1),axis=-1)
            axis=cross_prod/cross_prod_
            
            # angle-axis (expmap) notation for the joint
            aa[:,joint]=axis*angle

            # save cumulative rotation up to the joint
            rot_mat_temp=RotConvert.aa2rotmat(aa[:,joint])
            if left_mul:
                rot_mat[:,joint]=np.matmul(rot_mat_temp,rot_mat[:,parent_joint])
            else:
                rot_mat[:,joint]=np.matmul(rot_mat[:,parent_joint],rot_mat_temp)
        return np.reshape(aa,original_shape)