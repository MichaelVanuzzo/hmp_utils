import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from hmp_utils.motion.body_model_data import SMPL_KEYPOINTS, SMPL_OFFSETS, SMPL_PARENTS
from hmp_utils.motion.mocap_processor import load_mocap
from hmp_utils.motion.kinematics import ForwardKinematics
from hmp_utils.visualize.stickman_animation import pose_animation, CameraOrientation

# filepath = './example_mocaps/amass_sample.npz'
filepath = './example_mocaps/walking_poses.npz'
animation_framerate = 10

# -- load mocap --
motion = load_mocap(filepath, SMPL_KEYPOINTS)
assert motion is not None

# -- compute joint positions --
kine = ForwardKinematics(SMPL_OFFSETS['neutral'], SMPL_PARENTS)
keypoints_positions = kine.compute_fk(motion.rots, apply_root_rotation=True)


# -- ANIMATION 1 --
motions_list = [keypoints_positions, keypoints_positions]
subplot_ids = [0, 1]
subplot_colors = ['r', 'm', 'y']
subplot_titles = ['mocap 1', 'mocap 2', 'mocap3']
camera_orientations = [CameraOrientation(100, 20), CameraOrientation(0, 0), CameraOrientation(0, 0)]
pose_animation(motions_list, SMPL_PARENTS, subplot_ids=subplot_ids, 
               camera_orientations=camera_orientations, subplot_titles=subplot_titles,
               subplot_colors=subplot_colors)

# -- ANIMATION 2 --
# example_offset = np.array([0.5, 0, 0])
# motions_list = [keypoints_positions, keypoints_positions + example_offset]
# subplot_titles = ['MOCAP']
# camera_orientations = CameraOrientation(100, 20)
# pose_animation(motions_list, SMPL_PARENTS, camera_orientations=camera_orientations)