from hmp_utils.motion.body_model_data import SMPL_KEYPOINTS, SMPL_OFFSETS, SMPL_PARENTS, SMPLH_JOINT_NAMES
from hmp_utils.motion.mocap_processor import load_mocap
from hmp_utils.motion.kinematics import ForwardKinematics
from hmp_utils.visualize.stickman_animation import pose_animation, CameraOrientation

# filepath = './example_mocaps/amass_sample.npz'
filepath = './example_mocaps/Subject_11_F_10_poses.npz'
animation_framerate = 30

# -- load mocap --
motion = load_mocap(filepath, SMPL_KEYPOINTS)
assert motion is not None

# -- compute joint positions --
kine = ForwardKinematics(SMPL_OFFSETS['neutral'], SMPL_PARENTS)
keypoints_positions = kine.compute_fk(motion.rots, apply_root_rotation=True)

# List of motions keypoints positions, here the same motion
motions_list = [keypoints_positions, keypoints_positions+0.5]
subplot_ids = [0, 1]
subplot_titles = ['mocap 1', 'mocap 2']
camera_orientations = [CameraOrientation(100, 20), CameraOrientation(0, 20)]

pose_animation(motions_list, SMPL_PARENTS, subplot_ids=[0,1], camera_orientations=camera_orientations)