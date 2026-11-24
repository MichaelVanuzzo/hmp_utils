"""
# Visualization Demo Script

This script is intended for testing / demoing the stickman visualization utility.
You can change the animation parameters to test different options.
The example mocap is an AMASS dataset sample you can download from [here](https://amass.is.tue.mpg.de/index.html).
"""

import numpy as np
from hmp_utils.motion.body_models_constants import SmplConstants
from hmp_utils.motion.mocap_processor import MocapLoader
from hmp_utils.motion.kinematics import ForwardKinematics
from hmp_utils.visualize.stickman_animation import pose_animation, CameraOrientation

# -- ANIMATION PARAMETERS --
filepath = "./example_mocaps/walking_poses.npz"
animation_framerate = 30

# -- load mocap --
mocap_loader = MocapLoader(body_model_type=SmplConstants.BODY_MODEL_TYPE, keypoint_ids_to_load=SmplConstants.KEYPOINTS, target_framerate=animation_framerate)
motion = mocap_loader.load_mocap(filepath)
assert motion is not None

# -- compute joint positions --
kine = ForwardKinematics(SmplConstants.OFFSETS, SmplConstants.PARENTS)
kine.set_body_model_type(SmplConstants.BODY_MODEL_TYPE)
keypoints_positions = kine.compute_fk(motion.rots, apply_root_rotation=True)

# -- ask user --
to_show = int(input("Which animation do you want to see? (1 or 2): "))

# -- ANIMATION 0 --
if to_show == 1:
    motions_list = [keypoints_positions, keypoints_positions]
    subplot_ids = [0, 1]
    subplot_colors = ["r", "m", "y"]
    subplot_titles = ["mocap 1", "mocap 2", "mocap3"]
    camera_orientations = [CameraOrientation(100, 20), CameraOrientation(0, 0), CameraOrientation(0, 0)]
    pose_animation(
        motions_list,
        SmplConstants.PARENTS,
        subplot_ids=subplot_ids,
        camera_orientations=camera_orientations,
        subplot_titles=subplot_titles,
        subplot_colors=subplot_colors,
        animation_framerate=animation_framerate,
    )

# -- ANIMATION 1 --
if to_show == 2:
    example_offset = np.array([0.5, 0, 0])
    motions_list = [keypoints_positions, keypoints_positions + example_offset]
    subplot_titles = ["MOCAP"]
    camera_orientations = CameraOrientation(100, 20)
    pose_animation(motions_list, SmplConstants.PARENTS, camera_orientations=camera_orientations, animation_framerate=animation_framerate)
