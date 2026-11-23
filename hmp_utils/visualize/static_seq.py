import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from typing import Optional

from .. import MocapLoader, BodyModelType, SmplConstants, H36MConstants, ForwardKinematics, plot_body_keypoints


def generate_rotation_matrix(axis, theta) -> np.array:
    """
    Generate the rotation matrix for a given axis and angle
    Args:
        axis (np.array): axis of rotation
        theta (float): angle of rotation
    Returns:
        np.array: rotation matrix
    """
    return expm(cross(eye(3), axis / norm(axis) * theta))


def translate_and_rotate_poses(poses, axis, theta, offset_next_pose, offset_sequences) -> np.array:
    """
    Translate and rotate the poses of a body
    Args:
        poses (np.array): poses of the body
        axis (np.array): axis of rotation
        theta (float): angle of rotation
        offset_next_pose (float): offset to translate the body along the x axis
        offset_sequences (float): offset to translate the body along the z axis
    Returns:
        np.array: poses of the body after translation and rotation
    """
    poses = np.array(poses, dtype=float, copy=True)
    rot_mat = generate_rotation_matrix(axis, theta) if axis is not None and theta != 0 else None
    for sequence in range(len(poses)):
        for frame in range(len(poses[sequence])):
            for joint in range(len(poses[sequence, frame])):
                keypoint_pos = poses[sequence, frame, joint]
                if rot_mat is not None:
                    keypoint_pos = dot(rot_mat, keypoint_pos)
                keypoint_pos[0] += frame * offset_next_pose
                keypoint_pos[2] -= sequence * offset_sequences
                poses[sequence, frame, joint] = keypoint_pos
    return poses


def plot_static_seq(
    poses_list: np.array,
    body_model_type: BodyModelType,
    body_names_colors: Optional[list] = None,
    offset_next_pose: float = 1,
    offset_sequences: float = 0,
    color_list: Optional[list] = None,
    rotation_axis: Optional[list] = None,
    rotation_angle: Optional[float] = 0,
    elevation=0,
    azimuth=-90,
    link_size: int = 1,
    link_style: Optional[list] = None,
    show_keypoint: bool = True,
    keypoint_names: Optional[list] = None,
    keypoint_size: int = 5,
    keypoint_marker="o",
    show_keypoint_id: bool = False,
    show_grid: bool = False,
    legend_location: Optional[str] = None,
    fig_opts: Optional[dict] = None,
    image_margins_factor: float = 1,
    save_svg_path: Optional[str] = None,
    debug_axis_length: Optional[float] = None,
) -> None:
    """
    Plot a sequence of static poses
    Args:
    - poses_list (np.array): np.array of poses where the first dimension is the number of sequences and the second is the number of poses
    - body_model_type (BodyModelType): body model type
    - body_names_colors (list, optional): list of tuples (body_name, color) to print
    - offset_next_pose (float): offset to translate the body along the first axis
    - offset_sequences (float): offset to translate the body along the second axis
    - rotation_axis (list, optional): axis of rotation defined as a list of 3 coordinates. Defaults to None.
    - rotation_angle (float, optional): angle of rotation
    - elevation (int, optional): elevation angle of camera view. Defaults to 0.
    - azimuth (int, optional): angle of camera view. Defaults to -90.
    - link_size (int, optional): width of the lines connecting the keypoints. Defaults to 4.
    - link_style (list, optional): style of the lines connecting the keypoints. Defaults to None.
    - show_keypoint (bool, optional): whether to show the keypoints. Defaults to True.
    - keypoint_names (list, optional): list of keypoint names to print
    - keypoint_size (int, optional): size of the keypoints. Defaults to 100.
    - color_list (list, optional): list of color tuples to plot the skeleton (joints color, keypoints color)
    - keypoint_marker (str, optional): type of marker for the keypoints. Defaults to 'o'.
    - show_keypoint_id (bool, optional): whether to show keypoint ID. Defaults to True.
    - show_grid (bool, optional): whether to show grid. Defaults to False.
    - legend_location (str, optional): location of the plot legened
    - fig_opts (dict()): options for the pyplot.figure() call
    - image_margins_factor (float): by how much the margins shold be multiplied (<1 to shrink margins)
    - save_svg_path (str, optional): path to save the plot as an svg file
    - debug_axis_length (float, optional): length of the debug axis. Defaults to None.
    """
    if fig_opts is None:
        fig_opts = {"figsize": (10, 10), "dpi": 100}

    if poses_list.ndim == 3:
        poses_list = np.expand_dims(poses_list, axis=0)  # Add a dimension at the beginning
    poses_list = translate_and_rotate_poses(poses_list, rotation_axis, rotation_angle, offset_next_pose, offset_sequences)
    poses_list = poses_list.reshape(-1, poses_list.shape[-2], poses_list.shape[-1])

    if body_model_type == BodyModelType.H36M:
        parents_list = [H36MConstants.PARENTS for _ in range(len(poses_list))]
    elif body_model_type == BodyModelType.SMPL:
        parents_list = [SmplConstants.PARENTS for _ in range(len(poses_list))]
    else:
        raise ValueError("Body model type not recognized")

    plot_body_keypoints(
        keypoint_pos_lst=poses_list,
        parents_lst=parents_list,
        num_independent_bodies=len(poses_list),
        body_names_colors=body_names_colors,
        elevation=elevation,
        azimuth=azimuth,
        link_size=link_size,
        link_style=link_style,
        show_keypoint=show_keypoint,
        keypoint_names=keypoint_names,
        keypoint_size=keypoint_size,
        color_list=color_list,
        keypoint_marker=keypoint_marker,
        show_keypoint_id=show_keypoint_id,
        show_grid=show_grid,
        legend_location=legend_location,
        fig_opts=fig_opts,
        image_margins_factor=image_margins_factor,
        save_svg_path=save_svg_path,
        debug_axis_length=debug_axis_length,
    )
