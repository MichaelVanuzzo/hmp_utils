import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D
from .. import RotConverter


def plot_body_keypoints(
    keypoint_pos_lst: list,
    parents_lst: Optional[list] = None,
    num_independent_bodies: int = 1,
    link_size: int = 4,
    link_style: Optional[list] = None,
    show_keypoint: bool = True,
    keypoint_names: Optional[list] = None,
    keypoint_size: int = 100,
    color_list: Optional[list] = None,
    keypoint_marker="o",
    show_keypoint_id: bool = True,
    elevation=0,
    azimuth=-90,
    show_grid: bool = False,
    legend_location: Optional[str] = None,
    fig_opts: Optional[dict] = None,
    image_margins_factor: float = 0.9,
    save_svg_path: Optional[str] = None,
    body_rotation: tuple = (0, 0, 0),
    debug_axis_length: Optional[float] = None,
) -> None:
    """
    Scatter plot of body keypoints positions in 3d, connected if a parents list is provided

    ---
    ### Args:
    - keypoints_pos_lst (list): list of keypoint positions of shape (num_keypoints, 3), or a list of
        indipendent poses
    - parents_lst (list, optional): list of parents, in order to draw connecting links, same as
        keypoint_pos_lst. If parents list does not contain a parent for each independent body,
        those missing parents are set to None
    - num_independent_bodies: whether the positions belong to the same body or multiple bodies to plot
        in the same space
    - link_size (int, optional): width of the lines connecting the keypoints. Defaults to 4.
    - show_keypoint (bool, optional): whether to show the keypoints. Defaults to True.
    - keypoint_names (list, optional): list of keypoint names to print
    - keypoint_size (int, optional): size of the keypoints. Defaults to 100.
    - color_list (list, optional): list of color tuples to plot the skeleton (joints color, keypoints color)
    - keypoint_marker (str, optional): type of marker for the keypoints. Defaults to 'o'.
    - show_keypoint_id (bool, optional): whether to show keypoint ID. Defaults to True.
    - elevation (int, optional): elevation angle of camera view. Defaults to 0.
    - azimuth (int, optional): angle of camera view. Defaults to -90.
    - show_grid (bool, optional): whether to show grid. Defaults to False.
    - legend_location (str, optional): location of the plot legened
    - fig_opts (dict()): options for the pyplot.figure() call
    - image_margins_factor (float): by how much the margins shold be multiplied (<1 to shrink margins)
    - save_svg_path (str, optional): path to save the plot as an svg file
    - body_rotation (tuple(float)): euler angle in degrees (sequence xyz) to rotate the whole body poses
        this is deprecated but still supported
    - debug_axis_length (float, optional): length of the debug axis. Defaults to None.
    """

    if color_list is None:
        color_list = [
            ("b", "#000080"),
            ("g", "#006400"),
            ("r", "#8B0000"),
            ("c", "#008B8B"),
            ("m", "#800080"),
            ("y", "#808000"),
            ("#202020", "k"),
        ]

    fig_opts = {} if fig_opts is None else fig_opts
    # if fig_opts is None:
    #     fig_opts = {"figsize": (10, 10), "dpi": 80}

    if fig_opts is not None:
        fig = plt.figure(**fig_opts)
    ax = fig.add_subplot(111, projection="3d")

    if num_independent_bodies == 1:
        parents_lst = [parents_lst]
        keypoint_pos_lst = [keypoint_pos_lst]
        if keypoint_names is not None:
            keypoint_names = [keypoint_names]
    elif num_independent_bodies < 1:
        raise Exception("num_independent_bodies must be >=1")

    # keypoint_pos_lst is a list of poses
    # this is deprecated but still supported
    if body_rotation != (0, 0, 0):
        body_rotation = np.array(body_rotation) * np.pi / 180
        view_rotmat = RotConverter.euler2rotmat(body_rotation, eul_seq=("xyz"))
        for idx, body in enumerate(keypoint_pos_lst):
            keypoint_pos_lst[idx] = np.apply_along_axis(lambda x: np.matmul(view_rotmat, x), axis=-1, arr=body)

    plot_ranges_lst = []
    for pose_id, pose in enumerate(keypoint_pos_lst):
        color_id = pose_id % len(color_list)
        link_color, keypoint_color = color_list[color_id]

        if parents_lst is not None:
            try:
                parents = parents_lst[pose_id]
            except IndexError:
                parents = None
        else:
            parents = None

        for i, point in enumerate(pose):
            # Plot the keypoint location
            keyp_name = ""
            if keypoint_names is not None:
                keyp_name = f"{i}: {keypoint_names[pose_id][i]}" if i < len(keypoint_names[pose_id]) else ""
            if show_keypoint:
                ax.scatter(
                    point[0],
                    point[1],
                    point[2],
                    color=keypoint_color,
                    marker=keypoint_marker,
                    label=keyp_name,
                    s=keypoint_size,
                )

            # Keypoint label
            if show_keypoint_id:
                ax.text(point[0], point[1], point[2], f"{i}", fontsize=10)

            # Draw body links if parents are specified
            if parents is None:
                continue

            parent = parents[i]
            parent_point = pose[parent]
            l_style = "-"
            if link_style is not None:
                l_style = f"{i}: {link_style[pose_id][i]}" if i < len(keypoint_names[pose_id]) else "-"
            # if it has a parent (not itself and not -1), connect it with a line
            if parent != i and parent != -1:
                ax.plot(
                    [point[0], parent_point[0]],
                    [point[1], parent_point[1]],
                    [point[2], parent_point[2]],
                    color=link_color,
                    linestyle=l_style,
                    linewidth=link_size,
                )

        plot_ranges_lst.append([(np.min(pose), np.max(pose))])

    plot_ranges_lst = np.array(plot_ranges_lst)
    range_min, range_max = (np.min(plot_ranges_lst), np.max(plot_ranges_lst))
    image_margin = (range_max - range_min) * (1 - image_margins_factor) / 2
    range_min -= image_margin
    range_max += image_margin

    if debug_axis_length is not None:
        axis = [[[0, debug_axis_length], [0, 0], [0, 0]], [[0, 0], [0, debug_axis_length], [0, 0]], [[0, 0], [0, 0], [0, debug_axis_length]]]
        axis_colors = ["red", "green", "blue"]
        for i in range(3):
            ax.plot(axis[i][0], axis[i][1], axis[i][2], color=axis_colors[i])
            ax.text(axis[i][0][1], axis[i][1][1], axis[i][2][1], f"{['X','Y','Z'][i]}", fontsize=10 * debug_axis_length)
            ax.scatter(axis[i][0][1], axis[i][1][1], axis[i][2][1], color=axis_colors[i], marker="o", s=10 * debug_axis_length)

    ax.axes.set_xlim3d(left=range_min, right=range_max)
    ax.axes.set_ylim3d(bottom=range_min, top=range_max)
    ax.axes.set_zlim3d(bottom=range_min, top=range_max)

    # Set the camera angle
    ax.view_init(elev=elevation, azim=azimuth)

    if not show_grid:
        plt.axis("off")
    if keypoint_names is not None:
        if legend_location is not None:
            plt.legend(loc=legend_location)
        else:
            plt.legend(loc="best")

    if save_svg_path is not None:
        plt.savefig(save_svg_path, format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.show()


def interactive_plot_body_keypoints(frame_joint_pos, parents, fig_opts=None, **kwargs):
    """
    Set parents to None in order to use the origin as a parent for all the points.
    """

    def plot_body_keypoints_(elevation_slider, azimuth_slider, view_axis_slider, view_joint_slider):
        return plot_body_keypoints(
            keypoint_pos_lst=frame_joint_pos,
            parents_lst=parents,
            fig_opts=fig_opts,
            elevation=elevation_slider,
            azimuth=azimuth_slider,
            show_grid=view_axis_slider,
            show_keypoint_id=view_joint_slider,
            **kwargs,
        )

    # Create the sliders and display them
    elevation_slider = widgets.FloatSlider(min=0, max=360, value=20, description="Elevation")
    azimuth_slider = widgets.FloatSlider(min=0, max=360, value=100, description="Azimuth")
    view_axis_slider = widgets.IntSlider(min=0, max=1, value=1, description="Axis grids")
    view_joint_slider = widgets.IntSlider(min=0, max=1, value=1, description="Joint IDs")

    widgets.interact(
        plot_body_keypoints_,
        elevation_slider=elevation_slider,
        azimuth_slider=azimuth_slider,
        view_axis_slider=view_axis_slider,
        view_joint_slider=view_joint_slider,
    )
