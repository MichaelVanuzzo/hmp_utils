import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from typing import Optional


def plot_body_keypoints(keypoints_pos: list,
                        parents: Optional[list],
                        fig_opts=None,
                        keypoint_names:Optional[list]=None,
                        elevation=20,
                        azimuth=100,
                        show_grid: bool = False,
                        show_keypoint_id: bool = True,
                        keypoint_marker='o',
                        keypoint_color='r'
                        ) -> None:
    """
    Scatter plot of body keypoints positions in 3d, connected if a parents list is provided

    ---
    ### Args:
    - keypoints_pos (list): list of keypoint positions of shape (num_keypoints, 3)
    - parents (list, optional): list of parents, in order to draw connecting links
    - fig_opts (dict()): options for the pyplot.figure() call
    - keypoint_names (list, optional): list of keypoint names to print
    - elevation (int, optional): elevation angle of camera view. Defaults to 20.
    - azimuth (int, optional): angle of camera view. Defaults to 100.
    - show_grid (bool, optional): whether to show grid. Defaults to False.
    - show_keypoint_id (bool, optional): whether to show keypoint ID. Defaults to True.
    - keypoint_marker (str, optional): type of marker for the keypoints. Defaults to 'o'.
    - keypoint_color (str, optional): color of the marker. Defaults to 'r'.
    """

    if fig_opts is None:
        fig_opts = {"figsize": (10, 10), "dpi": 80}

    fig = plt.figure(**fig_opts)
    ax = fig.add_subplot(111, projection='3d')

    for i, point in enumerate(keypoints_pos):
        # Plot the keypoint location
        keyp_name = ''
        if keypoint_names is not None:
            keyp_name = f'{i}: {keypoint_names[i]}' if i < len(keypoint_names) else ''
        ax.scatter(point[0], point[1], point[2], c=keypoint_color, marker=keypoint_marker,
                   label=keyp_name)

        # Keypoint label
        if show_keypoint_id:
            ax.text(point[0], point[1], point[2], f'{i}', fontsize=10)

        # Draw body links if parents are specified
        if parents is None:
            continue

        parent = parents[i]
        parent_point = keypoints_pos[parent]

        # if it has a parent (not itself and not -1), connect it with a line
        if parent != i and parent != -1:
            ax.plot([point[0], parent_point[0]], [point[1], parent_point[1]],
                    [point[2], parent_point[2]], 'b-')

    range_min, range_max = (np.min(keypoints_pos), np.max(keypoints_pos))
    ax.axes.set_xlim3d(left=range_min, right=range_max) # type: ignore
    ax.axes.set_ylim3d(bottom=range_min, top=range_max) # type: ignore
    ax.axes.set_zlim3d(bottom=range_min, top=range_max) # type: ignore

    # Set the camera angle
    ax.view_init(elev=elevation, azim=azimuth)

    if not show_grid:
        plt.axis('off')
    if keypoint_names is not None:
        plt.legend()
    plt.show()


def interactive_plot_body_keypoints(frame_joint_pos, parents, fig_opts=None, **kwargs):
    """
    Set parents to None in order to use the origin as a parent for all the points.
    """

    def plot_body_keypoints_(elevation_slider, azimuth_slider, view_axis_slider, view_joint_slider):
        return plot_body_keypoints(keypoints_pos=frame_joint_pos, parents=parents, fig_opts=fig_opts,
                                   elevation=elevation_slider, azimuth=azimuth_slider,
                                   show_grid=view_axis_slider, show_keypoint_id=view_joint_slider,
                                   **kwargs)

    # Create the sliders and display them
    elevation_slider = widgets.FloatSlider(min=-90, max=90, value=20, description='Elevation')
    azimuth_slider = widgets.FloatSlider(min=-180, max=180, value=100, description='Azimuth')
    view_axis_slider = widgets.IntSlider(min=0, max=1, value=1, description='Axis grids')
    view_joint_slider = widgets.IntSlider(min=0, max=1, value=1, description='Joint IDs')

    widgets.interact(plot_body_keypoints_, elevation_slider=elevation_slider, azimuth_slider=azimuth_slider,
                     view_axis_slider=view_axis_slider, view_joint_slider=view_joint_slider)
