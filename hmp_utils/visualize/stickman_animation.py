import logging
import numpy as np
from enum import Enum, auto
import matplotlib.pyplot as plt
from typing import Optional, Union
from IPython.display import display, HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

"""
#TODO:
- add automatic colors when multiple bodies are visualized in the same subplot
- add to the pose_animation function the ability to have a parents list for each
    body pose motion, since right now it uses the same parents for all motions
"""


class AnimationMode(Enum):
    HTML = auto()
    HTML5 = auto()
    PLOTLIB = auto()


class CameraOrientation:
    def __init__(self, azimuth, elevation):
        self.azimuth = azimuth
        self.elevation = elevation


def pose_animation(
    mocap_list: list,
    parents: list,
    animation_framerate: float,
    animation_mode: AnimationMode = AnimationMode.PLOTLIB,
    gif_savepath: Optional[str] = None,
    subplot_titles: Optional["list[str]"] = None,
    subplot_ids: Optional[list] = None,
    subplot_colors: Optional[list] = None,
    camera_orientations: Optional[Union[CameraOrientation, "list[CameraOrientation]"]] = None,
    show_grid: bool = False,
    show_keypoint_id: bool = False,
    fig_opts: Optional[dict] = None,
):
    """
        Animation of one or multiple body mocaps, represented as stickman (keypoints linked by lines).

        ---
        ### Args:
    - mocap_list (list): list of body keypoints positions that evolve with time. It must be a list
        so if a single body motion is needed just put it in a list like [body_positions]. Where
        body_position has a shape like (num_frames, num_keypoints, 3)
    - parents (list): list of parents for the body keypoints, needed to link the keypoints with lines
    - animation_framerate (float): the framerate at which the animation should be rendered
    - animation_mode (AnimationMode): chose between matplotlib visualizer and html (the latter being
        used in jupyter notebooks)
    - gif_savepath (str): path where to save the plotlib animation as gif
    - subplot_titles (list, optional): the titles for the subplots
    - subplot_ids (list, optional): specify in order in which subplot to place the body motion
        sequences
    - subplot_colors (list, optional): same as subplot_ids, but chose colors for body motions
    - camera_orientations (list[CameraOrientation]): specify one CameraOrientation for all subplots
        or specify an orientation for each subplot
    - show_grid (bool): whether to show the plotlib grid or not
    - show_keypoint_id (bool): whether to show the keypoint number adjacent to keypoint position in 3d
    - fig_opts (dict, optional): figure options as argument for the plt.figure() method
    """

    # number of subplot to show
    if subplot_ids is None:
        # show all mocaps in the same plot
        subplot_ids = [0 for _ in mocap_list]
    num_subplots = len(set(subplot_ids))

    if subplot_colors is None:
        # assign the same color to all the skeletons
        subplot_colors = ["b" for _ in mocap_list]

    # subplot titles
    if subplot_titles is None:
        subplot_titles = [f"Mocap {i}" for i in range(num_subplots)]

    # figure plot arguments
    if fig_opts is None:
        fig_opts = {"figsize": (8, 4), "dpi": 80}

    # camera orientation for each subplot
    if camera_orientations is None:
        # apply default camera orientation
        camera_orientations = [CameraOrientation(100, 20) for _ in subplot_ids]
    if isinstance(camera_orientations, CameraOrientation):
        # a single orientation was provided, apply to each subplot
        camera_orientations = [camera_orientations for _ in subplot_ids]
    if len(camera_orientations) < num_subplots:
        raise Exception(
            "Please provide a single camera orientation, or provide as many " f"camera orientations as there are subplots. Num subplots: {num_subplots}"
        )

    # find minimum duration of motions, that is where to stop the animation
    min_len_frames = min([len(mocap_pos) for mocap_pos in mocap_list])

    # automatically compute camera range
    # the camera range on the 3 axis must be the same in order to preserve proportions
    plot_ranges_lst = []
    for mocap_positions in mocap_list:
        plot_ranges_lst.append(np.min(mocap_positions))
        plot_ranges_lst.append(np.max(mocap_positions))
    camera_range = [np.min(plot_ranges_lst), np.max(plot_ranges_lst)]
    image_margins = (camera_range[1] - camera_range[0]) / 10
    camera_range = [camera_range[0] - image_margins, camera_range[1] + image_margins]

    # create figure
    fig, axes = plt.subplots(1, num_subplots, subplot_kw={"projection": "3d"}, **fig_opts)
    # white figure background
    fig.set_facecolor("white")

    if not isinstance(axes, np.ndarray):
        # single subplot
        axes = [axes]
    else:
        axes = axes.tolist()

    # this is text to display other information during the animation
    subplot_info_text_lst = []
    for ax in axes:
        subplot_info_text_lst.append(ax.text2D(0.05, 0.05, "", transform=ax.transAxes, ha="left", va="bottom"))

    # -- DEFINE LINES TO BE PLOTTED --
    # For efficiency, the lines are plotted one time and then updated during the animation
    # shape: (num_mocaps, num_keypoints)
    plot_keypoint = []
    plot_lines = []
    for mocap_id, mocap_positions in enumerate(mocap_list):
        ax_id = subplot_ids[mocap_id]
        ax = axes[ax_id]
        # le linee sono inizializzate nel primo frame del mocap
        keypoint_positions = mocap_positions[0]

        plot_keypoint.append([])
        plot_lines.append([])
        for i, keypoint_pos in enumerate(keypoint_positions):
            if show_keypoint_id:
                plot_keypoint[mocap_id].append(
                    ax.text(
                        keypoint_pos[0],
                        keypoint_pos[1],
                        keypoint_pos[2],
                        f"{i}",
                        fontsize=10,
                    )
                )

            parent = parents[i]
            parent_point = keypoint_positions[parent]
            # if it has a parent (not itself and not -1), connect it with a line
            if parent != -1:
                (line,) = ax.plot(
                    [keypoint_pos[0], parent_point[0]],
                    [keypoint_pos[1], parent_point[1]],
                    [keypoint_pos[2], parent_point[2]],
                    "-o",
                    markersize=2.0,
                )
                line.set_color(subplot_colors[mocap_id])
            else:
                line = None
            plot_lines[mocap_id].append(line)

        # Set camera orientations for each subplot
        elev = camera_orientations[ax_id].elevation
        azim = camera_orientations[ax_id].azimuth
        ax.view_init(elev=elev, azim=azim)
        # grid visibility
        if not show_grid:
            ax.axis("off")

    def update(frame):
        """
        function to update the figure for each frame of the animation
        """
        for mocap_id, mocap_positions in enumerate(mocap_list):
            subplot_info_text_lst[subplot_ids[mocap_id]].set_text(f"frame: {frame+1}/{len(mocap_positions)} | time: {(frame+1)/animation_framerate:.2f} s")
            keypoint_positions = mocap_positions[frame]

            # Plot the points
            for i, point in enumerate(keypoint_positions):
                if show_keypoint_id:
                    plot_keypoint[mocap_id][i].set_position((point[0], point[1], point[2]))

                parent = parents[i]
                parent_point = keypoint_positions[parent]
                # -- Update lines positions --
                # if it has a parent (not itself and not -1), connect it with a line
                if parent != -1:
                    line = plot_lines[mocap_id][i]
                    line.set_data([point[0], parent_point[0]], [point[1], parent_point[1]])
                    line.set_3d_properties([point[2], parent_point[2]])

    # apply plot options to each axis
    for ax_id, ax in enumerate(axes):
        # title of subplot
        figure_title = subplot_titles[ax_id]
        ax.set_title(f"{figure_title}")

        # ranges of visualization
        ax.axes.set_xlim3d(left=camera_range[0], right=camera_range[1])
        ax.axes.set_ylim3d(bottom=camera_range[0], top=camera_range[1])
        ax.axes.set_zlim3d(bottom=camera_range[0], top=camera_range[1])

    ani = FuncAnimation(fig, update, frames=min_len_frames, interval=1000 / animation_framerate)

    # save animation as gif
    if gif_savepath:
        ani.save(gif_savepath, writer="pillow", fps=30)

    if animation_mode == AnimationMode.PLOTLIB:
        plt.show()

    if animation_mode == AnimationMode.HTML:
        html_ani = ani.to_jshtml()
        display(HTML(html_ani))
        plt.close()

    if animation_mode == AnimationMode.HTML5:
        html_ani = ani.to_html5_video()
        display(HTML(html_ani))
        plt.close()
