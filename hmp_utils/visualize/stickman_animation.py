import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum, auto
import logging
from IPython.display import display, HTML


class AnimationMode(Enum):
    html = auto()
    html5 = auto()
    plotlib = auto()

class CameraOrientation:
    def __init__(self, azimuth, elevation):
        self.azimuth = azimuth
        self.elevation = elevation

# NOTE: this needs to be replaced with something more elegant
DUMMY_MAX_INT = 100000000

def pose_animation(motions_list:list,
              parents,
              animation_mode:AnimationMode=AnimationMode.plotlib,
              gif_savepath:Optional[str]=None,
              animation_framerate:float=30,
              subplot_titles:Optional['list[str]']=None,
              subplot_ids:Optional[list]=None,
              camera_orientations:Optional[Union[CameraOrientation, 'list[CameraOrientation]']]=None,
              show_grid:bool=False,
              show_keypoint_id:bool=False,
              fig_opts:Optional[dict]=None
              ):

    # number of subplot to show
    if subplot_ids is None:
        # show all mocaps in the same plot
        subplot_ids = [0 for _ in motions_list]
    num_subplots = len(set(subplot_ids))
    
    # subplot titles
    if subplot_titles is None:
        subplot_titles = [f'Motion {i}' for i in range(num_subplots)]

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
        raise Exception('Please provide a single camera orientation, or provide as many '
                        f'camera orientations as there are subplots. Num subplots: {num_subplots}')
    
    # NOTE: keypoint ID visualization is not yet fully implemented. The text elements should be initialized like
    # the body lines and their positions updated every frame for efficiency.
    if show_keypoint_id:
        logging.warning('Show keypoint ID is not yet working properly')
    
    # find minimum duration of motions, that is where to stop the animation
    min_len_frames = DUMMY_MAX_INT
    for mocap_pos in motions_list:
        min_len_frames = min(min_len_frames, len(mocap_pos))

    # -- -- -- -- -- -- -- -- -- --
    # NOTE: THIS NEEDS TO BE FIXED
    # automatically compute camera range
    # the camera range on the 3 axis must be the same in order to preserve proportions
    camera_range = [DUMMY_MAX_INT, -DUMMY_MAX_INT]
    for mocap_positions in motions_list:
        camera_range = [np.min((np.min(mocap_positions), camera_range[0])), 
                        np.max((np.max(mocap_positions), camera_range[1]))]
    camera_range = [int(i) for i in camera_range]
    # ! OVERRIDE CAMERA RANGE
    camera_range = [-1, 1]
    # -- -- -- -- -- -- -- -- -- --

    # create figure
    fig, axes = plt.subplots(1, num_subplots, subplot_kw={'projection':'3d'}, **fig_opts)
    # white figure background
    fig.set_facecolor('white')
    
    if not isinstance(axes, np.ndarray):
        # single subplot
        axes = [axes]
    if isinstance(axes, np.ndarray):
        axes = axes.tolist()

    # this is text to display other information during the animation
    subplot_info_text_lst = []
    for ax in axes:
        subplot_info_text_lst.append(ax.text2D(0.05, 0.05, '', transform=ax.transAxes, ha='left', va='bottom')) # type: ignore

    # -- DEFINE LINES TO BE PLOTTED --
    # For efficiency, the lines are plotted one time and then updated during the animation
    # shape: (num_mocaps, num_keypoints)
    plot_lines = []
    for mocap_id, mocap_positions in enumerate(motions_list):
            ax_id = subplot_ids[mocap_id]
            ax = axes[ax_id]
            # le linee sono inizializzate nel primo frame del mocap
            keypoint_positions = mocap_positions[0]

            plot_lines.append([])
            for i, keypoint_pos in enumerate(keypoint_positions):

                if parents is None:
                    parent = None
                    parent_point = [0, 0, 0]
                else:
                    parent = parents[i]
                    parent_point = keypoint_positions[parent]

                # if it has a parent (not itself and not -1), connect it with a line
                if parent != i and parent != -1:
                    line = ax.plot([keypoint_pos[0], parent_point[0]], [keypoint_pos[1], parent_point[1]],
                            [keypoint_pos[2], parent_point[2]], 'b-', markersize=2.0)
                    plot_lines[mocap_id].append(line)

            # Set camera orientations for each subplot
            elev = camera_orientations[ax_id].elevation
            azim = camera_orientations[ax_id].azimuth
            ax.view_init(elev=elev, azim=azim)
            # grid visibility
            if not show_grid:
                ax.axis('off')

    def update(frame):
        """
        function to update the figure for each frame of the animation
        """

        for motion_id, motion_positions in enumerate(motions_list):

            ax = axes[subplot_ids[motion_id]]
            subplot_info_text_lst[subplot_ids[motion_id]].set_text(f'frame: {frame}/{len(motion_positions)} | time: {frame/animation_framerate:.2f} s')
            keypoint_positions = motion_positions[frame]

            # Plot the points
            for i, point in enumerate(keypoint_positions):

                # keypoint label - BUG: not working properly
                if show_keypoint_id:
                    ax.text(*point, f'{i}', fontsize=10)

                if parents is None:
                    parent = None
                    parent_point = [0, 0, 0]
                else:
                    parent = parents[i]
                    parent_point = keypoint_positions[parent]

                # -- Update lines positions --
                # if it has a parent (not itself and not -1), connect it with a line
                if parent != i and parent != -1:
                    if i == 23:  # ?? sistemare
                        continue
                    plot_lines[motion_id][i][0].set_data(np.array([[point[0], parent_point[0]], [point[1], parent_point[1]]]))
                    plot_lines[motion_id][i][0].set_3d_properties(np.array([point[2], parent_point[2]]))
                    plot_lines[motion_id][i][0].set_color('b')
    
    # apply plot options to each axis
    for ax_id, ax in enumerate(axes):
            # title of subplot
        figure_title = subplot_titles[ax_id]
        ax.set_title(f'{figure_title}')
        
        # ranges of visualization
        ax.axes.set_xlim3d(left=camera_range[0], right=camera_range[1]) # type: ignore
        ax.axes.set_ylim3d(bottom=camera_range[0], top=camera_range[1]) # type: ignore
        ax.axes.set_zlim3d(bottom=camera_range[0], top=camera_range[1]) # type: ignore
            
        # set camera orientations for each subplot
        elev = camera_orientations[ax_id].elevation
        azim = camera_orientations[ax_id].azimuth
        ax.view_init(elev=elev, azim=azim) # type: ignore
            
        # grid visibility
        if not show_grid:
            ax.axis('off')

    ani = FuncAnimation(fig, update, frames=min_len_frames, interval=1000/animation_framerate)

    # save animation as gif
    if gif_savepath:
        ani.save(gif_savepath, writer='pillow', fps=30)

    if animation_mode == AnimationMode.plotlib:
        plt.show()
        
    if animation_mode == AnimationMode.html:
        # html_ani = ani.to_html5_video()
        html_ani = ani.to_jshtml()
        display(HTML(html_ani))
        plt.close()

    if animation_mode == AnimationMode.html5:
        html_ani = ani.to_html5_video()
        display(HTML(html_ani))
        plt.close()
        