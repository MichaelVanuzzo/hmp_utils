import os
import numpy as np
from tqdm import trange
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum, auto

"""
#### GENERAZIONE DATI (solo per testing) ####

mocap_filepaths = ['/media/sata-SSD-250GB/home/frabors/AMASS_DATASET/BMLmovi/Subject_11_F_MoSh/Subject_11_F_10_poses.npz',
    '/media/sata-SSD-250GB/home/frabors/AMASS_DATASET/BMLmovi/Subject_11_F_MoSh/Subject_11_F_5_poses.npz']
target_framerate = 30
# SMPL model is based on 24 keypoints, the last 2 are the hands so we load the first 22 joints (0 to 21*3 inclusive)
joint_ids_to_load = np.arange(24)

forward_kine = ForwardKinematics(SMPL_OFFSETS['neutral'], SMPL_PARENTS)

mocaps_list = []
for filepath in mocap_filepaths:
    motion = mocap_processor.load_mocap(filepath, joint_ids_to_load, target_framerate)
    assert motion is not None

    mocap_positions = []
    for frame in trange(len(motion.rots)):
        mocap_positions.append(forward_kine.compute_fk(motion.rots[frame], apply_root_rotation=True))
    mocaps_list.append(mocap_positions)"""

#### VISUALIZZATORE ####
# INPUT: 
#   - lista di mocap in rappresentazione di posizioni assolute di tutti i keypoints
#   - lista di titoli da mettere negli eventuali subplots
#   - lista di ID del subplot. Se un mocap ha lo stesso ID di un altro significa che verranno 
#       sovrapposti nello stesso subplot

class AnimationMode(Enum):
    html = auto()
    plotlib = auto()
    save_gif = auto()

class CameraOrientation:
    def __init__(self, azimuth, elevation):
        self.azimuth = azimuth
        self.elevation = elevation

# NOTE: this needs to be replaced with something more elegant
DUMMY_MAX_INT = 100000000

def pose_animation(motions_list:list,
              parents,
              animation_mode:AnimationMode=AnimationMode.html,
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
    fig, *axes = plt.subplots(1, num_subplots, subplot_kw={'projection':'3d'}, **fig_opts)
    axes = axes[0]

    ax_texts = []
    for axid, ax in enumerate(axes):
        # title of subplot
        figure_title = subplot_titles[axid]
        ax.set_title(f'{figure_title}')
        # this is text to display other information during the animation
        ax_texts.append(ax.text2D(0.05, 0.05, '', transform=ax.transAxes, ha='left', va='bottom'))

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
            ax.view_init(elev=20, azim=100)
            # grid visibility
            if not show_grid:
                ax.axis('off')

    def update(frame):
        """
        function to update the figure for each frame of the animation
        """

        for motion_id, motion_positions in enumerate(motions_list):

            ax = axes[subplot_ids[motion_id]]
            ax_texts[subplot_ids[motion_id]].set_text(f'frame: {frame}/{len(motion_positions)} | time: {frame/animation_framerate:.2f} s')
            keypoint_positions = motion_positions[frame]

            # Plot the points
            for i, point in enumerate(keypoint_positions):

                # keypoint label
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
                    plot_lines[motion_id][i][0].set_data([[point[0], parent_point[0]], [point[1], parent_point[1]]])
                    plot_lines[motion_id][i][0].set_3d_properties([point[2], parent_point[2]])
                    plot_lines[motion_id][i][0].set_color('b')
                    # [point[2], parent_point[2]
    
    # apply plot options to each axis
    for ax_id, ax in enumerate(axes):
        ax.axes.set_xlim3d(left=camera_range[0], right=camera_range[1])
        ax.axes.set_ylim3d(bottom=camera_range[0], top=camera_range[1])
        ax.axes.set_zlim3d(bottom=camera_range[0], top=camera_range[1])
            
        # set camera orientations for each subplot
        elev = camera_orientations[ax_id].elevation
        azim = camera_orientations[ax_id].azimuth
        ax.view_init(elev=20, azim=100)
            
        # grid visibility
        if not show_grid:
            ax.axis('off')
        
    ani = FuncAnimation(fig, update, frames=min_len_frames, interval=1000/animation_framerate) #blit=False)
    plt.show()
