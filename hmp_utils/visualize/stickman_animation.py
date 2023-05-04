import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from hmp_utils.motion import mocap_processor
from hmp_utils.motion.kinematics import ForwardKinematics
from hmp_utils.motion.body_model_data import SMPL_OFFSETS, SMPL_PARENTS

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
    mocaps_list.append(mocap_positions)


#### VISUALIZZATORE ####
# INPUT: 
#   - lista di mocap in rappresentazione di posizioni assolute di tutti i keypoints
#   - lista di titoli da mettere negli eventuali subplots
#   - lista di ID del subplot. Se un mocap ha lo stesso ID di un altro significa che verranno 
#       sovrapposti nello stesso subplot

# -- PARAMETRI DEL PLOT --
view_joint_id = False
plot_joint_points = False
parents = SMPL_PARENTS

subplot_ids = [0, 1]
figure_titles = ['mocap 1', 'mocap 2']

# find minimum length of mocaps
min_len_frames = 100000000
for mocap_pos in mocaps_list:
    min_len_frames = min(min_len_frames, len(mocap_pos))

# find camera range
camera_range = [100000000, -100000000]
for mocap_positions in mocaps_list:
    camera_range = [np.min((np.min(mocap_positions), camera_range[0])), 
                    np.max((np.max(mocap_positions), camera_range[1]))]
camera_range = [int(i) for i in camera_range]
camera_range = [-1, 1]

num_subplots = len(set(subplot_ids))

# create figure
fig_opts = {"figsize": (10, 10), "dpi": 100}
fig, *axes = plt.subplots(1, num_subplots, subplot_kw={'projection':'3d'})
axes = axes[0]

fps_visualizer_text = fig.text(0.05, 0.05, '', va='bottom')

ax_texts = []
for axid, ax in enumerate(axes):
    figure_title = figure_titles[axid]
    ax.set_title(f'MOCAP {os.path.basename(figure_title)}')
    ax_texts.append(ax.text2D(0.05, 0.05, '', transform=ax.transAxes, ha='left', va='bottom'))

# shape: (num_mocaps, num_keypoints)
plot_lines = []
for mocap_id, mocap_positions in enumerate(mocaps_list):

        ax = axes[subplot_ids[mocap_id]]
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
                        [keypoint_pos[2], parent_point[2]], 'b-')
                plot_lines[mocap_id].append(line)

        # Set the camera angle
        ax.view_init(elev=20, azim=100)
        ax.axis('off')

# define function to update the figure for each frame of the animation
def update(frame):
    
    # frame_start_time = monotonic_ns()
    # to_clear = [True for _ in range(num_subplots)]

    for mocap_id, mocap_positions in enumerate(mocaps_list):

        ax = axes[subplot_ids[mocap_id]]
        ax_texts[subplot_ids[mocap_id]].set_text(f'frame: {frame}/{len(mocap_positions)} | time: {frame/target_framerate:.2f} s')
        keypoint_positions = mocap_positions[frame]

        # Plot the points
        for i, point in enumerate(keypoint_positions):

            # plot keypoint positions
            if plot_joint_points:
                ax.scatter(point[0], point[1], point[2], c='r', marker='o')

            # point label
            if view_joint_id:
                ax.text(*point, f'{i}', fontsize=10)

            if parents is None:
                parent = None
                parent_point = [0, 0, 0]
            else:
                parent = parents[i]
                parent_point = keypoint_positions[parent]

            # -- Update
            # if it has a parent (not itself and not -1), connect it with a line
            if parent != i and parent != -1:
                if i == 23:
                    continue
                plot_lines[mocap_id][i][0].set_data([[point[0], parent_point[0]], [point[1], parent_point[1]]])
                plot_lines[mocap_id][i][0].set_3d_properties([point[2], parent_point[2]])
                plot_lines[mocap_id][i][0].set_color('b')
                # [point[2], parent_point[2]
        
for ax in axes:
    ax.axes.set_xlim3d(left=camera_range[0], right=camera_range[1])
    ax.axes.set_ylim3d(bottom=camera_range[0], top=camera_range[1])
    ax.axes.set_zlim3d(bottom=camera_range[0], top=camera_range[1])
    ax.view_init(elev=20, azim=100)
    ax.axis('off')

ani = FuncAnimation(fig, update, frames=min_len_frames, interval=1000/target_framerate) #blit=False)
plt.show()
