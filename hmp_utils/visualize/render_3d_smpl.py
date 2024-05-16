# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import time
import torch
import smplx
import numpy as np
import os.path as osp
from tqdm import trange
from typing import Optional
from PIL import Image, ImageOps
import io
from multiprocessing import Pool
import imageio


# render with pyrender
def img_from_pyrender(vertices, vertex_colors, model_faces, width=800, height=800):
    import pyrender
    import trimesh

    vertex_colors = torch.ones([vertices.shape[0], 4]) * vertex_colors

    tri_mesh = trimesh.Trimesh(vertices, model_faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    #### 2D RENDER ####
    # -- CAMERA --
    # camera position
    yfov = np.pi / 3.0
    # Get the bounds of the mesh
    min_bounds, max_bounds = mesh.bounds
    min_bounds, max_bounds = torch.tensor(min_bounds), torch.tensor(max_bounds)
    # Compute the sphere that encompasses the mesh
    center = (min_bounds + max_bounds) / 2.0
    radius = torch.norm(max_bounds - center)
    # Calculate the optimal camera position
    distance = radius / torch.sin(torch.tensor(yfov / 2))  # yfov in our example is pi/3
    camera_pose = torch.eye(4)
    camera_pose[:3, 3] = center + torch.tensor([0, 0, distance])

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)

    # -- LIGHTS --
    spotlight = pyrender.SpotLight(color=[150, 150, 150], intensity=5)
    scene.add(spotlight)

    #### RENDERING ####

    # solid body
    renderer = pyrender.OffscreenRenderer(width, height)
    render_flags = pyrender.RenderFlags.FACE_NORMALS | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.RGBA
    # pyrender.RenderFlags.ALL_SOLID
    # pyrender.RenderFlags.ALL_WIREFRAME
    render_output = renderer.render(scene, flags=render_flags)

    assert render_output is not None
    color, _ = render_output

    # Convert the color buffer to a PIL image and save it to disk
    # RGBA = Red Green Blue Alpha (where alpha is transparancy)
    img1 = Image.fromarray(color)

    # wireframe
    render_flags = pyrender.RenderFlags.FACE_NORMALS | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_WIREFRAME
    # pyrender.RenderFlags.ALL_SOLID
    render_output = renderer.render(scene, flags=render_flags)

    assert render_output is not None
    color, _ = render_output

    img2 = Image.fromarray(color)

    # invert wireframe to make the image less dark
    img2 = img2.convert("RGB")
    img2 = ImageOps.invert(img2)

    img = Image.composite(img1, img2, mask=img1)

    # img.save(image_save_path)
    img = np.array(img)
    return img


# render with matplotlib
def img_from_matplotlib(vertices, model_faces, elev=90, azim=-90, figsize=(8, 8), lim3D=(-1, 1)):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("equal")
    ax.axis("off")
    ax.view_init(elev=elev, azim=azim)
    ax.axes.set_xlim3d(-1, 1)
    ax.axes.set_ylim3d(-1, 1)
    ax.axes.set_zlim3d(-1, 1)

    mesh = Poly3DCollection(vertices[model_faces], alpha=0.3)
    edge_color = (0.1, 0.1, 0.1)
    mesh.set_edgecolor(edge_color)
    face_color = (0.9, 0.9, 0.9)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    plt.close(fig)
    img = Image.open(img_buf)
    img = np.array(img)
    return img


# get the vertices and joints from the model for a single frame
def get_vertices_and_joints(pose, betas, expression, model):
    # Get the pose from the AMASS file
    pose = torch.clone(pose)
    # add dimension in position 0
    pose = torch.unsqueeze(pose, 0)
    #### SMPLX MODEL OUTPUT ####
    # combine all parameters and generate output
    output = model(
        betas=betas,
        expression=expression,
        global_orient=None,
        body_pose=pose,
        left_hand_pose=None,
        right_hand_pose=None,
        return_verts=True,
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    return vertices, joints


# process a single frame with the selected plotting module
def process_frame(
    vertices,
    vertex_colors,
    model_faces,
    plotting_module,
    print_progress=False,
):
    if plotting_module == "pyrender":
        img = img_from_pyrender(
            vertices=vertices,
            vertex_colors=vertex_colors,
            model_faces=model_faces,
        )
    elif plotting_module == "matplotlib":
        img = img_from_matplotlib(
            vertices=vertices,
            model_faces=model_faces,
        )
    else:
        raise Exception(f"Plotting module {plotting_module} not supported.")
    if print_progress:
        print(".", end="", flush=True)
    return img


# wrapper function to use more than one argument in multiprocessing
def multi_run_wrapper(args):
    return process_frame(**args)


# run processes in parallel
def run_multiprocessing(processors, arg_list):
    with Pool(processes=processors) as pool:
        return pool.map(multi_run_wrapper, arg_list)


def render_3d_smpl(
    mocap_poses,  #
    model_type="smpl",
    model_folder: Optional[str] = None,
    output_folder: Optional[str] = None,
    output_name: Optional[str] = None,
    plotting_module="matplotlib",
    framerate: int = 25,
    multithread: bool = True,
    processors: int = 1,
    gender="neutral",
    use_face_contour=False,
    num_betas=10,
    num_expression_coeffs=10,
    ext="npz",
    sample_shape=False,
    sample_expression=False,
):
    #### CREATE MODEL ####
    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext=ext,
    )

    #### BODY SHAPE ####
    if sample_shape:
        # random
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
        print("betas:\n", betas)
    else:
        # default betas: betas determine the "thickness" of the body shape
        betas = torch.tensor(
            [
                [
                    0.5371,
                    -0.7166,
                    -0.1071,
                    -0.1103,
                    1.3785,
                    -0.8452,
                    -1.8501,
                    0.5672,
                    -0.7708,
                    -1.3519,
                ]
            ]
        )

    #### FACE EXPRESSION ####
    if sample_expression:
        # random
        expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)
    else:
        expression = None

    vertex_colors = torch.tensor([0.1, 0.1, 0.1, 0.7])

    #### PROCESSING ####
    total_frames = mocap_poses.shape[0]
    print(f"Total frames to process: {total_frames}")
    if not multithread:
        # single thread version
        video_frames = []
        for frame_idx in trange(total_frames):
            vertices, joints = get_vertices_and_joints(mocap_poses[frame_idx], betas, expression, model)
            img = process_frame(
                vertices=vertices,
                vertex_colors=vertex_colors,
                model_faces=model.faces,
                plotting_module=plotting_module,
            )
            video_frames.append(img)
    else:
        # new multithread version
        arg_list = []
        for frame in range(total_frames):
            vertices, joints = get_vertices_and_joints(mocap_poses[frame], betas, expression, model)
            # arguments for process_frame function
            arg_list.append(
                {
                    "vertices": vertices,
                    "vertex_colors": vertex_colors,
                    "model_faces": model.faces,
                    "plotting_module": plotting_module,
                    "print_progress": True,
                }
            )
        print("Processing", end="", flush=True)
        start_time = time.time()
        video_frames = run_multiprocessing(processors=processors, arg_list=arg_list)
        print(f"\nProcessing completed in {time.time() - start_time:.3f} s")

    # create folder if missing
    if output_folder is None:
        output_folder = "./render/"
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # create output name if missing
    if output_name is None:
        output_name = f'render_{time.asctime().replace(":", "_").replace(" ", "_")}.mp4'

    output_file = os.path.join(output_folder, output_name)

    # Write the images to an mp4 video file
    print("Saving video", end=" ", flush=True)
    start_time = time.time()
    with imageio.get_writer(output_file, fps=framerate) as writer:
        for image in video_frames:
            writer.append_data(image)
    print(f"completed in {time.time() - start_time:.3f} s")
