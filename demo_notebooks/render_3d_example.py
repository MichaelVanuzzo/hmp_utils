import time
import sys

import numpy as np
import torch

from hmp_utils import render_3d_smpl, MocapLoader, SmplConstants


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="SMPL renderer")
    parser.add_argument("--model_folder", default=None, type=str, help="The path to the model folder")
    parser.add_argument(
        "--model_type",
        default="smpl",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument("--gender", type=str, default="neutral", help="The gender of the model")
    parser.add_argument(
        "--num_betas",
        default=10,
        type=int,
        help="Number of shape coefficients.",
    )
    parser.add_argument(
        "--num_expression_coeffs",
        default=10,
        type=int,
        help="Number of expression coefficients.",
    )
    parser.add_argument(
        "--plotting_module",
        type=str,
        default="matplotlib",
        choices=["pyrender", "matplotlib"],
        help="The module to use for plotting the result",
    )
    parser.add_argument("--ext", type=str, default="npz", help="Which extension to use for loading")
    parser.add_argument(
        "--sample_shape",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random shape",
    )
    parser.add_argument(
        "--sample_expression",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random expression",
    )
    parser.add_argument(
        "--use_face_contour",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Compute the contour of the face",
    )
    parser.add_argument("--output_folder", default=None, type=str, help="Path of the output folder")
    parser.add_argument("--output_name", default=None, type=str, help="Name of the output file")
    parser.add_argument(
        "--framerate",
        default=25,
        type=int,
        help="Mocap framerate",
    )
    parser.add_argument("--multithread", default=True, type=bool, help="Use multithreading")
    parser.add_argument(
        "--processors",
        default=1,
        type=int,
        help="Number of processors to use for multithreading",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # overwrite default arguments
    args.model_folder = "/home/vanuzzo/hmp_utils/demo_notebooks/smpl_model/SMPLX_NEUTRAL.npz"
    args.output_name = f'render_{time.asctime().replace(":", "_").replace(" ", "_")}.mp4'
    args.processors = 96

    # joints_to_load = 22  # dont' load the hands
    # frame_to_load = slice(0, amass_data["poses"].shape[0], framerate_factor)
    # amass_smpl_poses = np.array(amass_data["poses"])[frame_to_load, 3 : joints_to_load * 3]
    # mocap_poses = np.array(amass_smpl_poses, dtype=np.float32)

    filepath = "./example_mocaps/walking_poses.npz"
    animation_framerate = 25
    max_len_frames = 500

    # -- load mocap --
    mocap_loader = MocapLoader(
        body_model_type=SmplConstants.BODY_MODEL_TYPE, keypoint_ids_to_load=SmplConstants.KEYPOINTS, target_framerate=animation_framerate
    )
    motion = mocap_loader.load_mocap(filepath)
    amass_data = motion.rots[:max_len_frames]

    joints_to_load = 22  # dont' load the hands
    print("amass_data.shape:", np.array(amass_data).shape)
    amass_smpl_poses = np.array(amass_data)[:, 1:22, :]
    print("amass_smpl_poses.shape:", amass_smpl_poses.shape)
    mocap_poses = torch.tensor(amass_smpl_poses, dtype=torch.float32)

    args.mocap_poses = mocap_poses

    start_time = time.time()
    render_3d_smpl(**args.__dict__)
    print(f"Total execution time: {time.time() - start_time:.3f} s")
