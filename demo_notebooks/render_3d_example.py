import time
import sys

# add hmp_utils to path
hmp_path = "../"
if hmp_path not in sys.path:
    sys.path.append(hmp_path)
from hmp_utils import render_3d_smpl


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="SMPL renderer")
    parser.add_argument(
        "--model_folder", default=None, type=str, help="The path to the model folder"
    )
    parser.add_argument(
        "--model_type",
        default="smpl",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument(
        "--gender", type=str, default="neutral", help="The gender of the model"
    )
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
    parser.add_argument(
        "--ext", type=str, default="npz", help="Which extension to use for loading"
    )
    parser.add_argument(
        "--plot_joints",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="The path to the model folder",
    )
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
    parser.add_argument(
        "--gui_viewport", default=False, type=bool, help="Use the GUI viewport"
    )
    parser.add_argument(
        "--mocap_file", default=None, type=str, help="Path of the mocap file to load"
    )
    parser.add_argument(
        "--output_folder", default=None, type=str, help="Path of the output folder"
    )
    parser.add_argument(
        "--output_name", default=None, type=str, help="Name of the output file"
    )
    parser.add_argument(
        "--framerate_factor",
        default=1,
        type=int,
        help="Redude the framerate by this factor",
    )
    parser.add_argument(
        "--multithread", default=True, type=bool, help="Use multithreading"
    )
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
    args.model_folder = "/home/dtg2958/repository/renderer/SMPLX_NEUTRAL.npz"
    args.mocap_file = (
        "/home/dtg2958/repository/xsens_converter/converted_dataset/davichael/test3.npz"
    )
    args.output_folder = "/home/dtg2958/repository/renderer/rendered_video/"
    args.output_name = (
        f'render_{time.asctime().replace(":", "_").replace(" ", "_")}.mp4'
    )
    args.framerate_factor = 2
    args.processors = 96

    start_time = time.time()
    render_3d_smpl(**args.__dict__)
    print(f"Total execution time: {time.time() - start_time:.3f} s")
