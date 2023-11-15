import os
import sys
from glob import glob
from tqdm import tqdm

if "../" not in sys.path:
    sys.path.append("../")
from hmp_utils import (
    MocapLoader,
    Amass2HumanConverter,
    SmplConstants,
    save_h36m_mocap_np2txt,
    AMASS_DATASETS,
)

# General variables
target_framerate = 50
mocap_loader = MocapLoader(
    body_model_type=SmplConstants.BODY_MODEL_TYPE,
    keypoint_ids_to_load=SmplConstants.KEYPOINTS,
    target_framerate=target_framerate,
)
converter = Amass2HumanConverter()
amass_path = "/home/vanuzzo/datasets/AMASS_H"
save_dir = f"/home/vanuzzo/datasets/Amass_H2Human_{target_framerate}fps"
datasets = AMASS_DATASETS


def is_valid_mocap(mocap_str: str) -> bool:
    return (
        mocap_str.endswith(".npz")
        and "stagei.npz" not in mocap_str
        and "shape.npz" not in mocap_str
    )


print("Start of conversion", flush=True)
for it, dataset in enumerate(datasets):
    pathname = os.path.join(amass_path, dataset, "**", "*.npz")
    mocap_paths_list = glob(pathname, recursive=True)
    # Iterate over all the dataset
    for action_path in tqdm(
        mocap_paths_list, desc=f"Converting {dataset} ({it+1}/{len(datasets)})"
    ):
        sys.stdout.flush()
        if not is_valid_mocap(action_path):
            continue
        # Load Amass mocap
        mocap = mocap_loader.load_mocap(action_path)
        if mocap is None:
            print(f"Error loading {action_path}")
            continue
        # Convert Amass mocap to Human3.6m
        converted_mocap = converter.amass2h36m_mocap(mocap.rots, mocap.trans)
        # Save to txt
        relative_path = action_path.split(amass_path)[-1]
        save_path = os.path.join(save_dir, relative_path)
        save_dir_local = os.path.dirname(save_path)

        basename = os.path.basename(action_path)
        name, _ = os.path.splitext(basename)
        save_h36m_mocap_np2txt(
            mocap=converted_mocap,
            dir_path=save_dir + save_dir_local,
            filename=name,
        )
print("\nEnd of conversion", flush=True)
