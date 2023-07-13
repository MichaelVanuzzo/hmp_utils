import os
import sys
from glob import glob
from tqdm import tqdm

if "../" not in sys.path:
    sys.path.append("../")
from hmp_utils import (
    MocapLoader,
    Amass2HumanConvert,
    SmplConstants,
    h36m_np_to_txt,
    AMASS_DATASETS,
)

# General variables
target_framerate = 50
mocap_loader = MocapLoader(
    body_model_type=SmplConstants.BODY_MODEL_TYPE,
    keypoint_ids_to_load=SmplConstants.KEYPOINTS,
    target_framerate=target_framerate,
)
converter = Amass2HumanConvert()
amass_path = "/home/devlab/datasets/AMASS-SMPL-X/"
save_dir = f"/home/devlab/datasets/AmassX_to_Human_{target_framerate}fps/"
datasets = AMASS_DATASETS
# don't load these mocaps
exceptions = [
    "/home/devlab/datasets/AMASS-SMPL-X/EKUT/265/MTR03_stageii.npz",  # bugged
]

print("Start of conversion", flush=True)
for it, dataset in enumerate(datasets):
    pathname = os.path.join(amass_path, dataset, "**", "*.npz")
    mocap_paths_list = glob(pathname, recursive=True)
    # Iterate over all the dataset
    for action_path in tqdm(
        mocap_paths_list, desc=f"Converting {dataset} ({it+1}/{len(datasets)})"
    ):
        sys.stdout.flush()
        if action_path in exceptions or action_path.endswith("ei.npz"):
            continue
        # Load Amass mocap
        mocap = mocap_loader.load_mocap(action_path)
        assert mocap is not None
        # Convert Amass mocap to Human3.6m
        converted_mocap = converter.amass_to_h36m_mocap(mocap.rots, mocap.trans)
        # Save to txt
        relative_path = action_path.split(amass_path)[-1]
        save_path = os.path.join(save_dir, relative_path)
        save_dir_local = os.path.dirname(save_path)

        basename = os.path.basename(action_path)
        name, _ = os.path.splitext(basename)
        h36m_np_to_txt(motion=converted_mocap, dir_path=save_dir_local, filename=name)
print("\nEnd of conversion", flush=True)
