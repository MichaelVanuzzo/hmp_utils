import os
import sys
from glob import glob
from tqdm import tqdm

if "../" not in sys.path:
    sys.path.append("../")
from hmp_utils import MocapLoader, Amass2HumanConvert, SmplConstants, h36m_np_to_txt

# General variables
mocap_loader=MocapLoader(body_model_type=SmplConstants.BODY_MODEL_TYPE,
                         keypoint_ids_to_load=SmplConstants.KEYPOINTS,
                         target_framerate=25)
converter=Amass2HumanConvert()
amass_path = "/home/devlab/datasets/AMASS/"
save_dir = "/home/devlab/datasets/Amass_to_Human_25fps/"
datasets = ["ACCAD_copy","BMLrub"]

for dataset in datasets:
    pathname = os.path.join(amass_path,dataset, "**", "*.npz")
    mocap_paths_list  = glob(pathname, recursive=True)
    # Iterate over all the dataset
    for action_path in tqdm(mocap_paths_list,desc=f"Converting {dataset} to h36m format"):
        # Load Amass mocap
        mocap = mocap_loader.load_mocap(action_path)
        assert mocap is not None
        # Convert Amass mocap to Human3.6m
        converted_mocap = converter.amass_to_h36m_mocap(mocap.rots,mocap.trans)
        # Save to txt
        relative_path = action_path.split(amass_path)[-1]
        save_path = os.path.join(save_dir, relative_path)
        save_dir_local=os.path.dirname(save_path)

        basename = os.path.basename(action_path)
        name, _ = os.path.splitext(basename)
        h36m_np_to_txt(motion=converted_mocap,
                    dir_path=save_dir_local,
                    filename=name)