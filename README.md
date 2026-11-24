# Human Motion Prediction Utilities (`hmp_utils`)

A Python utility library for Human Motion Prediction providing tools for motion loading/processing, kinematics, evaluation metrics, and 2D/3D pose visualization.

## Overview

- **`hmp_utils.common`**: Rotation conversions (`RotConverter`), TF helpers (`TfUtils`), table utilities (`TableManager`), logging (`custom_logger`), and OS helpers.
- **`hmp_utils.motion`**: Skeleton constants (`SmplConstants`, `H36MConstants`), forward/inverse kinematics (`ForwardKinematics`, `InverseKinematics`), mocap loading, downsampling, and AMASS-to-H3.6M conversion (`MocapLoader`, `Amass2HumanConverter`).
- **`hmp_utils.metrics`**: Pose prediction metrics (`MetricsProcessor`, `MetricsPostProcessor`, `SpectrumMetricsProcessor`), evaluation protocols, and AMASS dataset splitting utilities.
- **`hmp_utils.visualize`**: Static plots (`plot_body_keypoints`, `plot_static_seq`), stickman animations (`pose_animation`), and SMPL 3D rendering (`render_3d_smpl`).

See `demo_notebooks/` for usage examples.

## Installation

Using a dedicated Python virtual environment (`venv` or `conda`) is recommended.

### From GitHub

```bash
pip install git+https://github.com/MichaelVanuzzo/hmp_utils.git
```

### From Local Source (for development or running `demo_notebooks/`)

```bash
git clone https://github.com/MichaelVanuzzo/hmp_utils.git
cd hmp_utils

# Optional (recommended): create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -e .

# Optional: register the environment as a Jupyter kernel to run demo_notebooks/
pip install ipykernel
python -m ipykernel install --user --name hmp-utils --display-name "Python (hmp_utils)"
```

## Demo Notebooks & Scripts

Examples are provided in `demo_notebooks/`:

- **Visualization**: `static_visualization_demo.ipynb`, `animated_visualization_demo.ipynb`, `static_seq_demo.ipynb`, `animated_visualization_plotlib_demo.py`, `render_3d_example.py`
- **Kinematics & Conversion**: `forward_kinematics_demo.ipynb`, `amass_to_h36m_example.ipynb`, `amass_to_h36m_dataset.py`
- **Dataset Splitting & Spectral Analysis**: `split_amass_dataset_demo.ipynb`, `amass_PSD.ipynb`, `amass2h36m_PSD.ipynb`, `h36m_PSD.ipynb`
- **Utilities**: `downsampling_test.ipynb`, `custom_logger.ipynb`, `table_demo.ipynb`

## External Resources & Configuration

Third-party body models and motion datasets are not bundled with this repository due to their licensing terms. Before running the demo scripts or notebooks that use them, download the required assets from their official sources and update the placeholder/example paths:

- **SMPL-X / SMPL Body Models** (`SMPLX_NEUTRAL.npz`): register and download from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) (or [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/)), then update `args.model_folder` in `demo_notebooks/render_3d_example.py`.
- **AMASS Dataset**: register and download from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/), then update `amass_path` / `save_dir` in `demo_notebooks/amass_to_h36m_dataset.py` (and the dataset paths inside `demo_notebooks/*.ipynb`). *(Small test clips in `demo_notebooks/example_mocaps/` are from the public AMASS GitHub sample data.)*
- **Human3.6M Dataset**: request access from [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) if running H3.6M evaluation or PSD scripts.

## Contributors & Acknowledgements

This library originated as a collaborative effort at the Department of Management and Engineering (DTG), University of Padova. Main contributors:

- Michael Vanuzzo
- Marco Casarin
- Francesco Borsatti
- Davide Carollo
- Stefano Michieletto

## License

This project is licensed under the GNU Affero General Public License v3.0 ([LICENSE](LICENSE)).