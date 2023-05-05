# Human Motion Prediction Utilities : _hmp_utils_

- [Contents](https://github.com/Human-Pose-Forecasting/hmp_utils/edit/main/README.md#contents)
  - [Metrics](https://github.com/Human-Pose-Forecasting/hmp_utils/edit/main/README.md#merics)
  - [Motion](https://github.com/Human-Pose-Forecasting/hmp_utils/edit/main/README.md#motion)
  - [Visualization](https://github.com/Human-Pose-Forecasting/hmp_utils/edit/main/README.md#visualization)
- [Installation](https://github.com/Human-Pose-Forecasting/hmp_utils/edit/main/README.md#package-install)

## Contents

### Merics
Code for evaluation of machine learning models using recognized metrics.

- **Euler angle error** :

- **Geodesic** (joint angle difference) : the angle needed to align each joint in the predictions to the targets. Credits to: [SPL](https://github.com/eth-ait/spl/blob/master/metrics/motion_metrics.py)

- **Positional** : 

### Motion
Code for loading, converting and processing mocap or general body pose data.

### Visualization

#### Stickman

Simple visualization tools for body pose as plots of keypoints connected by lines.
- **Static visualizer** : good for looking at a single frame of a body pose, jupyter notebooks
- **Animated visualizer** : inspired to the SPL visualizer ([code](https://github.com/eth-ait/spl))

## Package Install

#### Install in an apptainer container
Clone the repository in a directory, navigate into that directory and execute:
```
pip install . --user
```
```
pip install hmp_utils --user
```

#### Install from github
```
python -m pip install 'username @ git+https://github.com/...'
```
