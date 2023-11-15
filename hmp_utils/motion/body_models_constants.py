from enum import Enum, auto


class BodyModelType(Enum):
    SMPL = auto()
    H36M = auto()


class BodyConstants:
    """General body constants class"""

    NUM_KEYPOINTS: int
    KEYPOINTS: "list[int]"  # TODO: remove this, it's pointless
    PARENTS: "list[int, int]"
    CHILDS: "list[int, int]"  # TODO: remove this, seems wrong... if multiple childs, which one is the right one?
    # JOINTS = "list[int]"
    # END_EFFECTORS: "list[int]"
    MAJOR_JOINTS: "list[int]"
    JOINTS_TO_IGNORE: "list[int]"
    JOINT_NAMES: "list[int, str]"
    OFFSETS: list
    OFFSETS_ARE_RELATIVE: bool
    LEFT_MUL: bool
    BODY_MODEL_TYPE: BodyModelType


class SmplConstants(BodyConstants):
    # ---------------SMPL---------------
    # Keypoints are all the joints and end effectors combined
    NUM_KEYPOINTS = 24
    KEYPOINTS = [i for i in range(NUM_KEYPOINTS)]  # TODO: remove this, it's pointless

    # Parents of all SMPL keypoints
    # From https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/visualization/fk.py#L33
    PARENTS_ = {
        0: -1,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
        9: 6,
        10: 7,
        11: 8,
        12: 9,
        13: 9,
        14: 9,
        15: 12,
        16: 13,
        17: 14,
        18: 16,
        19: 17,
        20: 18,
        21: 19,
        22: 20,
        23: 21,
    }
    PARENTS = list(PARENTS_.values())

    # Used to find the child of a joint, -1 if there is no child and for the root
    CHILDS_ = {
        0: -1,
        1: 4,
        2: 5,
        3: 6,
        4: 7,
        5: 8,
        6: 9,
        7: 10,
        8: 11,
        9: 12,
        10: -1,
        11: -1,
        12: 15,
        13: 16,
        14: 17,
        15: -1,
        16: 18,
        17: 19,
        18: 20,
        19: 21,
        20: 22,
        21: 23,
        22: -1,
        23: -1,
    }  # TODO: remove this, seems wrong... if multiple childs, which one is the right one?
    CHILDS = list(CHILDS_.values())

    # Keypoints called joints are those which perform rotation relative to their parent
    # JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    # Keypoints that are at the end of the kinematics tree
    # END_EFFECTORS = [10, 11, 15, 22, 23]

    # Subset of keypoints used in metrics
    MAJOR_JOINTS = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]

    # Subset of keypoints that are not used in metrics
    JOINTS_TO_IGNORE = [10, 11, 22, 23]

    # From https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md
    SMPLX_JOINT_NAMES = {
        0: "pelvis",
        1: "left_hip",
        2: "right_hip",
        3: "spine1",
        4: "left_knee",
        5: "right_knee",
        6: "spine2",
        7: "left_ankle",
        8: "right_ankle",
        9: "spine3",
        10: "left_foot",
        11: "right_foot",
        12: "neck",
        13: "left_collar",
        14: "right_collar",
        15: "head",
        16: "left_shoulder",
        17: "right_shoulder",
        18: "left_elbow",
        19: "right_elbow",
        20: "left_wrist",
        21: "right_wrist",
        22: "jaw",
        23: "left_eye",
        24: "right_eye",
        25: "left_index1",
        26: "left_index2",
        27: "left_index3",
        28: "left_middle1",
        29: "left_middle2",
        30: "left_middle3",
        31: "left_pinky1",
        32: "left_pinky2",
        33: "left_pinky3",
        34: "left_ring1",
        35: "left_ring2",
        36: "left_ring3",
        37: "left_thumb1",
        38: "left_thumb2",
        39: "left_thumb3",
        40: "right_index1",
        41: "right_index2",
        42: "right_index3",
        43: "right_middle1",
        44: "right_middle2",
        45: "right_middle3",
        46: "right_pinky1",
        47: "right_pinky2",
        48: "right_pinky3",
        49: "right_ring1",
        50: "right_ring2",
        51: "right_ring3",
        52: "right_thumb1",
        53: "right_thumb2",
        54: "right_thumb3",
    }

    # From https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md
    SMPLH_JOINT_NAMES = {
        0: "pelvis",
        1: "left_hip",
        2: "right_hip",
        3: "spine1",
        4: "left_knee",
        5: "right_knee",
        6: "spine2",
        7: "left_ankle",
        8: "right_ankle",
        9: "spine3",
        10: "left_foot",
        11: "right_foot",
        12: "neck",
        13: "left_collar",
        14: "right_collar",
        15: "head",
        16: "left_shoulder",
        17: "right_shoulder",
        18: "left_elbow",
        19: "right_elbow",
        20: "left_wrist",
        21: "right_wrist",
        22: "left_index1",
        23: "left_index2",
        24: "left_index3",
        25: "left_middle1",
        26: "left_middle2",
        27: "left_middle3",
        28: "left_pinky1",
        29: "left_pinky2",
        30: "left_pinky3",
        31: "left_ring1",
        32: "left_ring2",
        33: "left_ring3",
        34: "left_thumb1",
        35: "left_thumb2",
        36: "left_thumb3",
        37: "right_index1",
        38: "right_index2",
        39: "right_index3",
        40: "right_middle1",
        41: "right_middle2",
        42: "right_middle3",
        43: "right_pinky1",
        44: "right_pinky2",
        45: "right_pinky3",
        46: "right_ring1",
        47: "right_ring2",
        48: "right_ring3",
        49: "right_thumb1",
        50: "right_thumb2",
        51: "right_thumb3",
    }

    # From https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md
    JOINT_NAMES_ = {
        0: "pelvis",
        1: "left_hip",
        2: "right_hip",
        3: "spine1",
        4: "left_knee",
        5: "right_knee",
        6: "spine2",
        7: "left_ankle",
        8: "right_ankle",
        9: "spine3",
        10: "left_foot",
        11: "right_foot",
        12: "neck",
        13: "left_collar",
        14: "right_collar",
        15: "head",
        16: "left_shoulder",
        17: "right_shoulder",
        18: "left_elbow",
        19: "right_elbow",
        20: "left_wrist",
        21: "right_wrist",
        22: "left_hand",
        23: "right_hand",
    }
    JOINT_NAMES = list(JOINT_NAMES_.values())

    # The offsets are vectors that map the position of body joints when the body is in the
    # T-pose standard SMPL pose, relative to the origin [0,0,0] #TODO: check
    # The i-th offset is the distance between the i-th joint and its parent
    OFFSETS_BY_GENDER = {
        "neutral": [
            [-1.79505953e-03, -2.23333446e-01, 2.82191255e-02],
            [6.77246757e-02, -3.14739671e-01, 2.14037877e-02],
            [-6.94655406e-02, -3.13855126e-01, 2.38993038e-02],
            [-4.32792313e-03, -1.14370215e-01, 1.52281192e-03],
            [1.02001221e-01, -6.89938274e-01, 1.69079858e-02],
            [-1.07755594e-01, -6.96424140e-01, 1.50492738e-02],
            [1.15910534e-03, 2.08102144e-02, 2.61528404e-03],
            [8.84055199e-02, -1.08789863e00, -2.67853442e-02],
            [-9.19818258e-02, -1.09483879e00, -2.72625243e-02],
            [2.61610388e-03, 7.37324481e-02, 2.80398521e-02],
            [1.14763659e-01, -1.14368952e00, 9.25030544e-02],
            [-1.17353574e-01, -1.14298274e00, 9.60854266e-02],
            [-1.62284535e-04, 2.87602804e-01, -1.48171829e-02],
            [8.14608431e-02, 1.95481750e-01, -6.04975478e-03],
            [-7.91430834e-02, 1.92565283e-01, -1.05754332e-02],
            [4.98955543e-03, 3.52572414e-01, 3.65317875e-02],
            [1.72437770e-01, 2.25950646e-01, -1.49179062e-02],
            [-1.75155461e-01, 2.25116450e-01, -1.97185045e-02],
            [4.32050017e-01, 2.13178586e-01, -4.23743412e-02],
            [-4.28897421e-01, 2.11787231e-01, -4.11194829e-02],
            [6.81283645e-01, 2.22164620e-01, -4.35452575e-02],
            [-6.84195501e-01, 2.19559526e-01, -4.66786778e-02],
            [7.65325829e-01, 2.14003084e-01, -5.84906248e-02],
            [-7.68817426e-01, 2.13442268e-01, -5.69937621e-02],
        ],
        "male": [
            [-2.17368030e-03, -2.40789171e-01, 2.85837929e-02],
            [5.64076719e-02, -3.23069211e-01, 1.09197095e-02],
            [-6.24834085e-02, -3.31302452e-01, 1.50412615e-02],
            [2.26577066e-03, -1.16385621e-01, -9.80142869e-03],
            [9.98591035e-02, -7.09538672e-01, 1.89567121e-02],
            [-1.05740035e-01, -7.14990334e-01, 1.01982190e-02],
            [6.75421124e-03, 2.15707733e-02, 1.70188978e-02],
            [8.50687677e-02, -1.13641306e00, -1.84712826e-02],
            [-8.66844890e-02, -1.13503592e00, -2.43634517e-02],
            [4.48962197e-03, 7.76031658e-02, 1.98739448e-02],
            [1.26123132e-01, -1.19669901e00, 1.03571141e-01],
            [-1.21524363e-01, -1.19714148e00, 1.05959836e-01],
            [-8.90056260e-03, 2.89238689e-01, -1.35936332e-02],
            [7.61920978e-02, 1.91602856e-01, 9.75772827e-04],
            [-7.84640284e-02, 1.90075502e-01, -3.83344230e-03],
            [1.21264883e-03, 3.78176049e-01, 3.68162277e-02],
            [1.99113481e-01, 2.36807951e-01, -1.80702272e-02],
            [-1.91692337e-01, 2.36928773e-01, -1.23055110e-02],
            [4.54445377e-01, 2.21158923e-01, -4.10167131e-02],
            [-4.51819845e-01, 2.22559484e-01, -4.35742399e-02],
            [7.20154633e-01, 2.33857031e-01, -4.83914475e-02],
            [-7.20928227e-01, 2.29353205e-01, -4.96010068e-02],
            [8.06845178e-01, 2.23221004e-01, -6.39857393e-02],
            [-8.09681985e-01, 2.20701637e-01, -5.97080838e-02],
        ],
        "female": [
            [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
            [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
            [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
            [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
            [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
            [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
            [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
            [8.95999143e-02, -1.04856032e00, -3.04155922e-02],
            [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],
            [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
            [1.12937580e-01, -1.10320516e00, 8.39545265e-02],
            [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],
            [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
            [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
            [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
            [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
            [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
            [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
            [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
            [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
            [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
            [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
            [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
            [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],
        ],
    }

    OFFSETS = OFFSETS_BY_GENDER["neutral"]

    # OFFSETS_ARE_RELATIVE: indicates that offsets are global, so referred to the root joint
    OFFSETS_ARE_RELATIVE = False

    # LEFT_MUL: forward kinematics computation needs to know in which order to perform matrix multiplication
    LEFT_MUL = False

    BODY_MODEL_TYPE = BodyModelType.SMPL


class H36MConstants(BodyConstants):
    # --------------Human3.6M---------------
    # Keypoints are all the joints and end effectors combined
    NUM_KEYPOINTS = 32
    KEYPOINTS = [i for i in range(NUM_KEYPOINTS)]  # TODO: remove this, it's pointless

    # Parents of all Human3.6M keypoints
    # From https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/visualization/fk.py#L28
    PARENTS_ = {
        0: -1,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 0,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 0,
        12: 11,
        13: 12,
        14: 13,
        15: 14,
        16: 12,
        17: 16,
        18: 17,
        19: 18,
        20: 19,
        21: 20,
        22: 19,
        23: 22,
        24: 12,
        25: 24,
        26: 25,
        27: 26,
        28: 27,
        29: 28,
        30: 27,
        31: 30,
    }
    PARENTS = list(PARENTS_.values())

    # Used to find the child of a joint, -1 if there is no child and for the root
    CHILDS_ = {
        0: -1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: -1,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: -1,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: -1,
        16: 17,
        17: 18,
        18: 19,
        19: 22,
        20: -1,
        21: -1,
        22: -1,
        23: -1,
        24: 25,
        25: 26,
        26: 27,
        27: 30,
        28: -1,
        29: -1,
        30: -1,
        31: -1,
    }  # TODO: remove this, seems wrong... if multiple childs, which one is the right one?
    CHILDS = list(CHILDS_.values())

    # Keypoints called joints are those which perform rotation relative to their parent
    # JOINTS = []
    # Keypoints that are at the end of the kinematics tree
    # END_EFFECTORS = []

    # Subset of keypoints used in metrics
    MAJOR_JOINTS = [
        0,
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        24,
        25,
        26,
        27,
    ]

    # Subset of static keypoints that are not used in metrics
    JOINTS_TO_IGNORE = [
        5,
        10,
        15,
        20,
        21,
        22,
        23,
        28,
        29,
        30,
        31,
    ]

    # From https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py#L42
    JOINT_NAMES_ = {
        0: "Hips",
        1: "RightUpLeg",
        2: "RightLeg",
        3: "RightFoot",
        4: "RightToeBase",
        5: "Site",
        6: "LeftUpLeg",
        7: "LeftLeg",
        8: "LeftFoot",
        9: "LeftToeBase",
        10: "Site",
        11: "Spine",
        12: "Spine1",
        13: "Neck",
        14: "Head",
        15: "Site",
        16: "LeftShoulder",
        17: "LeftArm",
        18: "LeftForeArm",
        19: "LeftHand",
        20: "LeftHandThumb",
        21: "Site",
        22: "L_Wrist_End",
        23: "Site",
        24: "RightShoulder",
        25: "RightArm",
        26: "RightForeArm",
        27: "RightHand",
        28: "RightHandThumb",
        29: "Site",
        30: "R_Wrist_End",
        31: "Site",
    }
    JOINT_NAMES = list(JOINT_NAMES_.values())

    # From https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/visualization/fk.py#L149
    OFFSETS = [
        [0.00000000e00, 0.00000000e00, 0.00000000e00],
        [-1.32948591e02, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, -4.42894612e02, 0.00000000e00],
        [0.00000000e00, -4.54206447e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 1.62767078e02],
        [0.00000000e00, 0.00000000e00, 7.49994370e01],
        [1.32948826e02, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, -4.42894413e02, 0.00000000e00],
        [0.00000000e00, -4.54206590e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 1.62767426e02],
        [0.00000000e00, 0.00000000e00, 7.49999480e01],
        [0.00000000e00, 1.00000000e-01, 0.00000000e00],
        [0.00000000e00, 2.33383263e02, 0.00000000e00],
        [0.00000000e00, 2.57077681e02, 0.00000000e00],
        [0.00000000e00, 1.21134938e02, 0.00000000e00],
        [0.00000000e00, 1.15002227e02, 0.00000000e00],
        [0.00000000e00, 2.57077681e02, 0.00000000e00],
        [0.00000000e00, 1.51034226e02, 0.00000000e00],
        [0.00000000e00, 2.78882773e02, 0.00000000e00],
        [0.00000000e00, 2.51733451e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 9.99996270e01],
        [0.00000000e00, 1.00000188e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, 2.57077681e02, 0.00000000e00],
        [0.00000000e00, 1.51031437e02, 0.00000000e00],
        [0.00000000e00, 2.78892924e02, 0.00000000e00],
        [0.00000000e00, 2.51728680e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 9.99998880e01],
        [0.00000000e00, 1.37499922e02, 0.00000000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00],
    ]

    # OFFSETS_ARE_RELATIVE: indicates that offsets are global, so referred to the root joint
    OFFSETS_ARE_RELATIVE = True

    # LEFT_MUL: forward kinematics computation needs to know in which order to perform matrix multiplication
    LEFT_MUL = True

    BODY_MODEL_TYPE = BodyModelType.H36M
