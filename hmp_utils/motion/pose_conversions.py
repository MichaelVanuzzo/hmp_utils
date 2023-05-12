import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_EULER_SEQ = 'zxy'


def rotmat2euler(array, to_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
    """
    Convert rotation matrix representation into Euler angle, with selected Euler sequence
    Shapes: from (..., 3, 3) to (..., 3, 3)

    Default Euler sequence is zxy
    """
    if type(array) is not np.ndarray:
        array = np.array(array)

    assert array.shape[-1] == array.shape[-2] == 3
    orig_shape = array.shape[:-1]

    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3, 3))
    rotation = Rotation.from_matrix(array)
    euler_rot = rotation.as_euler(seq=to_seq)

    # restore original shape
    return np.reshape(euler_rot, orig_shape)

def aa2euler(array, to_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
    """
    Convert angle-axis representation into Euler angle, with selected Euler sequence
    Shapes: from (..., 3) to (..., 3)

    Default Euler sequence is zxy
    """
    if type(array) is not np.ndarray:
        array = np.array(array)

    assert array.shape[-1] == 3
    orig_shape = array.shape

    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_rotvec(array)
    euler_rot = rotation.as_euler(seq=to_seq)

    # restore original shape
    return np.reshape(euler_rot, orig_shape)

def euler2aa(array, from_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
    """
    Convert Euler angle representation into angle-axis, with selected Euler sequence
    Shapes: from (..., 3) to (..., 3)

    Default Euler sequence is zxy
    """
    if type(array) is not np.ndarray:
        array = np.array(array)

    assert array.shape[-1] == 3
    orig_shape = array.shape

    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_euler(from_seq, array) 
    aa_rot = rotation.as_rotvec()
    
    # restore original shape
    return np.reshape(aa_rot, orig_shape)

def aa2rotmat(array) -> np.ndarray:
    """
    Convert angle-axis representation into rotation matrix,
    Shapes: from (..., 3) to (..., 3, 3)

    """
    if type(array) is not np.ndarray:
        array = np.array(array)

    assert array.shape[-1] == 3
    return_shape = (*array.shape[:-1], 3, 3)

    # isolate single rotations for conversion
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_rotvec(array)  
    rotmat = rotation.as_matrix()

    # restore original shape
    return np.reshape(rotmat, return_shape)

def rotmat2aa(array) -> np.ndarray:
    """
    Convert rotation matri representation into angle-axis,
    Shapes: from (..., 3, 3) to (..., 3)

    """
    if type(array) is not np.ndarray:
        array = np.array(array)

    assert array.shape[-1] == array.shape[-2] == 3
    return_shape = array.shape[:-1]

    # isolate single rotations for conversion
    array = np.reshape(array, (-1, 3, 3))
    rotation = Rotation.from_matrix(array) 
    aa = rotation.as_rotvec()

    # restore original shape
    return np.reshape(aa, return_shape)
