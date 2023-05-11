import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_EULER_SEQ = 'zxy'


def rotmat2euler(array, to_seq: str = DEFAULT_EULER_SEQ):

    array = np.array(array)
    assert array.shape[-1] == array.shape[-2] == 3

    orig_shape = array.shape[:-1]

    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3, 3))
    rotation = Rotation.from_matrix(array)  # type: ignore
    euler_rot = rotation.as_euler(seq=to_seq)

    # restore original shape, but with converted tensor, now in Euler angles form
    return np.reshape(euler_rot, orig_shape)

def aa2euler(array, to_seq: str = DEFAULT_EULER_SEQ):
    """
    Convert rotation vector representation into Euler angle, with selected Euler sequence
    Shapes: from (..., 3) to (..., 3)

    Default Euler sequence is zxy
    """
    array = np.array(array)
    assert array.shape[-1] == 3

    orig_shape = array.shape

    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_rotvec(array)  # type: ignore
    euler_rot = rotation.as_euler(seq=to_seq)

    # restore original shape, but with converted tensor, now in Euler angles form
    return np.reshape(euler_rot, orig_shape)

def euler2aa(array, from_seq: str = DEFAULT_EULER_SEQ):
    """
    Convert euler angle representation to angle-axis, with selected Euler sequence
    Shapes: from (..., 3) to (..., 3)

    Default Euler sequence is zxy
    """
    array = np.array(array)
    assert array.shape[-1] == 3

    orig_shape = array.shape
    # isolate single rotation vectors in order to convert them
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_euler(from_seq, array)  # type: ignore
    aa_rot = rotation.as_rotvec()
    
    # restore original shape, but with converted tensor
    return np.reshape(aa_rot, orig_shape)

def aa2rotmat(array: np.ndarray) -> np.ndarray:
    """
    Given a tensor of angles in the angle-axis representation, compute the equivalent
    rotation matrix. The input shape is (..., 3); the output shape is (..., 3, 3)
    """
    assert array.shape[-1] == 3
    return_shape = (*array.shape[:-1], 3, 3)

    # isolate single rotations for conversion
    array = np.reshape(array, (-1, 3))
    rotation = Rotation.from_rotvec(array)  # type: ignore
    rotmat = rotation.as_matrix()

    # restore original shape, but the aa vectors of shape (,3) are now (,3,3)
    return np.reshape(rotmat, return_shape)

def rotmat2aa(array: np.ndarray) -> np.ndarray:
    """
    Inverse of aa2rotmat. Takes a tensor of rotation matrices with shape (..., 3, 3) and returns
    a tensor of those rotations converted angle-axis representation with shape (..., 3)
    """
    # check that the input is in rotation matrix form
    assert array.shape[-1] == array.shape[-2] == 3
    # return shape is (..., 3)
    return_shape = array.shape[:-1]

    # isolate single rotations for conversion
    array = np.reshape(array, (-1, 3, 3))
    rotation = Rotation.from_matrix(array)  # type: ignore
    aa = rotation.as_rotvec()

    # restore original shape, but with converted tensor, now in angle-axis form
    return np.reshape(aa, return_shape)
