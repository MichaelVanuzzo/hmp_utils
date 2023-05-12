import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_EULER_SEQ = 'zxy'

class RotConvert:
    
    def rotmat2euler(input_seq, eul_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
        """
        Convert rotation matrix representation into Euler angle, with selected Euler sequence
        Shapes: from (..., 3, 3) to (..., 3, 3)

        Default Euler sequence is zxy
        """
        if type(input_seq) is not np.ndarray:
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == input_seq.shape[-2] == 3
        orig_shape = input_seq.shape[:-1]

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3, 3))
        rotation = Rotation.from_matrix(input_seq)
        euler_seq = rotation.as_euler(seq=eul_seq)

        # restore original shape
        return np.reshape(euler_seq, orig_shape)


    def aa2euler(input_seq, eul_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
        """
        Convert angle-axis representation into Euler angle, with selected Euler sequence
        Shapes: from (..., 3) to (..., 3)

        Default Euler sequence is zxy
        """
        if type(input_seq) is not np.ndarray:
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        orig_shape = input_seq.shape

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_rotvec(input_seq)
        euler_seq = rotation.as_euler(seq=eul_seq)

        # restore original shape
        return np.reshape(euler_seq, orig_shape)


    def euler2aa(input_seq, eul_seq: str = DEFAULT_EULER_SEQ) -> np.ndarray:
        """
        Convert Euler angle representation into angle-axis, with selected Euler sequence
        Shapes: from (..., 3) to (..., 3)

        Default Euler sequence is zxy
        """
        if type(input_seq) is not np.ndarray:
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        orig_shape = input_seq.shape

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_euler(eul_seq, input_seq)
        aa_seq = rotation.as_rotvec()

        # restore original shape
        return np.reshape(aa_seq, orig_shape)


    def aa2rotmat(input_seq) -> np.ndarray:
        """
        Convert angle-axis representation into rotation matrix,
        Shapes: from (..., 3) to (..., 3, 3)

        """
        if type(input_seq) is not np.ndarray:
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        return_shape = (*input_seq.shape[:-1], 3, 3)

        # isolate single rotations for conversion
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_rotvec(input_seq)
        rotmat_seq = rotation.as_matrix()

        # restore original shape
        return np.reshape(rotmat_seq, return_shape)


    def rotmat2aa(input_seq) -> np.ndarray:
        """
        Convert rotation matri representation into angle-axis,
        Shapes: from (..., 3, 3) to (..., 3)

        """
        if type(input_seq) is not np.ndarray:
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == input_seq.shape[-2] == 3
        return_shape = input_seq.shape[:-1]

        # isolate single rotations for conversion
        input_seq = np.reshape(input_seq, (-1, 3, 3))
        rotation = Rotation.from_matrix(input_seq)
        aa_seq = rotation.as_rotvec()

        # restore original shape
        return np.reshape(aa_seq, return_shape)
