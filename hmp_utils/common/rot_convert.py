import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union, Optional


class RotConvert:
    """
    Class for converting between different rotation representations

    """

    DEFAULT_EULER_SEQ = "zxy"

    @staticmethod
    def rotmat2euler(
        input_seq: Union[list, np.ndarray],
        eul_seq: str = DEFAULT_EULER_SEQ,
        eul_degrees: bool = False,
    ) -> np.ndarray:
        """
        Convert rotation matrix representation into Euler angle, with selected Euler sequence.
        Shapes: from (..., 3, 3) to (..., 3).

        Default Euler sequence is zxy.
        Default Euler unit is radians.

        ---
        ### Args:
        - `input_seq` (list, ndarray): rotation matrix sequence with shape (..., 3, 3)
        - `eul_seq` (str): Euler sequence to use for conversion
        - `eul_degrees` (bool): Euler unit
        ---
        ### Returns:
        - `euler_seq` (ndarray): Euler angle sequence with shape (..., 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == input_seq.shape[-2] == 3
        orig_shape = input_seq.shape[:-1]

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3, 3))
        rotation = Rotation.from_matrix(input_seq)
        euler_seq = rotation.as_euler(seq=eul_seq, degrees=eul_degrees)

        # restore original shape
        return np.reshape(euler_seq, orig_shape)

    @staticmethod
    def euler2rotmat(
        input_seq: Union[list, np.ndarray],
        eul_seq: str = DEFAULT_EULER_SEQ,
        eul_degrees: bool = False,
    ) -> np.ndarray:
        """
        Convert Euler angle representation into rotation matrix, with selected Euler sequence.
        Shapes: from (..., 3) to (..., 3, 3).

        Default Euler sequence is zxy.
        Default Euler unit is radians.

        ---
        ### Args:
        - `input_seq` (list, ndarray): Euler angle sequence with shape (..., 3)
        ---
        ### Returns:
        - `rotmat_seq` (ndarray): rotation matrix sequence with shape (..., 3, 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        return_shape = (*input_seq.shape[:-1], 3, 3)

        # isolate single rotations for conversion
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_euler(eul_seq, input_seq, eul_degrees)
        rotmat_seq = rotation.as_matrix()

        # restore original shape
        return np.reshape(rotmat_seq, return_shape)

    @staticmethod
    def aa2euler(
        input_seq: Union[list, np.ndarray],
        eul_seq: str = DEFAULT_EULER_SEQ,
        eul_degrees: bool = False,
    ) -> np.ndarray:
        """
        Convert angle-axis representation into Euler angle, with selected Euler sequence.
        Shapes: from (..., 3) to (..., 3).

        Default Euler sequence is zxy.
        Default Euler unit is radians.

        ---
        ### Args:
        - `input_seq` (list, ndarray): angle-axis sequence with shape (..., 3)
        - `eul_seq` (str): Euler sequence to use for conversion
        - `eul_degrees` (bool): Euler unit
        ---
        ### Returns:
        - `euler_seq` (ndarray): Euler angle sequence with shape (..., 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        orig_shape = input_seq.shape

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_rotvec(input_seq)
        euler_seq = rotation.as_euler(seq=eul_seq, degrees=eul_degrees)

        # restore original shape
        return np.reshape(euler_seq, orig_shape)

    @staticmethod
    def euler2aa(
        input_seq: Union[list, np.ndarray],
        eul_seq: str = DEFAULT_EULER_SEQ,
        eul_degrees: bool = False,
    ) -> np.ndarray:
        """
        Convert Euler angle representation into angle-axis, with selected Euler sequence.
        Shapes: from (..., 3) to (..., 3).

        Default Euler sequence is zxy.
        Default Euler unit is radians.

        ---
        ### Args:
        - `input_seq` (list, ndarray): Euler angle sequence with shape (..., 3)
        - `eul_seq` (str): Euler sequence to use for conversion
        - `eul_degrees` (bool): Euler unit
        ---
        ### Returns:
        - `aa_seq` (ndarray): angle-axis sequence with shape (..., 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        orig_shape = input_seq.shape

        # isolate single rotation vectors in order to convert them
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_euler(eul_seq, input_seq, eul_degrees)
        aa_seq = rotation.as_rotvec()

        # restore original shape
        return np.reshape(aa_seq, orig_shape)

    @staticmethod
    def aa2rotmat(input_seq: Union[list, np.ndarray]) -> np.ndarray:
        """
        Convert angle-axis representation into rotation matrix.
        Shapes: from (..., 3) to (..., 3, 3).

        ---
        ### Args:
        - `input_seq` (list, ndarray): angle-axis sequence with shape (..., 3)
        ---
        ### Returns:
        - `rotmat_seq` (ndarray): rotation matrix sequence with shape (..., 3, 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == 3
        return_shape = (*input_seq.shape[:-1], 3, 3)

        # isolate single rotations for conversion
        input_seq = np.reshape(input_seq, (-1, 3))
        rotation = Rotation.from_rotvec(input_seq)
        rotmat_seq = rotation.as_matrix()

        # restore original shape
        return np.reshape(rotmat_seq, return_shape)

    @staticmethod
    def rotmat2aa(input_seq: Union[list, np.ndarray]) -> np.ndarray:
        """
        Convert rotation matri representation into angle-axis.
        Shapes: from (..., 3, 3) to (..., 3).

        ---
        ### Args:
        - `input_seq` (list, ndarray): rotation matrix sequence with shape (..., 3, 3)
        ---
        ### Returns:
        - `aa_seq` (ndarray): angle-axis sequence with shape (..., 3)
        """
        if not isinstance(input_seq, np.ndarray):
            input_seq = np.array(input_seq)

        assert input_seq.shape[-1] == input_seq.shape[-2] == 3
        return_shape = input_seq.shape[:-1]

        # isolate single rotations for conversion
        input_seq = np.reshape(input_seq, (-1, 3, 3))
        rotation = Rotation.from_matrix(input_seq)
        aa_seq = rotation.as_rotvec()

        # restore original shape
        return np.reshape(aa_seq, return_shape)
