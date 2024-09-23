import numpy as np
from typing import Union


class TfUtils:
    """
    Class that stores functions to interact with transformation matrices.
    """

    @staticmethod
    def apply_transformation(
        H: np.ndarray,
        points: Union[np.ndarray, list],
    ) -> np.ndarray:
        """
        This function applies the transformation of the H matrix to an array of points.

        ---
        ### Args:
        - H (np.ndarray): 4x4 homogeneous matrix that represent the transformation
        - points (np.ndarray): Array of N points with shape (..., N, 3)

        ### Returns:
        - new_points (np.ndarray): The transformed points
        """

        # convert points to numpy if it is a list
        if type(points) == list:
            points = np.array(points)

        # add one dimension if points has only one
        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)

        # check H shape
        if H.shape != (4, 4):
            raise ValueError(
                f"The shape of the homogeneous matrix is wrong. Expected (4, 4) but got {H.shape}."
            )

        # check points shape
        if points.shape[-1] != 3:
            raise ValueError(
                f"The shape of the points array is wrong. Expected (..., N, 3) but got {points.shape}."
            )

        # save original shape
        original_shape = points.shape

        # remove extra dimensions
        points.reshape((-1, 3))

        # convert to homogeneous coordinates
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])

        # calculate new coordinates
        new_points = (H @ points_hom.T).T

        # return from homogeneous coordinates
        new_points = new_points[:, :3]

        # return new_points with original shape
        return new_points.reshape(original_shape)
