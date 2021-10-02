
from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import quaternion


class AbstractPose(ABC):
    """Represents a Pose, or translation and rotation. Contains methods to
    instantiate and convert to several formats as well as interpolate, compose
    and transform vectors.
    """

    @property  # type: ignore
    @abstractmethod
    def rotation(self) -> Union[np.ndarray, np.quaternion]:
        """Get the rotation component of the pose as a quaternion.

        Returns:
            Union[np.ndarray, np.quaternion]: A quaternion or a length #N vector
                of quaternions.
        """
        pass  # pragma: no cover

    @rotation.setter  # type: ignore
    @abstractmethod
    def rotation(self, val: Union[np.ndarray, np.quaternion]):
        """Sets the rotation component of the pose using a quaternion or
        an array of quaternions. This must be the same length as the translation
        component.

        Args:
            val (Union[np.ndarray, np.quaternion]): A quaternion or length 4
                vector representing a quaternion, or a length #N vector of
                quaternions or an #Nx4 array representing quaternions.
        """
        pass  # pragma: no cover

    @property  # type: ignore
    @abstractmethod
    def translation(self) -> np.ndarray:
        """Get the translation component of the pose.

        Returns:
            np.ndarray: A length 3 vector or an #Nx3 array representing the
                translation.
        """
        pass  # pragma: no cover

    @translation.setter  # type: ignore
    @abstractmethod
    def translation(self, val: np.ndarray):
        """Sets the translation component of the pose. This must be the same
        length as the rotation component.

        Args:
            val (np.ndarray): A length 3 vector or an #Nx3 array representing
                the translation.
        """
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_quat_pos(cls, quat: Union[np.quaternion, np.ndarray] = None,
                      translation: np.ndarray = None) -> "AbstractPose":
        """Instantiate a pose from a quaternion and position or an array
        of quaternions and positions.

        Args:
            quat (Union[np.quaternion, np.ndarray], optional): A quaternion or
                length 4 vector representing a quaternion, or a length #N vector
                of quaternions or an #Nx4 array representing quaternions.
                Defaults to None.
            pos (np.ndarray, optional): A length 3 vector or #Nx3 array
                representing the translation component of the pose. Defaults to
                None.
        Returns:
            AbstractPose: The resulting pose.
        """
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_rmat_pos(cls, rmat: np.ndarray = None, pos: np.ndarray = None
                      ) -> "AbstractPose":
        """Instantiate a pose from a rotation matrix and position or an array
        of rotation matrices and positions.

        Args:
            rmat (np.ndarray, optional): An 3x3 rotation matrix or #Nx3x3 array
                of rotation matrices. Defaults to None.
            pos (np.ndarray, optional): A length 3 vector or #Nx3 array
                representing the translation component of the pose. Defaults to
                None.
        Returns:
            AbstractPose: The resulting pose.
        """
        pass  # pragma: no cover

    @abstractmethod
    def as_rmat_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Represent the pose as a rotation matrix and position, or array of
        rotation matrices and positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple comprising:
                np.ndarray: An 3x3 rotation matrix or #Nx3x3 array of rotation
                    matrices.
                np.ndarray: A length 3 vector or #Nx3 array representing the
                    translation component of the pose.
        """
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_rvec_pos(cls, rvec: np.ndarray = None, pos: np.ndarray = None
                      ) -> "AbstractPose":
        """Instantiate a pose from a rotation vector and position or an array
        of rotation vectors and positions.

        Args:
            rvec (np.ndarray, optional): A length 3 vector or an #Nx3 array
                representing the rotation vector. A rotation vectors directional
                component represents the axis of rotation, and the norm
                represents the rotation amount in radians. Defaults to None.
            pos (np.ndarray, optional): A length 3 vector or an #Nx3 array
                representing the translational componentt of the pose. Defaults
                to None.
        Returns:
            AbstractPose: The resulting pose.
        """
        pass  # pragma: no cover

    @abstractmethod
    def as_rvec_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Represent the pose as a rotation vector and position, or array of
        rotation vectors and positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple comprising:
                np.ndarray: A length 3 vector or #Nx3 array represnting the
                    rotation vector of the pose. The vector represents the axis
                    of rotation, and the norm represents the angle in radians of
                    the rotation.
                np.ndarray: A length 3 vector or #Nx3 array representing the
                    translation component of the pose.
        """
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_trans_mat(cls, trans_mat: np.ndarray) -> "AbstractPose":
        """Instantiate a pose from a transformation matrix or array of
        transformation matrices.

        Args:
            trans_mat (np.ndarray): A 4x4 or #Nx4x4 array of transformation
                matrices.
        Returns:
            AbstractPose: The resulting pose instance.
        """
        pass  # pragma: no cover

    @abstractmethod
    def as_trans_mat(self) -> np.ndarray:
        """Represent the pose as a transformation matrix or array of
        transformation matrices.

        Returns:
            np.ndarray: An 4x4 or #Nx4x4 array of transformation matrices.
        """
        pass  # pragma: no cover

    @abstractmethod  # type: ignore
    def interp(self, t_init: np.ndarray, t_interp: np.ndarray) -> "AbstractPose":
        """Interpolate the pose along times given by t_interp, given that the
        existing poses are associated with times provided by t_init. This
        performs SQUAD for rotation interpolation, and LERP for translation.

        Args:
            t_init (np.ndarray): An array of length #N (the same length as this
                instance) representing times which the poses are associated
                with.
            t_interp (np.ndarray): An array of length #M representing times to
                sample the interpolant.
        Returns:
            AbstractPose: The resulting interpolated pose.
        """
        pass  # pragma: no cover

    @abstractmethod  # type: ignore
    def transform_vecs(self, vec: np.ndarray) -> np.ndarray:
        """Transforms a vector or vector array by the pose. The size of
        vec must be the same as the number of transforms represented by this
        pose object.
        Args:
            vec (np.ndarray): A length 3 vector or an #Nx3 vector array to
                transform.
        Returns:
            np.ndarray: The transformed vector or vector array.
        """
        pass  # pragma: no cover
