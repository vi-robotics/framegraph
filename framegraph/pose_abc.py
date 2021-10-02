
from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np
import quaternion


class AbstractPose(ABC):

    @property  # type: ignore
    @abstractmethod
    def rotation(self) -> Union[np.ndarray, np.quaternion]:
        pass  # pragma: no cover

    @rotation.setter  # type: ignore
    @abstractmethod
    def rotation(self, val: Union[np.ndarray, np.quaternion]):
        pass  # pragma: no cover

    @property  # type: ignore
    @abstractmethod
    def translation(self) -> np.ndarray:
        pass  # pragma: no cover

    @translation.setter  # type: ignore
    @abstractmethod
    def translation(self, val: np.ndarray):
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_quat_pos(cls, quat: Union[np.quaternion, np.ndarray] = None,
                      translation: np.ndarray = None):
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_rmat_pos(cls, rmat: np.ndarray = None, pos: np.ndarray = None):
        pass  # pragma: no cover

    @abstractmethod
    def as_rmat_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_rvec_pos(cls, rvec: np.ndarray = None, pos: np.ndarray = None):
        pass  # pragma: no cover

    @abstractmethod
    def as_rvec_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        pass  # pragma: no cover

    @classmethod  # type: ignore
    @abstractmethod
    def from_trans_mat(cls, trans_mat: np.ndarray):
        pass  # pragma: no cover

    @abstractmethod
    def as_trans_mat(self) -> np.ndarray:
        pass  # pragma: no cover

    @abstractmethod  # type: ignore
    def interp(self, t_init: np.ndarray, t_interp: np.ndarray):
        pass  # pragma: no cover

    @abstractmethod  # type: ignore
    def transform_vecs(self, vec: np.ndarray) -> np.ndarray:
        """Transforms a vector or vector array by the pose. The size of
        vec must be the same as the number of transforms represented by this
        pose object.
        Args:
            vec (np.ndarray): A #Nx3 vector array or a 1x3 vector to
                transform.
        Returns:
            np.ndarray: The transformed vector or vector array.
        """
        pass  # pragma: no cover
