from typing import Union, Any
import numpy as np
import quaternion

from framegraph.utils import transform_vecs, transform_vec
from framegraph.pose_abc import AbstractPose


class Pose(AbstractPose):
    def __init__(self,
                 rotation: Union[np.ndarray, np.quaternion] = None,
                 translation: np.ndarray = None):

        if rotation is None and translation is None:
            rotation = quaternion.one
            translation = np.zeros(3)
        elif rotation is None and translation is not None:
            if translation.ndim == 1:
                rotation = quaternion.one
            else:
                rotation = np.full(
                    translation.shape[0], quaternion.one, dtype=np.quaternion)
        elif rotation is not None and translation is not None:
            pass
        # Rotation is not none and translation is none
        else:
            rotation = self._get_standard_rotation(rotation)
            translation = np.zeros((len(rotation), 3))

        self.rotation = rotation
        self.translation = translation
        self._check_consistency()

    @classmethod
    def from_quat_pos(cls, quat: Union[np.quaternion, np.ndarray] = None,
                      translation: np.ndarray = None):
        return cls(rotation=quat, translation=translation)

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation: np.ndarray):
        if translation.ndim not in [1, 2]:
            raise ValueError("translation must be 1 or 2 dimensional")
        if translation.shape[-1] != 3:
            raise ValueError("translation must have last axis length of 3")
        self._translation = translation

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Union[np.ndarray, np.quaternion]):
        self._rotation = self._get_standard_rotation(rotation)

    def interp(self, t_init: np.ndarray, t_interp: np.ndarray):
        interp_rots = quaternion.squad(self.rotation, t_init, t_interp)
        interp_trans = np.apply_along_axis(
            lambda x: np.interp(t_interp, t_init, x), 0, self.translation)
        return self.__class__(interp_rots, interp_trans)

    def transform_vecs(self, vec: np.ndarray) -> np.ndarray:
        rot = quaternion.as_float_array(self.rotation)
        if rot.ndim == 1:
            return transform_vec(rot, self.translation, vec)
        return transform_vecs(rot, self.translation, vec)

    def as_trans_mat(self):
        r_mat = quaternion.as_rotation_matrix(self.rotation)
        if r_mat.ndim == 3:
            trans_mat = np.tile(np.eye(4), (r_mat.shape[0], 1, 1))
        else:
            trans_mat = np.eye(4)

        trans_mat[..., :3, :3] = r_mat
        trans_mat[..., :3, 3] = self.translation
        return trans_mat

    @classmethod
    def from_trans_mat(cls, trans_mat: np.ndarray):
        translation = trans_mat[..., :3, 3]
        rotation = quaternion.from_rotation_matrix(trans_mat[..., :3, :3])
        return cls(rotation=rotation, translation=translation)

    def as_rmat_pos(self):
        return quaternion.as_rotation_matrix(self.rotation), self.translation

    @classmethod
    def from_rmat_pos(cls, rmat: np.ndarray = None, pos: np.ndarray = None):
        quat = quaternion.from_rotation_matrix(rmat)
        return cls(rotation=quat, translation=pos)

    def as_rvec_pos(self):
        return quaternion.as_rotation_vector(self.rotation), self.translation

    @classmethod
    def from_rvec_pos(cls, rvec: np.ndarray = None, pos: np.ndarray = None):
        return cls(rotation=quaternion.from_rotation_vector(rvec), translation=pos)

    def _check_consistency(self):
        if isinstance(self.rotation, np.quaternion):
            assert self.translation.shape == (3,)
        elif isinstance(self.rotation, np.ndarray):
            assert self.translation.ndim == 2
            assert self.translation.shape[0] == self.rotation.shape[0]
        else:
            raise AssertionError("Rotation is of inconsistent type.")

    @staticmethod
    def _get_standard_rotation(rotation: Any):
        std_form = None
        if isinstance(rotation, np.quaternion):
            std_form = rotation
        elif isinstance(rotation, np.ndarray):
            if rotation.dtype == np.quaternion:
                std_form = rotation
            elif rotation.ndim == 1 or rotation.ndim == 2:
                std_form = quaternion.from_float_array(rotation)
            else:
                raise ValueError("rotation has invalid number of dimensions:"
                                 f" {rotation.ndim}")
        else:
            raise ValueError(f"rotation has invalid type: {type(rotation)}")
        return std_form

    def __getitem__(self, key):
        return self.__class__(rotation=self.rotation[key],
                              translation=self.translation[key])

    def __mul__(self, other):
        rot = self.rotation * other.rotation
        trans = self.transform_vecs(other.translation)
        return self.__class__(rotation=rot, translation=trans)
