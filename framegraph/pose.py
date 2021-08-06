from typing import Union, Any
import numpy as np
from numba import jit
import quaternion


@jit(nopython=True, cache=True)
def transform_vec(rotation: np.ndarray,
                  translation: np.ndarray,
                  vec: np.ndarray) -> np.ndarray:
    """Transforms a vector by a pose represented by a rotation then translation.
    Args:
        rotation (np.ndarray): A 1x4 array representing the rotation quaternion.
        translation (np.ndarray): A 1x3 array representing the translation.
        vec (np.ndarray): The vector to transform
    Returns:
        np.ndarray: The transformed vector
    """
    scalar_comp = rotation[0]
    imag_comp = rotation[1:]
    norm = np.sum(rotation**2)
    comp_a = 2 * imag_comp
    comp_b = scalar_comp * vec
    comp_c = np.cross(imag_comp, vec)
    comp_d = (comp_b + comp_c) / norm
    rot = (np.cross(comp_a, comp_d) + vec)
    return rot + translation


@jit(nopython=True, cache=True)
def transform_vecs(rotation: np.ndarray,
                   translation: np.ndarray,
                   vec_array: np.ndarray) -> np.ndarray:
    """Transforms an array of vectors by an array of poses represented by an
    array of rotations (quaternions) and array of translations.
    Args:
        rotation (np.ndarray): A #Nx4 array representing the rotation
            quaternion array.
        translation (np.ndarray): A #Nx3 array representing the
            translation array.
        vec_array (np.ndarray):  An #Nx3 vector array to transform
    Returns:
        np.ndarray: The #Nx3 transformed vector array
    """
    scalar_comp = rotation[:, 0]
    imag_comp = rotation[:, 1:]
    norm = np.sum(rotation**2, axis=1)
    comp_a = 2 * imag_comp
    comp_b = np.expand_dims(scalar_comp, 1) * vec_array
    comp_c = np.cross(imag_comp, vec_array)
    comp_d = (comp_b + comp_c) / np.expand_dims(norm, 1)
    rot = (np.cross(comp_a, comp_d) + vec_array)
    return rot + translation


class Pose():
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
        else:
            rotation = self._get_standard_rotation(rotation)
            translation = np.zeros((len(rotation), 3))

        self.rotation = rotation
        self.translation = translation
        self._check_consistency()

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
        """Transforms a vector or vector array by the pose. The size of
        vec must be the same as the number of transforms represented by this
        pose object.
        Args:
            vec (np.ndarray): A #Nx3 vector array or a 1x3 vector to
                transform.
        Returns:
            np.ndarray: The transformed vector or vector array.
        """
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
