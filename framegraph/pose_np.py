from typing import Union, Any, Tuple
from numba import jit
import numpy as np
import quaternion


from framegraph.pose_abc import AbstractPose
from framegraph.utils import transform_vecs, transform_vec


class PoseNP(np.ndarray, AbstractPose):

    @classmethod
    def from_quat_pos(cls, quat: Union[np.quaternion, np.ndarray] = None,
                      translation: np.ndarray = None):
        quat, translation = cls._fill_defaults(quat, translation)
        quat = cls._get_float_quat_rep(quat)
        arr = np.concatenate([quat, translation], axis=-1)
        return cls(arr)

    @classmethod
    def from_rmat_pos(cls, rmat: np.ndarray = None, pos: np.ndarray = None):
        quat = quaternion.from_rotation_matrix(rmat)
        return cls.from_quat_pos(quat, pos)

    def as_rmat_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        return quaternion.as_rotation_matrix(self.rotation), self.translation

    @classmethod
    def from_rvec_pos(cls, rvec: np.ndarray = None, pos: np.ndarray = None):
        quat = quaternion.from_rotation_vector(rvec)
        return cls.from_quat_pos(quat, pos)

    def as_rvec_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        return quaternion.as_rotation_vector(self.rotation), self.translation

    @classmethod
    def from_trans_mat(cls, trans_mat: np.ndarray):
        rot_mat = trans_mat[..., :3, :3]
        quat = quaternion.from_rotation_matrix(rot_mat)
        trans = trans_mat[..., :3, 3]
        return cls.from_quat_pos(quat, trans)

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
    def identity(cls, n: int = 1):
        arr = np.zeros((n, 7), dtype=np.float64)
        arr[:, 0] = 1
        return cls(np.squeeze(arr))

    def interp(self, t_init: np.ndarray, t_interp: np.ndarray):
        interp_rots = quaternion.squad(self.rotation, t_init, t_interp)
        interp_trans = np.apply_along_axis(
            lambda x: np.interp(t_interp, t_init, x), 0, self.translation)
        return self.from_quat_pos(interp_rots, interp_trans)

    @property
    def translation(self):
        return super().__getitem__(np.s_[..., 4:])

    @translation.setter
    def translation(self, value: np.ndarray):
        super().__setitem__(np.s_[..., 4:], value)

    @property
    def rotation(self):
        return quaternion.from_float_array(super().__getitem__(np.s_[..., :4]))

    @rotation.setter
    def rotation(self, val: Union[np.ndarray, np.quaternion]):
        super().__setitem__(np.s_[..., :4], self._get_float_quat_rep(val))

    def transform_vecs(self, vec: np.ndarray) -> np.ndarray:
        rotation = quaternion.as_float_array(self.rotation)
        if rotation.ndim == 1:
            return transform_vec(rotation, self.translation, vec)
        return transform_vecs(rotation, self.translation, vec)

    def approx_equal(self, other: "PoseNP") -> bool:
        self_rot = quaternion.as_float_array(self.rotation)
        other_rot = quaternion.as_float_array(other.rotation)
        rots_eq = bool(np.all(np.bitwise_or(np.isclose(self_rot, other_rot),
                                            np.isclose(self_rot, -1 * other_rot))))
        trans_eq = np.allclose(self.translation, other.translation)
        return (trans_eq and rots_eq)

    @ classmethod
    def _fill_defaults(cls, quats: Union[np.ndarray, np.quaternion] = None,
                       translation: np.ndarray = None):
        if quats is None and translation is None:
            quats = np.array([1, 0, 0, 0], dtype=np.float64)
            translation = np.zeros(3)
        elif quats is None and translation is not None:
            translation_len = cls._get_num_trans(translation)
            quats = np.squeeze(
                np.zeros((translation_len, 4), dtype=np.float64))
            quats[..., 0] = 1
        elif quats is not None and translation is None:
            quat_len = cls._get_num_quats(quats)
            translation = np.squeeze(np.zeros((quat_len, 3), dtype=np.float64))
        return quats, translation

    @ staticmethod
    def _get_num_quats(quats: Union[np.ndarray, np.quaternion]) -> int:
        if isinstance(quats, np.quaternion):
            return quats.size
        elif isinstance(quats, np.ndarray):
            if quats.dtype == np.quaternion:
                return quats.size
            else:
                if quats.ndim == 1:
                    return 1
                else:
                    return np.cumprod(quats.shape[:-1])[0]
        else:
            raise TypeError("quats is not a valid quaternion array.")

    @ staticmethod
    def _get_num_trans(translation: np.ndarray) -> int:
        if translation.ndim == 1:
            return 1
        else:
            return np.cumprod(translation.shape[:-1])[0]

    @ staticmethod
    def _get_float_quat_rep(value: Union[np.ndarray, np.quaternion]) -> np.ndarray:
        res = None
        if isinstance(value, np.ndarray):
            if value.dtype == np.quaternion:
                res = quaternion.as_float_array(value)
            else:
                # This is a numpy array with no quaternions.
                res = value
        else:
            if isinstance(value, np.quaternion):
                res = quaternion.as_float_array(value)
            else:
                # This isn't a numpy array, and not a quaternion. So it's
                # an invalid value.
                raise TypeError(f"value is type {type(value)}, which is not"
                                " a numpy array or quaternion.")
        return res

    def __matmul__(self, other):
        rotation = self.rotation * other.rotation
        translation = self.transform_vecs(other.translation)
        return self.__class__.from_quat_pos(rotation, translation)

    def __new__(cls, input_array):
        if hasattr(cls, '__abstractmethods__') and len(cls.__abstractmethods__) > 0:
            raise TypeError(  # pragma: no cover
                f"Can't instantiate abstract class {cls.__name__} with abstract"
                f" methods {', '.join(cls.__abstractmethods__)}")
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        return obj
