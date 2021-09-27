import numpy as np
from numba import jit


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
