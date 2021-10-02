from typing import Callable
import jax.numpy as jnp
from jax import jit


@jit
def rodrigues(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Convert from a rotation vector in axis-angle format to a rotation matrix.

    Args:
        axis (jnp.ndarray): A unit vector of length 3 representing the axis of
            rotation
        angle (float): The angle in radians to rotate around the given axis.

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    wx = axis[0]
    wy = axis[1]
    wz = axis[2]
    return jnp.array([
        [c + (wx**2) * (1 - c), wx * wy * (1 - c) -
         wz * s, wy * s + wx * wz * (1 - c)],
        [wz * s + wx * wy * (1 - c), c + (wy**2) *
         (1 - c), -wx * s + wy * wz * (1 - c)],
        [-wy * s + wx * wz * (1 - c), wx * s + wy * wz *
         (1 - c), c + (wz**2) * (1 - c)]
    ])


class Joint():
    """A helper class to generate transform callbacks for common joint
    scenarios.
    """

    @staticmethod
    def revolute(axis: jnp.ndarray) -> Callable[[jnp.array], jnp.array]:
        """A revolute joint representing rotation around a given axis.

        Args:
            axis (jnp.ndarray): An #Nx3 array representing the axis of rotation.

        Returns:
            Callable[[jnp.array], jnp.array]: A callback which takes in a length
                1 array representing the angle around the provided axis and
                returns a 4x4 transformation matrix representing the rotation
                (no translation occurs).
        """
        axis = axis / jnp.linalg.norm(axis)

        @jit
        def callback(params: jnp.ndarray):
            theta, = params
            r_mat = rodrigues(axis, theta)
            t_mat = jnp.eye(4, dtype=jnp.float32)
            return t_mat.at[:3, :3].set(r_mat)
        return callback

    @staticmethod
    def cylindrical(axis: jnp.ndarray):
        """A cylindrical joint representing rotation around a given axis and
        translation along the axis.

        Args:
            axis (jnp.ndarray): An #Nx3 array representing the axis of rotation
                and translation.

        Returns:
            Callable[[jnp.array], jnp.array]: A callback which takes in a length
                2 array representing the angle around the provided axis and the
                disatnce along the axis to translate, and returns a 4x4
                transformation matrix representing the transformation.
        """
        axis = axis / jnp.linalg.norm(axis)

        @jit
        def callback(params: jnp.ndarray):
            theta, dist = params

            r_mat = rodrigues(axis, theta)
            t_mat = jnp.eye(4, dtype=jnp.float32)
            t_mat = t_mat.at[:3, :3].set(r_mat)
            t_mat = t_mat.at[:3, 3].set(dist * axis)
            return t_mat

        return callback

    @staticmethod
    def revolute_from_dh(theta_offset: float, alpha: float, a: float, d: float):
        """A revolute joint representing a rotation around an axis with a
        translational offset calculated from Denavit-Hartenberg parameters.

        Args:
            theta_offset (float): Angle about previous z, from old x to new x
            alpha (float): Angle about the common normal, from old z axis to new
                axis.
            a (float): Length of the common normal. This is the radius about the
                previous z.
            d (float): Offset along the previous z to the common normal.

        Returns:
            Callable[[jnp.array], jnp.array]: A callback which takes in a length
                1 array representing joint angle, and returns a 4x4
                transformation matrix representing the transformation.
        """
        @jit
        def callback(params: jnp.ndarray):
            theta, = params
            theta += theta_offset
            c = jnp.cos(theta)
            s = jnp.sin(theta)
            sa = jnp.sin(alpha)
            ca = jnp.cos(alpha)
            t_mat = jnp.array([
                [c, -s * ca, s * sa, a * c],
                [s, c * ca, -c * sa, a * s],
                [0, sa, ca, d],
                [0, 0, 0, 1]])
            return t_mat
        return callback

    @staticmethod
    def fixed(trans_mat: jnp.ndarray):
        """A fixed joint representing a static transformation.

        Args:
            trans_mat (jnp.ndarray): A 4x4 transformation matrix representing
                the fixed transform.

        Returns:
            Callable[[jnp.array], jnp.array]: A callback which takes in no
                parameters, and returns a 4x4 transformation matrix.
        """
        def callback():
            return trans_mat
        return callback
