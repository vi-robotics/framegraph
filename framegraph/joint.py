import jax.numpy as jnp
from jax import jit


@jit
def rodrigues(axis: jnp.ndarray, angle: float) -> jnp.ndarray:

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

    @staticmethod
    def revolute(axis: jnp.ndarray):
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
        def callback():
            return trans_mat
        return callback
