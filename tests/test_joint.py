import unittest
import numpy as np
import quaternion
from framegraph.joint import Joint, rodrigues


class TestJoint(unittest.TestCase):

    def test_revolute(self):
        """Test that a revolute joint works correctly
        """
        # Rotate around the z axis
        r = Joint.revolute(np.array([0, 0, 1]))
        t_mat = r(np.array([np.pi / 2]))
        rot_vec = np.dot(t_mat, np.array([1, 0, 0, 1]))[:3]
        self.assertTrue(np.allclose(
            rot_vec, np.array([0, 1, 0]), rtol=1e-5, atol=1e-5))

    def test_cylindrical(self):
        """Test that a cylindrical joint works correctly
        """
        # Rotate around the z axis
        r = Joint.cylindrical(np.array([0, 0, 1]))
        t_mat = r(np.array([np.pi / 2, 1.0]))

        rot_vec = np.dot(t_mat[:3, :3], np.array([1, 0, 0]))

        self.assertTrue(np.allclose(
            rot_vec, np.array([0, 1, 0]), rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(t_mat[2, 3], 1))

    def test_revolute_from_dh(self):
        """Test that creating a revolute joint from dh parameters works.
        """
        x_offset = 1
        z_offset = 2
        # Rotate around the z axis
        r = Joint.revolute_from_dh(0, 0, x_offset, z_offset)
        t_mat = r(np.array([np.pi / 2]))
        rot_vec = np.dot(t_mat[:3, :3], np.array([1, 0, 0]))
        self.assertTrue(np.allclose(
            rot_vec, np.array([0, 1, 0]), rtol=1e-5, atol=1e-5))
        self.assertTrue(np.allclose(t_mat[2, 3], z_offset))
        # x was rotated 90 degrees, and is now y
        self.assertTrue(np.allclose(t_mat[1, 3], x_offset))

    def test_rodrigues(self):
        """Test that the rodrigues formula works
        """
        quat = np.random.rand(4)
        quat = quaternion.from_float_array(quat / np.linalg.norm(quat))

        r_vec = quaternion.as_rotation_vector(quat)
        r_mat_targ = quaternion.as_rotation_matrix(quat)

        r_norm = np.linalg.norm(r_vec)
        r_axis = r_vec / r_norm
        r = rodrigues(r_axis, r_norm)
        self.assertTrue(np.allclose(r, r_mat_targ, rtol=1e-5, atol=1e-5))


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
