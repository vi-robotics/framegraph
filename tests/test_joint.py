import unittest
import numpy as np
from framegraph.joint import Joint


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


if __name__ == '__main__':
    unittest.main()
