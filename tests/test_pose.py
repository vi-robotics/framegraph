import unittest
import numpy as np
import quaternion
from framegraph.pose import Pose


class TestFrameGraph(unittest.TestCase):

    def test_pose(self):
        """Test initializing a pose works.
        """
        p = Pose()
        self.assertTrue(p.rotation == quaternion.one)
        self.assertTrue(np.allclose(p.translation, np.zeros(3)))

    def test_interp(self):
        """Test that interpolating with a pose works.
        """
        rots = quaternion.from_rotation_vector(np.array(
            [[np.pi / 2, 0, 0],
             [0, 0, 0],
             [-np.pi / 2, 0, 0]]
        ))
        poss = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [2, 4, 6]
        ])
        p = Pose(rotation=rots, translation=poss)
        t_init = np.array([0, 0.5, 1])
        t_interp = np.linspace(0, 1, 10)
        p_interp = p.interp(t_init, t_interp)

        p_interp_rot_targ = quaternion.squad(rots, t_init, t_interp)
        p_interp_pos_targ = np.zeros((len(t_interp), 3))
        for i in range(3):
            p_interp_pos_targ[:, i] = np.interp(t_interp, t_init, poss[:, i])

        self.assertTrue(np.allclose(p_interp.translation, p_interp_pos_targ))
        self.assertTrue(np.allclose(
            quaternion.as_float_array(p_interp.rotation),
            quaternion.as_float_array(p_interp_rot_targ)))

    def test_pose_multiply(self):
        """Test that multiplying two poses is equivalent to multiplying the
        transformation matrices created by the poses.
        """
        p0 = Pose(
            rotation=quaternion.from_rotation_vector(
                np.array([0, 0, np.pi / 4])),
            translation=np.array([1, 0, 0]))
        p1 = Pose(
            rotation=quaternion.from_rotation_vector(
                np.array([0, 0, np.pi / 4])),
            translation=np.array([0, 1, 0])
        )
        pm = p0 * p1
        tm_targ = np.matmul(p0.as_trans_mat(), p1.as_trans_mat())
        self.assertTrue(np.allclose(pm.as_trans_mat(), tm_targ))


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
