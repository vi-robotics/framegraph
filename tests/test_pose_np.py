import unittest
import numpy as np
import quaternion

from framegraph.pose_np import PoseNP


class TesetPoseNP(unittest.TestCase):

    def test_init(self):
        pos = np.array([1, 2, 3])
        new_pos = np.array([-1, -2, -3])
        new_rot = np.array([0, 1, 0, 0])
        rot = np.array([5, 10, 15, 20])
        p = PoseNP(np.array([*rot, *pos]))

        self.assertTrue(np.allclose(p.translation, pos))
        self.assertTrue(np.allclose(
            quaternion.as_float_array(p.rotation), rot))

        p.translation = new_pos
        p.rotation = new_rot
        self.assertTrue(np.allclose(p.translation, new_pos))
        self.assertTrue(np.allclose(
            quaternion.as_float_array(p.rotation), new_rot))

    def test_identity(self):
        p = PoseNP.identity(5)
        self.assertTrue(p.shape == (5, 7))

        p = PoseNP.identity(1)
        self.assertTrue(p.shape == (7,))
        self.assertTrue(np.allclose(p, np.array([1, 0, 0, 0, 0, 0, 0])))

    def test_pose_multiply(self):
        p1 = PoseNP.from_quat_pos(translation=np.array([1, 2, 3]))
        p2 = PoseNP.from_quat_pos(quat=np.array([0, 1, 0, 0]))

        p1_tmat = p1.as_trans_mat()
        p2_tmat = p2.as_trans_mat()

        p12_tmat = np.matmul(p1_tmat, p2_tmat)
        p12_targ = PoseNP.from_trans_mat(p12_tmat)
        p12 = p1 @ p2

        self.assertTrue(p12.approx_equal(p12_targ))


if __name__ == '__main__':
    unittest.main()
