import unittest
import numpy as np
import quaternion
from framegraph.pose import Pose


class TestPose(unittest.TestCase):

    def test_init(self):
        """Test initializing a pose works.
        """
        p = Pose()
        self.assertTrue(p.rotation == quaternion.one)
        self.assertTrue(np.allclose(p.translation, np.zeros(3)))

        # Test only translation initialization.
        p = Pose(translation=np.zeros(3))
        self.assertTrue(p.rotation == quaternion.one)
        self.assertTrue(np.allclose(p.translation, np.zeros(3)))

        # Test only translation with non-unit shape initialization.
        p = Pose(translation=np.zeros((10, 3)))
        self.assertTrue(len(p.rotation) == 10)
        self.assertTrue(p.translation.shape == (10, 3))

        # Test that if rotation is not None and translation is None, then
        # translation is initialized properly
        p = Pose(rotation=quaternion.from_float_array(np.random.rand(10, 4)))
        self.assertTrue(len(p.rotation) == 10)
        self.assertTrue(p.translation.shape == (10, 3))

        # Check that the class method required by AbstractPose works.
        p = Pose.from_quat_pos(
            quaternion.from_float_array(np.random.rand(10, 4)))
        self.assertTrue(len(p.rotation) == 10)
        self.assertTrue(p.translation.shape == (10, 3))

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

    def test_setting_translation(self):
        """Test that setting translation works and throws proper errors when
        bad values are set.
        """
        p = Pose(translation=np.zeros((10, 3)))

        def cause_error():
            p.translation = np.zeros((10, 1, 3,))
        self.assertRaises(ValueError, cause_error)

        def cause_error():
            p.translation = np.zeros((10, 4))
        self.assertRaises(ValueError, cause_error)

    def test_rot_vecs(self):
        """Test that rotating vectors works.
        """
        # Test that rotating with a single pose works.
        quat = quaternion.from_float_array(np.random.rand(4))
        t = np.random.rand(3)
        p = Pose(quat, t)
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = quaternion.as_rotation_matrix(quat)
        trans_mat[:3, 3] = t

        vec = np.random.rand(3)
        tmat_rot_vec = np.dot(trans_mat, np.array([*vec, 1]))[:3]
        p_rot_vec = p.transform_vecs(vec)

        self.assertTrue(np.allclose(tmat_rot_vec, p_rot_vec))
        self.assertTrue(np.allclose(p.as_trans_mat(), trans_mat))
        # We just checked that as_trans_mat works, so now let's use it to test
        # from_trans_mat
        q = Pose.from_trans_mat(trans_mat)
        self.assertTrue(np.allclose(trans_mat, q.as_trans_mat()))

        # Test that a pose with multiple transforms rotates vectors correctly.
        quat = quaternion.from_float_array(np.random.rand(10, 4))
        t = np.random.rand(10, 3)
        p = Pose(quat, t)
        trans_mat = np.tile(np.eye(4), (10, 1, 1))
        trans_mat[..., :3, :3] = quaternion.as_rotation_matrix(quat)
        trans_mat[..., :3, 3] = t

        vec = np.random.rand(10, 4)
        vec[:, 3] = 1
        tmat_rot_vec = np.einsum('ijk,ik->ij', trans_mat, vec)[:, :3]
        p_rot_vec = p.transform_vecs(vec[:, :3])
        self.assertTrue(np.allclose(tmat_rot_vec, p_rot_vec))
        self.assertTrue(np.allclose(p.as_trans_mat(), trans_mat))
        q = Pose.from_trans_mat(trans_mat)
        self.assertTrue(np.allclose(trans_mat, q.as_trans_mat()))

    def test_pose_conversions(self):
        """Test common pose conversions works
        """
        # Test that converting to rotation matrix and translation works.
        quat = quaternion.from_float_array(np.random.rand(4))
        t = np.random.rand(3)
        p = Pose(quat, t)
        r, t_conv = p.as_rmat_pos()
        self.assertTrue(np.allclose(r, quaternion.as_rotation_matrix(quat)))
        self.assertTrue(np.allclose(t, t_conv))

        q = Pose.from_rmat_pos(r, t_conv)
        r_reconv, t_reconv = q.as_rmat_pos()
        self.assertTrue(np.allclose(r, r_reconv))
        self.assertTrue(np.allclose(t, t_reconv))

        rvec, t_rvec = p.as_rvec_pos()
        self.assertTrue(np.allclose(rvec, quaternion.as_rotation_vector(quat)))
        self.assertTrue(np.allclose(t_rvec, t))

        s = Pose.from_rvec_pos(rvec, t_rvec)
        r, t_conv = s.as_rmat_pos()
        self.assertTrue(np.allclose(r, quaternion.as_rotation_matrix(quat)))
        self.assertTrue(np.allclose(t, t_conv))

    def test_private_funcs(self):
        """Test some private functions of the Pose class
        """
        quat = quaternion.from_float_array(np.random.rand(4))
        t = np.random.rand(3)
        p = Pose(quat, t)
        # Let's break rotation
        p._rotation = "Dog"

        def cause_error():
            p._check_consistency()
        self.assertRaises(AssertionError, cause_error)

        q = p._get_standard_rotation(np.random.rand(4))
        self.assertTrue(isinstance(q, np.quaternion))

        # Wrong number of dimensions
        def cause_error():
            p._get_standard_rotation(np.random.rand(4, 4, 1))
        self.assertRaises(ValueError, cause_error)

        # Wrong type
        def cause_error():
            p._get_standard_rotation("dog")
        self.assertRaises(TypeError, cause_error)

    def test_indexing(self):
        """Test that indexing a Pose works.
        """
        quat = quaternion.from_float_array(np.random.rand(10, 4))
        t = np.random.rand(10, 3)
        p = Pose(quat, t)

        q = p[1:5]
        self.assertTrue(len(q.rotation) == 4)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
