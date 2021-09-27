import time
import unittest
import numpy as np
import quaternion
import jax.numpy as jnp

from framegraph.framegraph import FrameGraph
from framegraph.joint import Joint


class TestFrameGraph(unittest.TestCase):

    def test_performance(self):
        """Test that performance of the frame graph is sufficient
        """
        trials = 1000

        fg = FrameGraph()
        fg.add_node("frame")
        fg.add_node("end_effector")
        self.assertTrue(len(fg._graph.vs) == 3)

        j1 = Joint.revolute(np.array([1, 0, 0]))
        j2 = Joint.cylindrical(np.array([0, 1, 0]))
        fg.add_edge("frame", "world", j1, np.array([0.0]))
        fg.add_edge("end_effector", "frame", j2, np.array([0.1, 0.2]))

        fg.register_transform("world", "end_effector")
        fg.get_relative_transform("world", "end_effector")

        t0 = time.time()
        for i in range(trials):
            fg.get_relative_transform("world", "end_effector")
        tf = time.time()
        iter_time = (tf - t0) / trials
        self.assertTrue(iter_time < 1e-4)

    def test_get_params(self):
        """Test that getting parameters from a framegraph works
        """
        target_named_edges = [["world", "frame"], ["frame", "end_effector"]]
        target_params = [np.array([0.0]), np.array([0.1, 0.2])]
        fg = FrameGraph()
        fg.add_node("frame")
        fg.add_node("end_effector")
        j1 = Joint.revolute(np.array([1, 0, 0]))
        j2 = Joint.cylindrical(np.array([0, 1, 0]))
        fg.add_edge("frame", "world", j1, target_params[0])
        fg.add_edge("end_effector", "frame", j2, target_params[1])

        named_edges, ps = fg.get_params("world", "end_effector")

        for edge, targ_e in zip(named_edges, target_named_edges):
            for node, targ_node in zip(edge, targ_e):
                self.assertTrue(node == targ_node)

        for param, targ_parm in zip(ps, target_params):
            self.assertTrue(np.allclose(param, targ_parm))

    def test_already_present_node(self):
        """Test that an error is raised for a node already present in the 
        graph.
        """
        fg = FrameGraph()
        fg.add_node("frame")

        def cause_error():
            fg.add_node("frame")
        self.assertRaises(ValueError, cause_error)

    def test_bad_params(self):
        """Test that an edge added with bad params raises an error
        """
        fg = FrameGraph()
        fg.add_node("frame")
        cb = Joint.fixed(np.eye(4))

        # Check a non-array
        def cause_error():
            fg.add_edge("world", "frame", cb, 3)
        self.assertRaises(ValueError, cause_error)

        cb = Joint.cylindrical(np.array([0, 1, 0]))

        # Check a non-1d array
        def cause_error():
            fg.add_edge("world", "frame", cb, np.array([[0.0, 0.1]]))
        self.assertRaises(ValueError, cause_error)

    def test_fixed_pose(self):
        """Test that using a parameter-free pose works correctly in the graph
        """
        fg = FrameGraph()
        fg.add_node("frame")
        fg.add_node("end_effector")
        j1 = Joint.revolute(np.array([1, 0, 0]))
        j2 = Joint.fixed(np.eye(4))
        fg.add_edge("world", "frame", j1, np.array([1.0]))
        fg.add_edge("frame", "end_effector", j2, None)
        fg.get_relative_transform("world", "end_effector", ret_jac=True)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
