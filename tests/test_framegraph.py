import time
import unittest
import warnings
import os
import numpy as np

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
        if os.environ.get('SLOW_COMPUTER', False):
            self.assertTrue(iter_time < 1e-3)  # pragma: no cover
        else:
            self.assertTrue(iter_time < 1e-4)  # pragma: no cover

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

    def test_bad_callback(self):
        """Test that a callback with the wrong number of parameters throws an
        error.
        """
        fg = FrameGraph()
        fg.add_node("frame")
        j1 = Joint.revolute(np.array([1, 0, 0]))

        def cause_error():
            fg.add_edge("world", "frame", j1, np.array([1.0, 2.0]))

        self.assertRaises(RuntimeError, cause_error)

    def test_no_path(self):
        """Test that if no path exists from target to source, an error is
        raised.
        """
        fg = FrameGraph()
        fg.add_node("frame")

        def cause_error():
            fg.get_relative_transform("world", "frame")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertRaises(RuntimeError, cause_error)

        def cause_error():
            fg.register_transform("world", "frame")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertRaises(RuntimeError, cause_error)

    def test_long_path(self):
        """Test that making long paths (more than 16 edges) works correctly as
        well.
        """
        fg = FrameGraph("0")
        for i in range(1, 20):
            fg.add_node(f"{i}")
            axis = np.array([0, i, 2 * i], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
            cb = Joint.revolute(axis)
            fg.add_edge(f"{i-1}", f"{i}", cb, np.array([0.0]))

        f, j = fg.get_relative_transform("0", "19", ret_jac=True)

        self.assertTrue(f.shape == (4, 4))
        self.assertTrue(j.shape == (4, 4, 19))

    def test_path_rerouted(self):
        """Test that a path properly re-routes if a shorter path is introduced
        by adding in an edge.
        """
        fg = FrameGraph("A")
        fg.add_node("B")
        fg.add_node("C")

        j1 = Joint.revolute(np.array([1, 0, 0]))
        j2 = Joint.cylindrical(np.array([0, 1, 0]))
        fg.add_edge("A", "B", j1, np.array([0.0]))
        fg.add_edge("B", "C", j2, np.array([0.0, 0.1]))

        _, j = fg.get_relative_transform("A", "C", ret_jac=True)
        self.assertTrue(j.shape == (4, 4, 3))

        fg.add_edge("A", "C", j1, np.array([0.0]))
        _, j = fg.get_relative_transform("A", "C", ret_jac=True)
        self.assertTrue(j.shape == (4, 4, 1))

        fg = FrameGraph("A")
        fg.add_node("B")
        fg.add_node("C")
        fg.add_node("D")
        fg.add_edge("A", "B", j1, np.array([0.0]))
        fg.add_edge("A", "C", j1, np.array([0.0]))
        fg.add_edge("B", "D", j1, np.array([0.0]))
        fg.add_edge("C", "D", j2, np.array([0.0, 0.1]))
        param_es, _ = fg.get_params("A", "D")
        _, j = fg.get_relative_transform("A", "D", ret_jac=True)
        shape_init = j.shape
        fg.delete_edge(*param_es[1])

        _, j = fg.get_relative_transform("A", "D", ret_jac=True)
        shape_final = j.shape
        self.assertTrue(shape_final != shape_init)

    def test_int_node(self):
        """Test that using integer node ids works.
        """
        fg = FrameGraph("A")
        fg.add_node("B")
        j1 = Joint.revolute(np.array([1, 0, 0]))
        fg.add_edge(0, "B", j1, np.array([0.0]))
        param_es, _ = fg.get_params("A", 1)
        # Forward and backwards edge
        self.assertTrue(len(fg._graph.es) == 2)
        self.assertTrue(param_es[0] == ("A", "B"))


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
