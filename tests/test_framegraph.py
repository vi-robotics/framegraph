import unittest
import numpy as np

from framegraph.framegraph import FrameGraph


class TestFrameGraph(unittest.TestCase):

    def test_double_node(self):
        """Test that a two node frame graph returns proper transforms
        """
        fg = FrameGraph()
        fg.add_node("frame")


if __name__ == '__main__':
    unittest.main()
