import unittest
import numpy as np

from framegraph.np_array_list import NpArrayList


class TestNpArrayList(unittest.TestCase):

    def test_two_d_array(self):
        """Test simple properties of array list on 2d array
        """
        arr = NpArrayList(np.zeros((10, 3)))

        arr.append([3, 4, 5])
        arr.append([6, 7, 8])
        self.assertTrue(arr.size == 36)
        self.assertTrue(np.allclose(arr[-1].data, [6, 7, 8]))
        self.assertTrue(arr.shape == (12, 3))


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
