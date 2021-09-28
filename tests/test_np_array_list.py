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

    def test_set_extend_axis(self):
        """Test that trying to set the extend axis to an invalid value raises
        an error.
        """
        arr = NpArrayList(np.zeros((10, 3)))
        self.assertTrue(arr.extend_axis == 0)

        def cause_error():
            arr.extend_axis = 2
        self.assertRaises(ValueError, cause_error)

        arr.extend_axis = 1
        arr.append(np.random.rand(10))
        self.assertTrue(arr.shape == (10, 4))

    def test_set_bad_data(self):
        """Test that setting the data to a non-array throws an error
        """
        arr = NpArrayList(np.zeros((10, 3)))
        arr.extend_axis = 1

        def cause_error():
            arr.data = 3
        self.assertRaises(ValueError, cause_error)

        def cause_error():
            arr.data = np.random.rand(10)
        self.assertRaises(ValueError, cause_error)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
