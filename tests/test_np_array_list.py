import unittest
import numpy as np

from framegraph.np_array_list import NpArrayList

class TestNpArrayList(unittest.TestCase):

    def test_one_d_array(self):

        arr = NpArrayList((10, 3))
        arr[:] = 20
        arr.append([3,4,5])
        x = arr[:11]
        y = np.zeros(x.shape)
        print(y)
        print(arr)
        print(arr + y)


if __name__ == '__main__':
    unittest.main()