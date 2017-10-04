import unittest
from nose.tools import with_setup, nottest
import numpy as np

import ltsa


class UtilsTestCase(unittest.TestCase):
    """
    Test utils.
    """
    def setUp(self):

        num_rows = 3
        self.mat1 = np.random.rand(num_rows, 2)
        self.mat2 = np.random.rand(num_rows, 2)
        self.mat3 = np.random.rand(num_rows, 2)

        self.matrix = np.hstack([self.mat1, np.ones((num_rows, 2)),
                                 self.mat2, 2*np.ones((num_rows, 1)),
                                 self.mat3, -1*np.ones((num_rows, 3))])

    def tearDown(self):
        self.matrix = None

    @with_setup(setUp, tearDown)
    def test_removes_all_constants(self):
        """
        Test we correctly remove the constant columns
        """
        reduced, _, _ = ltsa.utils.preprocessing._remove_constant(self.matrix)
        true_reduced = np.hstack([self.mat1, self.mat2, self.mat3])

        assert(np.allclose(true_reduced, reduced))

    @with_setup(setUp, tearDown)
    def test_adds_all_constants(self):
        """
        Test we correctly add back the constant columns
        """
        reduced, index, constants = ltsa.utils.preprocessing._remove_constant(self.matrix)
        matrix_recovered = ltsa.utils.preprocessing._add_constant(reduced, index, constants)

        assert(np.allclose(matrix_recovered, self.matrix))

    @with_setup(setUp, tearDown)
    def test_invertable(self):
        """
        Test if normalising and unnormalising recovers original matrix
        """
        matrix_reduced, dict = ltsa.utils.preprocessing.pre(self.matrix)
        matrix = ltsa.utils.preprocessing.post(matrix_reduced, dict)

        assert(np.allclose(matrix, self.matrix))

if __name__ == '__main__':
    print "Running unit tests for UtilsTestCase"
    unittest.main()
