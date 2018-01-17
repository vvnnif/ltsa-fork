import unittest
from nose.tools import with_setup, nottest
import numpy as np
import ltsa


class UtilsTestCase(unittest.TestCase):
    """
    Test utils module.
    """
    def setUp(self):

        num_rows = 300
        self.mat1 = np.random.rand(num_rows, 2)
        self.mat2 = np.random.rand(num_rows, 2)
        self.mat3 = np.random.rand(num_rows, 2)

        self.matrix = np.hstack([self.mat1, np.ones((num_rows, 2)),
                                 self.mat2, 2*np.ones((num_rows, 1)),
                                 self.mat3, -1*np.ones((num_rows, 3))])

        self.MatrixPre, dictOut = ltsa.utils.preprocessing.pre(self.matrix)
        self.MatrixPost = ltsa.utils.preprocessing.post(self.MatrixPre, dictOut)

    def tearDown(self):
        self.matrix = None
        self.MatrixPre = None
        self.MatrixPost = None

    @with_setup(setUp, tearDown)
    def test_util_invertible(self):
        """
        test_util_invertible:                 Normalising invertible on features.
        """
        assert(np.allclose(self.matrix, self.MatrixPost))

    @with_setup(setUp, tearDown)
    def test_util_remove_constant(self):
        """
        test_util_remove_constant:            Remove the constant columns.
        """
        reduced, _, _ = ltsa.utils.preprocessing._remove_constant(self.matrix)
        true_reduced = np.hstack([self.mat1, self.mat2, self.mat3])

        assert(np.allclose(true_reduced, reduced))

    @with_setup(setUp, tearDown)
    def test_util_add_constant(self):
        """
        test_util_add_constant:               Add back the constant columns.
        """
        reduced, index, constants = ltsa.utils.preprocessing._remove_constant(self.matrix)
        matrix_recovered = ltsa.utils.preprocessing._add_constant(reduced, index, constants)

        assert(np.allclose(matrix_recovered, self.matrix))

if __name__ == '__main__':
    print "Running unit tests for UtilsTestCase"
    unittest.main()
