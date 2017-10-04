import scipy.io as spio
import h5py
import unittest
from nose.tools import with_setup, nottest
import numpy as np

import ltsa


class LTSATestCase(unittest.TestCase):
    """
    Test pre-image from LocalTangentSpaceAlignment.
    """
    def setUp(self):

        switch = 1
        if switch:
            fTr = spio.loadmat('ltsa/testing/baseline/KLRF_train75.mat')
            Output = fTr['output']
        else:
            fTr = h5py.File('baseline/Train_N400.mat')
            Output = np.reshape(fTr['output'].value.T, [400, 26 * 26 * 26])

        self.Output = Output
        OutputR, dictOut = ltsa.utils.preprocessing.pre(Output)
        self.OutputR = OutputR
        OutputF = ltsa.utils.preprocessing.post(OutputR, dictOut)
        self.OutputF = OutputF

        self.k = 10
        self.d = 5

    def tearDown(self):
        self.Output = None
        self.OutputR = None
        self.OutputF = None
        self.k = None
        self.d = None

    @with_setup(setUp, tearDown)
    def test_temp(self):

        ManifoldModel = ltsa.LocalTangentSpaceAlignment(self.OutputR, self.k, self.d)
        ManifoldModel.solve()

if __name__ == '__main__':
    print "Running unit tests for LTSAModuleTestCase (this may take a while)"
    unittest.main()
