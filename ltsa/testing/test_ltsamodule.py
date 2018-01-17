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
    def test_ltsamodule_cacheQ_equalq(self):
        """
        test_ltsamodule_cacheQ:               Test if caching and recomputing Q_i are consistent
        """
        model_cached = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d, cacheQ=True)
        model_cached.get_qi()                # to actually cache it.
        assert(model_cached._Q is not None)

        model_notcached = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d, cacheQ=False)
        assert(model_notcached._Q is None)

        for i in range(model_cached.n):
            assert( np.allclose(model_notcached.get_qi(cluster = i), model_cached.get_qi(cluster=i)) )

    @with_setup(setUp, tearDown)
    def test_ltsamodule_cacheQ_equalresult(self):
        """
        test_ltsamodule_cacheQ_equalresult:   Test if caching and recomputing Q_i leads to the same model.
        """
        model_cached = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d, cacheQ=True)
        model_cached.solve()
        assert (model_cached._Q is not None)

        model_notcached = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d, cacheQ=False)
        model_notcached.solve()
        assert (model_notcached._Q is None)

        assert( np.allclose(model_cached.t, model_notcached.t) )
        assert( np.allclose(model_cached._L, model_notcached._L) )
        assert (np.allclose(model_cached._Linv, model_notcached._Linv))
        assert (np.allclose(model_cached._theta, model_notcached._theta))
        assert (np.allclose(model_cached._theta_pinv, model_notcached._theta_pinv))


    @with_setup(setUp, tearDown)
    def test_ltsamodule_getters(self):
        """
        test_ltsamodule_getters:              Test getters and setters.
        """
        # TODO: is this one necessary?
        pass


if __name__ == '__main__':
    print("Running unit tests for LTSAModuleTestCase (this may take a while)")
    unittest.main()
