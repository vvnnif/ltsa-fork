import h5py
import scipy.io as spio
import unittest
from nose.tools import with_setup, nottest
import numpy as np

import ltsa
from sklearn import manifold


class PreImageTestCase(unittest.TestCase):
    """
    Test pre-image from LocalTangentSpaceAlignment.

    This class tests the accuracy of the pre-image mapping on our training data. These tests do not guarantee the
    accuracy of the pre-image in test cases, but check the pre-image behaves as expected, and training data is recovered
    with high accuracy.
    """
    def setUp(self):

        switch = 0
        if switch:
            fTr = spio.loadmat('ltsa/testing/baseline/KLRF_train75.mat')
            Output = fTr['output']
        else:
            fTr = h5py.File('ltsa/testing/baseline/Train_N400.mat')
            self.Output = fTr['output_norm'].value.T

        self.OutputPre, dictOut = ltsa.utils.preprocessing.pre(self.Output)
        self.OutputPost = ltsa.utils.preprocessing.post(self.OutputPre, dictOut)

        self.k = 25
        self.d = 15

        self.fun = self.norm_error                    # Using handle allows us to easily change the error function used.

    def tearDown(self):
        self.Output = None
        self.OutputR = None
        self.OutputF = None

    @nottest
    def error(self, mat1, mat2, fun):
        """
        Wrapper for chosen error function
        :param mat1:      matrix:               first matrix
        :param mat2:      matrix:               second matrix
        :param fun:       function handle:      error function handle
        :return:          int:                  error
        """
        assert(mat1.shape == mat2.shape)
        return fun(mat1, mat2)

    @nottest
    def MAPEerror(self, mat1, mat2):
        """
        Mean absolute percentage error
        :param mat1:
        :param mat2:
        :return:
        """
        assert(mat1.shape == mat2.shape)
        n, _ = mat1.shape
        ape = 0
        for i in range(n):
            ape += np.linalg.norm((mat1[i, :] - mat2[i, :]) / mat1[i, :])
        return ape / n

    @nottest
    def norm_error(self, mat1, mat2):
        """
        Normalised error
        :param mat1:
        :param mat2:
        :return:
        """
        return np.mean(np.linalg.norm((mat1 - mat2) / np.linalg.norm(mat1)))

    @with_setup(setUp, tearDown)
    def test_preimage_util_reconstruction(self):
        """
        test_preimage_util_reconstruction:    Test the pre-image on post processed latent variables.
        """
        delta = 1e-10

        model_orig = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d)
        model_orig.solve()

        latent_r, dictLat = ltsa.utils.preprocessing.pre(model_orig.t)
        latent_f = ltsa.utils.preprocessing.post(latent_r, dictLat)

        error = self.error(model_orig.t, latent_f, self.fun)
        #print(error)
        assert(error < delta)

    @with_setup(setUp, tearDown)
    def test_preimage_rawtrain(self):
        """
        test_preimage_rawtrain:               Pre-image for features against true + sklearn benchmark.
        """
        delta = 0.05         # this will have higher error as it is normalised.

        model_orig = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d)
        model_orig.solve()
        preimage = model_orig.pre_image(model_orig.t)

        # benchmark
        clf = manifold.LocallyLinearEmbedding(self.k, self.d, method='ltsa')
        _ = clf.fit_transform(self.Output)

        err = self.error(self.Output, preimage, self.fun)
        #print(err, clf.reconstruction_error_)
        assert(err < max(delta, clf.reconstruction_error_))

    @with_setup(setUp, tearDown)
    def test_preimage_normtrain(self):
        """
        test_preimage_normtrain:              Pre-image for normalised features against true + sklearn benchmark.
        """
        delta = 0.05

        model_pre = ltsa.LocalTangentSpaceAlignment(self.OutputPre, self.k, self.d)
        model_pre.solve()
        preimage = model_pre.pre_image(model_pre.t)

        # benchmark
        clf = manifold.LocallyLinearEmbedding(self.k, self.d, method='ltsa')
        _ = clf.fit_transform(self.OutputPre)

        err = self.error(self.OutputPre, preimage, self.fun)
        # print(err, clf.reconstruction_error_)
        assert (err < max(delta, clf.reconstruction_error_))

    @with_setup(setUp, tearDown)
    def test_preimage_normlatenttrain(self):
        """
        test_preimage_normlatenttrain:        Numerical instability from numerical errors introduced by normalising.
        """
        delta = 0.05

        manifold_model = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d)
        manifold_model.solve()

        latent_r, dictLat = ltsa.utils.preprocessing.pre(manifold_model.t)
        latent_f = ltsa.utils.preprocessing.post(latent_r, dictLat)

        preimage = manifold_model.pre_image(manifold_model.t)
        preimageNorm = manifold_model.pre_image(latent_f)

        err = self.error(preimage, preimageNorm, self.fun)
        #print(err)
        assert(err < delta)

    @with_setup(setUp, tearDown)
    def test_preimage_testlatent(self):
        """
        test_preimage_testlatent:             Unseen test point. We won't know if this is accurate.
        """
        errors = []

        manifold_model = ltsa.LocalTangentSpaceAlignment(self.Output, self.k, self.d)
        manifold_model.solve()

        try:
            _ = manifold_model.pre_image(np.random.normal(0,1,(100,self.d)))
        except Exception as e:
            errors.append('test_preimage1: '+str(e))

        # assert no error message has been registered, else print messages
        self.assertTrue(not errors, "errors occured:\n{}".format("\n".join(errors)))


if __name__ == '__main__':
    print("Running unit tests for PreImageTestCase (this may take a while)")
    unittest.main()
