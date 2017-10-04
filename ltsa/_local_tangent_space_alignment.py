"""
A class for Local Tangent Space Alignment

Author:
    Charles Gadd

Date:
    20/09/2017

"""

import scipy
from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt

__all__ = ['LocalTangentSpaceAlignment']


class LocalTangentSpaceAlignment(object):
    """
    A class for Local Tangent Space Alignment.
    """

    # public
    _X = None
    _k = None
    _new_dim = None
    _T = None
    # private
    _neighbourhoods = None
    _theta = None
    _theta_pinv = None
    _L = None
    _Linv = None
    _qsign = None

    @property
    def x(self):
        """
        Get the data.
        """
        return self._X

    @property
    def org_dim(self):
        """
        Get the dimension of the feature space.
        """
        return self._X.shape[1]

    @property
    def n(self):
        """
        Get the number of feature vectors.
        """
        return self._X.shape[0]

    @property
    def k(self):
        """
        Get the number of neighbours we use.
        """
        return self._k

    @property
    def new_dim(self):
        """
        Get the dimension of the latent space.
        """
        return self._new_dim

    @property
    def t(self):
        """
        Get the latent points
        """
        return self._T

    def __init__(self, x, k, new_dim, name='LTSA'):
        """
        Initialize the object.
        """
        self._X = x
        self._k = k
        self._new_dim = new_dim
        self.__name__ = name

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(x)
        distances, indices = nbrs.kneighbors(x)
        self._neighbourhoods = indices

    def __str__(self):
        """
        Return a string representation of the object.
        """
        s = "\n# name: " + self.__name__ + '\n'
        s += "Number of training points: " + str(self.n) + "\n"
        s += "Number of features: " + str(self.org_dim) + "\n"
        s += "Number of neighbours: " + str(self.k) + "\n"
        s += "Number of latent dimensions: " + str(self.new_dim) + "\n"
        if self.t is None:
            s += "T not solved for\n"
        else:
            s += "T solved for\n"
        if self._theta is None or self._theta_pinv is None:
            s += "L & Linv not solved for\n"
        else:
            s += "L & Linv solved for\n"

        return s

    def __call__(self):
        """
        Pre-image of latent point
        """

        return NotImplementedError

    def solve(self):
        """
        Find the latent points
        """
        # First find the local co-ordinates and Moor-Penrose generalized inverse of them
        #   In doing this we discard the Qi matrices as caching is expensive. We can instead recover with theta
        self._solve_theta()

        self._solve_wt()

        self._solve_l()


    def _solve_theta(self):
        """
        Solve for the local co-ordinates, theta \in \mathbb{R}^{num_data*k*new_dim).

        :return:    theta         the collection of matrices of local co-ordinates for each neighbourhood.
                    theta_pinv    the collection of Moore-Penrose generalized inverses of the thetai.

        This involves computation of Qi using the SVD. We do not want to repeat this each time we need Qi (including
        using the pre-image map), but as this is a large matrix (num_data*org_dim*new_dim) storing is memory intensive.
        Instead we discard Qi but cache the theta_i for i=1,...,num_data. Qi can then be recovered by solving a system
        of equations which should be quicker than repeating the SVD.
        """

        theta = np.zeros((self.n, self.new_dim, self.k))
        theta_pinv = np.zeros((self.n, self.k, self.new_dim))
        qifirstsign = np.zeros((self.n, self.new_dim))
        for i in range(self.n):
            # Local points
            xi = np.zeros([self.k, self.org_dim])
            for j in range(self.k):
                xi[j, :] = self._X[self._neighbourhoods[i, j], :]

            # centering the neighbor point matrix
            # xic=xi(I-ee'/k)
            xibar = np.dot(xi.T, (np.ones((self.k, self.k)) / self.k))
            xic = xi.T - xibar

            # compute Qi, the d left singular vector of Xkc.
            qi, _, _ = scipy.sparse.linalg.svds(xic, self.new_dim)
            # these are in increasing order, flip. Dont think this matters though - just keeping consistent with MATLAB.
            qi = np.fliplr(qi)
            qifirstsign[i, :] = (qi[0, :] > 0)

            # compute theta and theta_inverse, refer to theta and theta +
            thetai = np.dot(qi.T, xic)
            thetai_pinv = np.linalg.pinv(thetai)

            # debug
            #if i == 1:
                #print qi
                #print qifirstsign[i, :]
                #print qifirstsign[i, :]
                # plt.hist(xi[0,:])
                # plt.show()
                # print xi[:, :5]
                # print xic[:6, :6]
                # print qi[:6, :]
                # print thetai[:,:5]
                #print thetai_pinv[:, :5]
                # print qi.shape
                # print xic.shape

            theta[i, :, :] = thetai
            theta_pinv[i, :, :] = thetai_pinv

        self._theta = theta
        self._theta_pinv = theta_pinv
        self._qsign = qifirstsign

    def _solve_wt(self):
        """
        Perform the global alignment of the Local Tangent Space Alignment procedure.
            We do not pre-compute Wi to save memory. Instead it is calculated as needed. This does not increase
            computational cost.

        Using this instead of solve() allows the user to find T without computing L & Linv, required for the pre-image
        """

        # compute B
        b = np.zeros((self.n, self.n))
        for i in range(self.n):
            # compute Wi.
            # These computations are independent between neighbourhoods and do not need to be saved to memory.
            wi1 = (np.eye(self.k) - np.ones((self.k, 1)) * np.ones((1, self.k)) / self.k)
            wi2 = np.eye(self.k) - self._theta_pinv[i, :, :].dot(self._theta[i, :, :])
            wi = np.dot(wi1, wi2)

            b[np.ix_(self._neighbourhoods[i, :], self._neighbourhoods[i, :])] = b[np.ix_(self._neighbourhoods[i, :],
                                                                                         self._neighbourhoods[i, :])] \
                                                                                + wi

        # compute T
        eig_vals, eig_vecs = np.linalg.eig(b)
        sort = eig_vals.argsort()
        # Sort smallest -> largest
        eig_vals.sort()
        eig_vecs = eig_vecs[:, sort]
        t = eig_vecs[:, 1:(self.new_dim+1)]

        self._T = t

    def _solve_l(self):
        """
        Solve for the local affine transformation matrix and the inverse for the local tangent space of a neighbourhood.
        """

        # do not run this before running (and saving results from) _solve_theta()
        assert self._theta is not None
        assert self._theta_pinv is not None

        l = np.zeros((self.n, self.new_dim, self.new_dim))
        linv = np.zeros((self.n, self.new_dim, self.new_dim))
        for i in range(self.n):
            ti = self.t[self._neighbourhoods[i, :], :].T
            l[i, :, :] = ti.dot(np.eye(self.k, self.k) - np.ones((self.k, self.k))/self.k).dot(self._theta_pinv[i, :, :])
            linv[i, :, :] = np.linalg.inv(l[i, :, :])

        self._L = l
        self._Linv = linv

    def pre_image(self, t_pred):
        """
        Find the pre-image in feature space for a matrix of latent vector points.

        :param t_pred:
        """
        assert t_pred.shape[1] == self.new_dim
        n_pred = t_pred.shape[0]

        memory_req = (n_pred*self.org_dim*8) / (10**9)
        if memory_req > 5:
            print "Trying to cache " + str(memory_req)+"Gb of predictions."

        # find the nearest neighbour to t_pred out of self.t
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.t)
        distances, indices = nbrs.kneighbors(t_pred)
        #print self._neighbourhoods.shape

        # Loops over predicting for each vector in t_pred
        Y_pred = np.zeros((n_pred, self.org_dim))
        for i in range(n_pred):
            # we already have the closest point, now we find the neighbourhood around that point.

            #print self.t[self._neighbourhoods[indices[i], :], :6]
            # find the neighbourhood around t_predi -> indices[i]^th tangent space
            xi = np.squeeze(self.x[self._neighbourhoods[indices[i], :], :])
            xibar = np.dot(xi.T, np.ones((self.k, 1)) / self.k)

            si = np.zeros((self.n,self.k))
            for j in range(self.k):
                si[self._neighbourhoods[i, j], j] = 1

            ti = np.dot(self.t.T, si)
            tibar = np.dot(ti, np.ones((self.k, 1)) / self.k)


            # Can we avoid redoing SVD to compute Q. Didn't cache to save memory, perhaps implement memory check?
            #thetai = np.squeeze(self._theta[indices[i]])
            #qi_approx = scipy.sparse.linalg.lsqr(xic.T, thetai.T)
            xic = xi.T - np.matlib.repmat(xibar, 1, self.k)
            # compute Qi, the d left singular vector of Xkc.
            qi, _, _ = scipy.sparse.linalg.svds(xic, self.new_dim)
            qi = np.fliplr(qi)
            # compare the sign of the first rows of earlier (not saved) qi and this version. Up to sign they will be
            # equal due to constraints.
            wrongsign = np.abs((qi[0, :] > 0) - self._qsign[i, :])
            for col in range(qi.shape[1]):
                if wrongsign[col] == 1:
                    qi[:, col] = -qi[:, col]


            part2 = np.dot(qi, np.squeeze(self._Linv[indices[i], :, :]))
            part3 = t_pred[i, :][:, None] - tibar
            Y_pred[i, :] = (xibar + np.dot(part2, part3))[:, 0]


            #if i == 3:
                #print "test"
                #print qi
                #print qi[0, :] > 0
                #print -1* np.abs( (qi > 0) - np.matlib.repmat(self._qsign[i, :], np.shape(qi)[0], 1))
                #print qi
                #print np.squeeze(self._Linv[indices[i], :, :])
                #print part2

                #print np.mean(xi, 0)[:5]
                #print self._neighbourhoods
                #print self.t[:, :4]

                #plt.scatter(range(1913), part2[:, 4])
                #plt.show()
                #print (xibar + np.dot(part2, part3))

                #print xibar
                #print part2
                #print part3
                #print np.dot(part2, part3).shape
                #print Y_pred[i, :5]
                #plt.scatter(range(1913), Y_pred[i, :])
                #plt.show()

        return Y_pred

    def pre_image_GaussianLatent(self):

        return NotImplementedError

