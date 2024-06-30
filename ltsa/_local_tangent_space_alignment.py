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
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=25)

__all__ = ['LocalTangentSpaceAlignment']

class LocalTangentSpaceAlignment(object):
    """
    A class for Local Tangent Space Alignment.
    """

    # public
    _X = None                                                                 # Features
    _k = None                                                                 # Number of nearest neighbours
    _new_dim = None                                                           # Manifold dimension
    _T = None                                                                 # Latent variables
    # private
    _seed = None                                                              # seed, currently unused.
    _cache_q = [None, None]                                                   # [do we cache?, have we cached?]
    _Q = None                                                                 # where we cache Q too.
    _neighbourhoods = None                                                    # neighbourhoods of the tangent spaces
    _theta = None
    _theta_pinv = None
    _L = None
    _Linv = None

    @property
    def x(self):
        """
        Get the features.
        :return:        matrix:      self.n*self.org_dim
        """
        return self._X

    @property
    def org_dim(self):
        """
        Get the dimension of the feature space.
        :return:        int:         number of features
        """
        return self._X.shape[1]

    @property
    def n(self):
        """
        Get the number of feature vectors.
        :return:        int:         number of samples
        """
        return self._X.shape[0]

    @property
    def k(self):
        """
        Get the number of neighbours we use.
        :return:        int:         number of neighbours
        """
        return self._k

    @property
    def new_dim(self):
        """
        Get the dimension of the latent space.
        :return:        int:        number of latent features
        """
        return self._new_dim

    @property
    def t(self):
        """
        Get the latent points
        :return:        matrix:      self.n*self.new_dim
        """
        return self._T

    # getters for computations we don't (always) cache    #TODO: turn into properties?  i.e. xi = property(get_xi)
    def get_ti(self, cluster=0):
        """
        Get the latent vectors belonging to cluster i.
        :param cluster:       int:      the cluster index
        :return:              matrix:   self.k*self.new_dim
        """
        return self.t[self._neighbourhoods[cluster, :], :]

    def get_xi(self, cluster=0):
        """
        Get the features vectors belonging to cluster i.
        :param cluster:       int:      the cluster index
        :return:              matrix:   self.k*self.org_dim
        """
        return self.x[self._neighbourhoods[cluster,:], :]

    def get_qi(self, cluster=0):
        """
        A wrapper around self.solve_qi for caching, or recomputing with constrained singular vectors.
        :param cluster:       int:      the cluster index
        :return:              matrix:   self.d*self.org_dim
        """
        if self._cache_q[0] is True:                                                                # cache
            if self._cache_q[1] == False:                                                           # save
                self._Q = np.zeros((self.n, self.org_dim, self.new_dim))
                for i in range(self.n):
                    self._Q[i, :, :] = self.solve_qi(cluster=i)
                self._cache_q[1] = True
                qi = np.squeeze(self._Q[cluster, :, :])
            else:                                                                                   # load
                assert(self._cache_q[1]==True)
                qi = np.squeeze(self._Q[cluster, :, :])
        else:                                                                                       # don't cache
            assert(self._cache_q[0] is False)
            qi = self.solve_qi(cluster=cluster)

        return qi

    def solve_qi(self, cluster):
        """
        Get the left singular vectors Q_i.
        :param cluster:       int:      the cluster index
        :return:              matrix:   self.d*self.org_dim

        This involves computation of Qi using the SVD. We do not want to repeat this each time we need Qi (including
        using the pre-image map), but as this is a large matrix (num_data*org_dim*new_dim) storing is memory intensive.
        We therefore allow the option to cache these where the system and data allows.
        """

        #np.random.seed(int(self._seed))                              # Set np seed (doesn't resolve flipping)

        xi = self.get_xi(cluster)
        xibar = np.mean(xi, axis=0)
        xic = xi - np.matlib.repmat(xibar, self.k, 1)                 # Centering neighbor point matrix, xic=xi(I-ee'/k)

        qi, s, vt = scipy.sparse.linalg.svds(xic.T, self.new_dim)     # Compute Qi, the d left singular vector of Xkc.
        qi = np.fliplr(qi)                                            # Increasing order=>flip. Consistency with MATLAB.

        # Only unique up to sign, replicable by constraining second row to be positive.
        sign = np.sign(qi[1, :])
        for col in range(qi.shape[1]):
            if sign[col] == -1:
                qi[:, col] = -qi[:, col]

        return qi

    def __init__(self, x, k, new_dim, theta = None, pinv = None, neighbors = None, name='LTSA', cacheQ=False, skip_data = False):
        """
        Initialize the object.
        """
        self._seed = time.time()
        self._k = k
        self._new_dim = new_dim
        self.__name__ = name
        self._cache_q = [cacheQ, False]

        if skip_data is False:
            self._X = x
            nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(x)
            distances, indices = nbrs.kneighbors(x)
            self._neighbourhoods = indices
        else:
            self._neighbourhoods = neighbors
            self._theta = theta
            self._pinv = pinv

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
        if theta_and_pinvs is None:
            self._solve_theta()
            
        self._solve_wt()

        self._solve_l()

    def _solve_theta(self):
        """
        Solve for the local co-ordinates, theta \in \mathbb{R}^{num_data*k*new_dim).

        :return:    theta         the collection of matrices of local co-ordinates for each neighbourhood.
                    theta_pinv    the collection of Moore-Penrose generalized inverses of the thetai.
        """

        self._qsecond = np.zeros((self.n, self.new_dim))
        theta = np.zeros((self.n, self.new_dim, self.k))
        theta_pinv = np.zeros((self.n, self.k, self.new_dim))
        for i in range(self.n):
 
            qi = self.get_qi(cluster=i)

            xi = self._X[self._neighbourhoods[i, :], :]
            xibar = np.dot(xi.T, (np.ones((self.k, self.k)) / self.k))
            xic = xi.T - xibar
            # compute theta and psuedo inverse, refer to theta and theta+
            theta[i, :, :] = np.dot(qi.T, xic)
            theta_pinv[i, :, :] = np.linalg.pinv(np.squeeze(theta[i, :, :]))

        self._theta = theta
        self._theta_pinv = theta_pinv

    def _solve_wt(self):
        """
        Perform the global alignment of the Local Tangent Space Alignment procedure.
            We do not cache Wi to save memory. Instead it is calculated as needed. This does not increase
            computational cost greatly.

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

            b[np.ix_(self._neighbourhoods[i, :],self._neighbourhoods[i, :])] = b[np.ix_(self._neighbourhoods[i, :],
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

        Run after solving for theta -> _solve_theta()
        """
        assert self._theta is not None
        assert self._theta_pinv is not None

        l = np.zeros((self.n, self.new_dim, self.new_dim))
        linv = np.zeros((self.n, self.new_dim, self.new_dim))
        for i in range(self.n):
            tit = np.transpose(self.get_ti(cluster=i))
            l[i, :, :] = tit.dot(np.eye(self.k, self.k) - np.ones((self.k, self.k))/self.k).dot(self._theta_pinv[i, :, :])
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
            print("Trying to cache " + str(memory_req)+"Gb of predictions.")

        # find the nearest neighbour to t_pred out of self.t
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.t)
        distances, TangentSpaceIndex = nbrs.kneighbors(t_pred)

        # Loops over predicting for each vector in t_pred
        Y_pred = np.zeros((n_pred, self.org_dim))
        for testi in range(n_pred):
            cluster = np.int(TangentSpaceIndex[testi])

            # Find the neighbourhood centered at the nearest neighbour to t_pred(testi).
            xi = self.get_xi(cluster=cluster)
            xibar = np.mean(xi, axis=0)

            # compute Qi, d left singular vector of the centered neighbor point matrix, xic=xi(I-ee'/k) (up to sign)
            qi = self.get_qi(cluster=cluster)

            ti = self.get_ti(cluster=cluster)
            tibar = np.mean(ti, axis=0)

            part2 = np.dot(qi, np.squeeze(self._Linv[cluster, :, :]))
            part3 = (t_pred[testi, :][None,:] - tibar).T

            Y_pred[testi, :] = (xibar[:,None] + np.dot(part2, part3)).ravel()

        return Y_pred

    def pre_image_GaussianLatent(self):

        return NotImplementedError

