import numpy as np
import os
from scipy.special import bernoulli
from scipy import linalg
from scipy.linalg import block_diag
import alphashape
import subprocess
import pickle
import matplotlib
import matplotlib.pyplot as plt
from manifpy import SE3, SO3, SE3Tangent

#This file is the work of Brossard slightly modified, source: https://github.com/CAOR-MINES-ParisTech/3d-icp-cov

class SigmaPoints:
    def __init__(self, n, alpha, beta, kappa):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1

    def sigma_points(self, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the  covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        """
        n = self.n

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = np.linalg.cholesky((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[1: n+1] = U
        sigmas[n+1:] = -U
        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


class UTSE3:
    def __init__(self, n, alpha, beta, kappa, Q_prior):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()
        self.sp = SigmaPoints(n, alpha, beta, kappa)
        self.Q_prior = Q_prior
        self.sps = self.sp.sigma_points(Q_prior)
        self.n_mc = 0

    def unscented_transform(self, sigmas):
        x = np.dot(self.Wm, sigmas)
        P = np.dot(sigmas.T, np.dot(np.diag(self.Wc), sigmas))
        return x, P

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

    def unscented_transform_se3(self, T_ut):
        T_inv = T_ut[0].inverse() #SE3.inv(T_ut[0])
        sp_new = np.zeros((13, 6))
        for n in range(13):
            sp_new[n] = (T_ut[n]*T_inv).log().coeffs() #SE3.log(SE3.mul(T_ut[n], T_inv))  # xi = log( T_sp * T_hat^{-1} )
        sp_mean, cov_ut = self.unscented_transform(sp_new)
        T_mean = (SE3Tangent(sp_mean).exp()*T_ut[0]).transform() #SE3.mul(SE3.exp(sp_mean), T_ut[0])
        cov_full = self.Wc[1]*12*np.cov(np.hstack((self.sps, sp_new - sp_mean)).T)
        cov_cross = cov_full[:6, 6:]
        return T_mean, sp_mean, cov_ut, cov_cross