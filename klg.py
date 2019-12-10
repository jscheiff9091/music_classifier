from __future__ import division
import numpy as np
import math


def compute_klg_dist(mu1, cov1, mu2, cov2, gamma=10):
    """
    Computes the klg distance between the two tracks
    Each track is summarized by its mean and covariance
    We assume K=12

    Params
    ------
    mu1 : vector (12 x 1)
        The mean of features (pcp's or mfcc's) of track 1
    cov1 : matrix (12 x 12)
        The cov of the features of track 1
    mu2 : vector (12 x 1)
    cov2 : matrix (12 x 12)

    Returns
    -------
    float
        The distance between the two tracks
    """

    K = 12
    icov1 = np.linalg.inv(cov1)
    icov2 = np.linalg.inv(cov2)
    ico1_co2 = np.dot(icov1, cov2)
    ico2_co1 = np.dot(icov2, cov1)
    mu_diff = mu1 - mu2
    its_a_dot_product = np.dot((icov1 + icov2), mu_diff)
    epsilon = np.finfo(float).eps
    
    kl = .5 * (np.trace(ico1_co2 + ico2_co1) + 0.5*np.dot(mu_diff.T, its_a_dot_product)) -K
    d = 1 - math.exp(-gamma/(abs(kl) + epsilon))
    return d


def compute_d_bar(distance_matrix):
    '''
    Compute average distance between each genre.

    Params
    ------
    distances : matrix {150 x 150}
        distance between each song and every other song

    Returns
    ------
    d_bar : {6 x 6} 
        Average distance between songs in one genre to
        to songs in every other genre
    '''

    d_bar = np.zeros((6,6))
    diag = 325
    off_diag = 25**2

    for i in range(6):
        start_i = i*25
        end_i = start_i + 25
        for j in range(6):
            start_j = j*25
            end_j = start_j + 25

            #get distances between two genres
            dist_subset = distance_matrix[start_i:end_i, start_j:end_j]

            if i == j:                            #comparing same genre -> avoid double counting
                avg_dist = np.sum(dist_subset) / (diag * 2)
            else:
                avg_dist = np.sum(dist_subset) / off_diag

            d_bar[i, j] = avg_dist

    return d_bar