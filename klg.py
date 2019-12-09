import numpy as np


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
    pass