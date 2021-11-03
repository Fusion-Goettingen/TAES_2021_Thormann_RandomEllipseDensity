"""
Author: Kolja Thormann

Contains support functions for the ellipse fusion and test setup
"""

import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy.linalg import slogdet
from scipy.linalg import sqrtm

from sklearn.cluster import DBSCAN

from FusionMethods.constants import *


def rot_matrix(alpha):
    """
    Calculates a rotation matrix based on the input orientation.
    :param alpha:   Input orientation
    :return:        Rotation matrix for alpha
    """
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    if len(rot.shape) == 3:
        return rot.transpose((2, 0, 1))
    else:
        return rot


def to_matrix(alpha, l, w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix.
    :param alpha:   Orientation of the ellipse
    :param l:       Semi-axis length of the ellipse
    :param w:       Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix
    """
    p = 1 if sr else 2
    rot = rot_matrix(alpha)
    if len(rot.shape) == 3:
        lw_diag = np.array([np.diag([l[i], w[i]]) for i in range(len(l))])
        return np.einsum('xab, xbc, xdc -> xad', rot, lw_diag ** p, rot)
    else:
        return np.dot(np.dot(rot, np.diag([l, w]) ** p), rot.T)


def to_matrix_params(alpha, l, w, sr):
    """
    Turn ellipse parameters into a 3D vector containing the matrix or square root matrix elements
    parameter.
    :param alpha:   Orientation of the ellipse
    :param l:       Semi-axis length of the ellipse
    :param w:       Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        3D vector containing diagonal and corner of shape or square root matrix
    """
    rot = rot_matrix(alpha)
    if len(rot.shape) == 3:
        mats = to_matrix(alpha, l, w, sr)
        return np.array([mats[:, 0, 0], mats[:, 0, 1], mats[:, 1, 1]]).T
    else:
        mat = to_matrix(alpha, l, w, sr)
        return np.array([mat[0, 0], mat[0, 1], mat[1, 1]])


def get_ellipse_params(ell):
    """
    Calculate the ellipse semi-axis length and width and orientation based on shape matrix.
    :param ell: Input ellipse as 2x2 shape matrix
    :return:    Semi-axis length, width and orientation of input ellipse
    """
    ellipse_axis, v = np.linalg.eig(ell)
    ellipse_axis = np.sqrt(ellipse_axis)
    ax_l = ellipse_axis[0]
    ax_w = ellipse_axis[1]
    al = np.arctan2(v[1, 0], v[0, 0])

    return al, ax_l, ax_w


def get_ellipse_params_from_sr(sr):
    """
    Calculate ellipse semi-axis length and width and orientation based on the elements of the square root matrix.
    :param sr:  Elements of the square root matrix [top-left, corner, bottom-right]
    :return:    Semi-axis length, width and orientation of input square root matrix
    """
    # calculate shape matrix based on square root matrix elements
    ell_sr = np.array([
        [sr[0], sr[1]],
        [sr[1], sr[2]],
    ])
    ell = np.dot(ell_sr, ell_sr)

    return get_ellipse_params(ell)


def single_particle_approx_gaussian(prior, cov, n_particles, sr=True):
    """
    Calculate the particle density of the prior in square root space and approximate it as a Gaussian.
    :param prior:       Prior in original state space
    :param cov:         Covariance of the prior
    :param n_particles: Number of particles used for particle approximation
    :param sr:          Utilize square root matrix or normal matrix for particles
    :return:            Approximated mean and covariance in square root space and the particle density
    """
    particle = sample_m(prior, cov, False, n_particles)  # sample particles from the prior density

    # transform particles into square root space
    for i in range(n_particles):
        # calculate square root
        mat_sr = to_matrix(particle[i, AL], particle[i, L], particle[i, W], sr)

        # save transformed particle
        particle[i, SR] = np.array([mat_sr[0, 0], mat_sr[0, 1], mat_sr[1, 1]])

    # calculate mean and variance of particle density
    mean_sr = np.sum(particle, axis=0) / n_particles
    var_sr = np.sum(np.einsum('xa, xb -> xab', particle - mean_sr, particle - mean_sr), axis=0) / n_particles
    var_sr += var_sr.T
    var_sr *= 0.5

    return mean_sr, var_sr, particle


def sample_m(mean, cov, shift, amount):
    """
    Create samples from input density.
    :param mean:    Mean with which to sample
    :param cov:     Covariance with which to sample
    :param shift:   Boolean to determine whether to shift the orientation
    :param amount:  Number of samples to be drawn
    :return:        The sample for amount==1 and an array of samples for amount > 1
    """
    # shift mean by 90 degree and switch length and width if demanded
    s_mean = mean.copy()
    if shift:
        save_w = s_mean[W]
        s_mean[W] = s_mean[L]
        s_mean[L] = save_w
        s_mean[AL] += 0.5 * np.pi

    # draw sample
    samp = mvn(s_mean, cov, amount)

    # enforce restrictions
    samp[:, L] = np.maximum(0.1, samp[:, L])
    samp[:, W] = np.maximum(0.1, samp[:, W])
    samp[:, AL] %= 2 * np.pi

    # if only one sample, do not store it in a 1d array
    if amount == 1:
        samp = samp[0]

    return samp


def mmgw_estimate_from_particles(particles):
    """
    Calculate the MMGW estimate of a particle density in ellipse parameter space.
    :param particles:   particle density in ellipse parameter space
    :return:            MMGW estimate in ellipse parameter space
    """
    particles[:, SR] = to_matrix_params(particles[:, AL], particles[:, L], particles[:, W], True)
    est = np.mean(particles, axis=0)
    est[SR] = get_ellipse_params_from_sr(est[SR])
    return est


def turn_mult(x, cov):
    """
    Turn the input density into a RED approximation with 4 components by splitting it in the four different orientations
    possible between -pi and pi describing the same ellipse.
    :param x:   the input ellipse mean, parameterized with center, orientation, and semi-axes (optional velocities)
    :param cov: the input covariance
    :return:    multimodal Gaussian density with 4 components, including means, covariances, and weights
    """
    x_mult = np.zeros((4, len(x)))
    cov_mult = np.zeros((4, len(x), len(x)))

    x_mult[0] = np.copy(x)
    cov_mult[0] = np.copy(cov)
    for i in range(1, 4):
        x_mult[i, M] = x[M]
        x_mult[i, AL] += x[AL] + i * 0.5 * np.pi
        x_mult[i, AL] = (x_mult[i, AL] + np.pi) % (2 * np.pi) - np.pi
        x_mult[i, L] = x_mult[i - 1, W]
        x_mult[i, W] = x_mult[i - 1, L]
        if len(x) > 5:
            x_mult[i, V] = x[V]
        cov_mult[i] = np.copy(cov)
        cov_mult[i, L, :3] = cov_mult[i - 1, W, :3]
        cov_mult[i, W, :3] = cov_mult[i - 1, L, :3]
        cov_mult[i, :3, L] = cov_mult[i - 1, :3, W]
        cov_mult[i, :3, W] = cov_mult[i - 1, :3, L]
        cov_mult[i, L, L] = cov_mult[i - 1, W, W]
        cov_mult[i, W, W] = cov_mult[i - 1, L, L]
        if len(x) > 5:
            cov_mult[i, L, 5:] = cov_mult[i - 1, W, 5:]
            cov_mult[i, W, 5:] = cov_mult[i - 1, L, 5:]
            cov_mult[i, 5:, L] = cov_mult[i - 1, 5:, W]
            cov_mult[i, 5:, W] = cov_mult[i - 1, 5:, L]

    w = 0.25 * np.ones(4)

    return x_mult, cov_mult, w


def sample_mult(x, cov, w, n_samples):
    """
    Sample from a multimodal Gaussian density.
    :param x:           the components' means
    :param cov:         the components' covariances
    :param w:           the components' weights
    :param n_samples:   the number of samples to be drawn
    :return:            the samples
    """
    # sample the components with repetition
    chosen = np.random.choice(len(x), n_samples, True, p=w)

    # sample from the respective components
    samples = np.zeros((n_samples, len(x[0])))
    for i in range(len(x)):
        if np.sum(chosen == i) > 0:
            samples[chosen == i] = mvn(x[i], cov[i], np.sum(chosen == i))

    return samples


def reduce_mult(means, covs, w):
    """
    Reduce mixture density by removing unlikely components and merging close components.
    :param means:   set of means
    :param covs:    set of covariances
    :param w:       weights of the components
    :return:        reduced set of component means, covariances, and weights
    """
    if len(means.shape) == 2:
        x_dim = len(means[0])
    else:
        x_dim = len(means)
    means_red, covs_red, w_red = means.copy(), covs.copy(), w.copy()

    # remove unlikely components
    keep = np.atleast_1d(w_red > WEIGHT_THRESH)
    if not any(keep):  # keep most likely component as we assume the target always exists
        keep[np.argmax(w_red)] = True
        means_red, covs_red, w_red = means_red[keep], covs_red[keep], w_red[keep]
        w_red /= np.sum(w_red)
        return means_red, covs_red, w_red

    means_red, covs_red, w_red = means_red[keep], covs_red[keep], w_red[keep]

    # merging
    nus = means_red[:, None, :] - means_red[None, :, :]
    inn_covs = covs_red[:, None, :, :] + covs_red[None, :, :, :]
    dists = np.einsum('xya, xyab, xyb -> xy', nus, np.linalg.inv(inn_covs), nus)
    db = DBSCAN(eps=CLOSE_THRESH, min_samples=1, metric='precomputed').fit(dists)
    labels = np.unique(db.labels_)
    means_merged = np.zeros((len(labels), x_dim))
    covs_merged = np.zeros((len(labels), x_dim, x_dim))
    w_merged = np.zeros(len(labels))
    for i in range(len(labels)):
        in_cluster = db.labels_ == i
        if np.sum(in_cluster) > 1:
            w_merged[i] = np.sum(w_red[in_cluster])
            means_merged[i] = np.sum(w_red[in_cluster, None] * means_red[in_cluster], axis=0) / w_merged[i]
            covs_merged[i] = np.sum(w_red[in_cluster, None, None]
                                    * (np.einsum('xa, xb -> xab', means_red[in_cluster], means_red[in_cluster])
                                       + covs_red[in_cluster]), axis=0) / w_merged[i] - np.outer(means_merged[i],
                                                                                                 means_merged[i])
        else:
            w_merged[i] = w_red[in_cluster]
            means_merged[i] = means_red[in_cluster]
            covs_merged[i] = covs_red[in_cluster]

    close_thresh = CLOSE_THRESH
    while len(w_merged) > MAX_COMP:
        w_red = w_merged.copy()
        means_red = means_merged.copy()
        covs_red = covs_merged.copy()

        close_thresh += 0.05
        nus = means_red[:, None, :] - means_red[None, :, :]
        inn_covs = covs_red[:, None, :, :] + covs_red[None, :, :, :]
        dists = np.einsum('xya, xyab, xyb -> xy', nus, np.linalg.inv(inn_covs), nus)
        db = DBSCAN(eps=close_thresh, min_samples=1, metric='precomputed').fit(dists)
        labels = np.unique(db.labels_)
        means_merged = np.zeros((len(labels), x_dim))
        covs_merged = np.zeros((len(labels), x_dim, x_dim))
        w_merged = np.zeros(len(labels))
        for i in range(len(labels)):
            in_cluster = db.labels_ == i
            if np.sum(in_cluster) > 1:
                w_merged[i] = np.sum(w_red[in_cluster])
                means_merged[i] = np.sum(w_red[in_cluster, None] * means_red[in_cluster], axis=0) / w_merged[i]
                covs_merged[i] = np.sum(w_red[in_cluster, None, None]
                                        * (np.einsum('xa, xb -> xab', means_red[in_cluster] - means_merged[i],
                                                     means_red[in_cluster] - means_merged[i])
                                           + covs_red[in_cluster]), axis=0) / w_merged[i]
            else:
                w_merged[i] = w_red[in_cluster]
                means_merged[i] = means_red[in_cluster]
                covs_merged[i] = covs_red[in_cluster]
    w_merged /= np.sum(w_merged)

    return means_merged, covs_merged, w_merged


def reduce_mult_salmond(means, covs, w, pruning=True):
    """
    Reduce mixture density by removing unlikely components and merging close components based on
    Salmond, D. J. "Mixture reduction algorithms for point and extended object tracking in clutter." IEEE Transactions
    on Aerospace and Electronic Systems 45.2 (2009): 667-686.
    :param means:   set of means
    :param covs:    set of covariances
    :param w:       weights of the components
    :return:        reduced set of component means, covariances, and weights
    """
    if len(means.shape) == 2:
        x_dim = len(means[0])
    else:
        x_dim = len(means)
    means_merged, covs_merged, w_merged = means.copy(), covs.copy(), w.copy()

    if pruning:
        # remove unlikely components
        keep = np.atleast_1d(w_merged > WEIGHT_THRESH)
        if not any(keep):  # keep most likely component as we assume the target always exists
            keep[np.argmax(w_merged)] = True
            means_merged, covs_merged, w_merged = means_merged[keep], covs_merged[keep], w_merged[keep]
            w_merged /= np.sum(w_merged)
            return means_merged, covs_merged, w_merged
        means_merged = means_merged[keep]
        covs_merged = covs_merged[keep]
        w_merged = w_merged[keep]
        w_merged /= np.sum(w_merged)

    num = len(w_merged)

    # merging
    close_thresh = CLOSE_THRESH
    while num > MAX_COMP:
        # cluster
        clusters = np.ones(num) * -1  # -1 means unclustered
        cluster_id = 0
        while any(clusters == -1):
            central_id = np.arange(num)[clusters == -1][np.argmax(w_merged[clusters == -1])]
            clusters[central_id] = cluster_id
            for i in np.arange(num)[clusters == -1]:
                dist = (w_merged[central_id]*w_merged[i] / (w_merged[central_id] + w_merged[i])) \
                       * ((means_merged[i] - means_merged[central_id]) @ np.linalg.inv(covs_merged[central_id])# + covs_merged[i])
                          @ (means_merged[i] - means_merged[central_id]))
                if dist < close_thresh:
                    clusters[i] = cluster_id
            cluster_id += 1

        # merge
        labels = np.arange(cluster_id)
        new_means_merged = np.zeros((cluster_id, x_dim))
        new_covs_merged = np.zeros((cluster_id, x_dim, x_dim))
        new_w_merged = np.zeros(cluster_id)
        for i in range(len(labels)):
            in_cluster = (clusters == i)
            if np.sum(in_cluster) > 1:
                new_w_merged[i] = np.sum(w_merged[in_cluster])
                new_means_merged[i] = np.sum(w_merged[in_cluster, None] * means_merged[in_cluster], axis=0) \
                                      / new_w_merged[i]
                new_covs_merged[i] = np.sum(w_merged[in_cluster, None, None]
                                        * (np.einsum('xa, xb -> xab', means_merged[in_cluster] - new_means_merged[i],
                                                     means_merged[in_cluster] - new_means_merged[i])
                                           + covs_merged[in_cluster]), axis=0) / new_w_merged[i]
            else:
                new_w_merged[i] = w_merged[in_cluster]
                new_means_merged[i] = means_merged[in_cluster]
                new_covs_merged[i] = covs_merged[in_cluster]
        new_w_merged /= np.sum(new_w_merged)

        means_merged = new_means_merged.copy()
        covs_merged = new_covs_merged.copy()
        w_merged = new_w_merged.copy()
        num = len(w_merged)

        close_thresh += 0.05

    return means_merged, covs_merged, w_merged


def reduce_mult_runnalls(means, covs, w):
    """
    Reduce mixture density based on Runnalls' algorithm.
    :param means:   set of means
    :param covs:    set of covariances
    :param w:       weights of the components
    :return:        reduced set of component means, covariances, and weights
    """
    means_merged, covs_merged, w_merged = means.copy(), covs.copy(), w.copy()
    num = len(w_merged)

    # create cost matrix
    cost = np.array([[get_cost(w_merged, means_merged, covs_merged, i, j) for j in range(num)] for i in range(num)])

    while num > MAX_COMP:
        min_id = np.argmin(cost)
        i = int(np.floor(min_id / num))
        j = min_id % num

        # save merged result at position i
        w_merged[i], means_merged[i], covs_merged[i] = merge_two(w_merged, means_merged, covs_merged, i, j)

        # update cost matrix
        cost[i, :] = [get_cost(w_merged, means_merged, covs_merged, i, k) for k in range(num)]
        cost[:, i] = cost[i, :]

        # remove j
        w_merged = np.delete(w_merged, j)
        means_merged = np.delete(means_merged, j, axis=0)
        covs_merged = np.delete(covs_merged, j, axis=0)
        cost = np.delete(cost, j, axis=0)
        cost = np.delete(cost, j, axis=1)
        num -= 1

    return means_merged, covs_merged, w_merged


def get_cost(w_merged, means_merged, covs_merged, i, j):
    """
    Get cost of merging. No merging with itself, so return infinite for i==j.
    :param w_merged:        weights
    :param means_merged:    means
    :param covs_merged:     covariances
    :param i:               ID of first component to be merged
    :param j:               ID of second component to be merged
    :return:                cost
    """
    if i == j:
        return np.inf

    w_ij, mean_ij, p_ij = merge_two(w_merged, means_merged, covs_merged, i, j)
    return 0.5 * ((w_merged[i] + w_merged[j]) * slogdet(p_ij)[1] - w_merged[i] * slogdet(covs_merged[i])[1]
                  - w_merged[j] * slogdet(covs_merged[j])[1])


def merge_two(w_merged, means_merged, covs_merged, i, j):
    """
    Merge two components indicated by i and j.
    :param w_merged:        weights
    :param means_merged:    means
    :param covs_merged:     covariances
    :param i:               ID of first component to be merged
    :param j:               ID of second component to be merged
    :return:                weight, mean, and covariance of merged components
    """
    w_ij = w_merged[i] + w_merged[j]
    mean_ij = (w_merged[i] * means_merged[i] + w_merged[j] * means_merged[j]) / w_ij
    p_ij = (w_merged[i] * (covs_merged[i] + np.outer(means_merged[i] - mean_ij, means_merged[i] - mean_ij))
            + w_merged[j] * (covs_merged[j] + np.outer(means_merged[j] - mean_ij, means_merged[j] - mean_ij))) \
           / w_ij

    return w_ij, mean_ij, p_ij


def barycenter(particles, w, n_particles, particles_sr=np.zeros(0)):
    """
    Determine Barycenter of particles in shape space via optimization (based on G. Puccetti, L. Rüschendorf, and
    S. Vanduffel, “On the Computation of Wasserstein Barycenters,” Available at SSRN 3276147, 2018).
    :param particles:       Particles in shape space [m1, m2, c11, c12, c22] with cnm being members of the covariance
                            matrix
    :param w:               Weights of the particles
    :param n_particles:     Number of particles
    :param particles_sr:    Particles with shape parameters in SR form; if given, used for initial guess
    :return:                Barycenter of the particles as [m1, m2, c11, c12, c22]
    """
    # Calculate covariances
    covs = np.zeros((n_particles, 2, 2))
    covs[:, 0, 0] = particles[:, 2]
    covs[:, 0, 1] = particles[:, 3]
    covs[:, 1, 0] = particles[:, 3]
    covs[:, 1, 1] = particles[:, 4]

    # Calculate Barycenter
    if len(particles_sr) > 0:
        covs_sr = np.zeros((n_particles, 2, 2))
        covs_sr[:, 0, 0] = particles_sr[:, 2]
        covs_sr[:, 0, 1] = particles_sr[:, 3]
        covs_sr[:, 1, 0] = particles_sr[:, 3]
        covs_sr[:, 1, 1] = particles_sr[:, 4]
        bary_sr = np.sum(w[:, None, None] * covs_sr, axis=0)
        bary = np.dot(bary_sr, bary_sr)
    else:
        bary = np.eye(2)
        bary_sr = np.eye(2)
    conv = False
    # loop until convergence
    while not conv:
        res = np.zeros((n_particles, 2, 2))
        for i in range(n_particles):
            res[i] = np.dot(np.dot(bary_sr, covs[i]), bary_sr)
            res[i] = sqrtm(res[i]) * w[i]
        bary_new = np.sum(res, axis=0)
        bary_sr = sqrtm(bary_new)

        # check convergence
        # bary_new = np.dot(bary_sr, bary_sr)
        diff = np.sum(abs(bary_new - bary))
        conv = diff < 1e-6
        bary = bary_new

    # Calculate mean and Barycenter in SR space
    result = np.zeros(5)
    result[M] = np.sum(w[:, None] * particles[:, M], axis=0)
    result[2] = bary[0, 0]
    result[3] = bary[0, 1]
    result[4] = bary[1, 1]

    return result
