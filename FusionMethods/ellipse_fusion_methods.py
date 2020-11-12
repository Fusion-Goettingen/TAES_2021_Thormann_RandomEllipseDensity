"""
Author: Kolja Thormann

Contains various ellipse fusion methods
"""

import numpy as np

from FusionMethods.ellipse_fusion_support import get_ellipse_params, get_ellipse_params_from_sr, to_matrix,\
    single_particle_approx_gaussian, mmgw_estimate_from_particles, sample_m, turn_mult, reduce_mult, sample_mult
from FusionMethods.error_and_plotting import error_and_plotting
from FusionMethods.constants import *


def mmgw_mc_update(mmgw_mc, meas, cov_meas, n_particles, gt, step_id, steps, plot_cond, save_path):
    """
    Fusion using MMGW-MC; creates particle density in square root space of measurements and approximates it as a
    Gaussian distribution to fuse it with the current estimate in Kalman fashion.
    :param mmgw_mc:     Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in ellipse parameter space
    :param cov_meas:    Covariance of measurement in ellipse parameter space
    :param n_particles: Number of particles used for approximating the transformed density
    :param gt:          Ground truth
    :param step_id:     Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    """
    # predict
    mmgw_mc['x'] = np.dot(F, mmgw_mc['x'])
    error_mat = np.array([
        [0.5 * T ** 2, 0.0],
        [0.0, 0.5 * T ** 2],
        [0.0,          0.0],
        [0.0,          0.0],
        [0.0,          0.0],
        [T, 0.0],
        [0.0, T],
    ])
    error_cov = np.dot(np.dot(error_mat, np.diag([SIGMA_V1, SIGMA_V2]) ** 2), error_mat.T)
    error_cov[SR, SR] = np.asarray(SIGMA_SHAPE) ** 2
    mmgw_mc['cov'] = np.dot(np.dot(F, mmgw_mc['cov']), F.T) + error_cov

    # convert measurement
    meas_sr, cov_meas_sr, particles_meas = single_particle_approx_gaussian(meas, cov_meas, n_particles)

    # store prior for plotting
    m_prior = mmgw_mc['x'][M]
    al_prior, l_prior, w_prior = get_ellipse_params_from_sr(mmgw_mc['x'][SR])

    # Kalman fusion
    innov_cov = np.dot(np.dot(H, mmgw_mc['cov']), H.T) + cov_meas_sr
    gain = np.dot(np.dot(mmgw_mc['cov'], H.T), np.linalg.inv(innov_cov))
    mmgw_mc['x'] = mmgw_mc['x'] + np.dot(gain, meas_sr - np.dot(H, mmgw_mc['x']))
    mmgw_mc['cov'] = mmgw_mc['cov'] - np.dot(np.dot(gain, innov_cov), gain.T)

    # save error and plot estimate
    al_post_sr, l_post_sr, w_post_sr = get_ellipse_params_from_sr(mmgw_mc['x'][SR])
    mmgw_mc['error'][step_id::steps] += error_and_plotting(mmgw_mc['x'][M], l_post_sr, w_post_sr, al_post_sr, m_prior,
                                                           l_prior, w_prior, al_prior, meas[M], meas[L], meas[W],
                                                           meas[AL], gt[M], gt[L], gt[W], gt[AL], plot_cond,
                                                           'MC Approximated Fusion',
                                                           save_path + 'exampleMCApprox%i.svg' % step_id,
                                                           mmgw_mc['figure'], mmgw_mc['axes'],
                                                           est_color=mmgw_mc['color'])


def regular_update(regular, meas, cov_meas, gt, step_id, steps, plot_cond, save_path, use_mmgw):
    """
    Fuse estimate and measurement in original state space in Kalman fashion.
    :param regular:     Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space
    :param gt:          Ground truth
    :param step_id:     Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    :param use_mmgw:    Use the MMGW instead of the ordinary estimate
    """
    # predict
    regular['x'] = np.dot(F, regular['x'])
    error_mat = np.array([
        [0.5 * T ** 2, 0.0],
        [0.0, 0.5 * T ** 2],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [T, 0.0],
        [0.0, T],
    ])
    error_cov = np.dot(np.dot(error_mat, np.diag([SIGMA_V1, SIGMA_V2]) ** 2), error_mat.T)
    error_cov[SR, SR] = SIGMA_SHAPE ** 2
    regular['cov'] = np.dot(np.dot(F, regular['cov']), F.T) + error_cov

    # store prior for plotting
    all_prior = regular['est'].copy()

    # Kalman fusion
    innov = meas - np.dot(H, regular['x'])
    innov[AL] = (innov[AL] + np.pi) % (2*np.pi) - np.pi
    innov_cov = np.dot(np.dot(H, regular['cov']), H.T) + cov_meas
    gain = np.dot(np.dot(regular['cov'], H.T), np.linalg.inv(innov_cov))
    regular['x'] = regular['x'] + np.dot(gain, innov)
    regular['cov'] = regular['cov'] - np.dot(np.dot(gain, innov_cov), gain.T)
    if use_mmgw:
        particles = sample_m(regular['x'], regular['cov'], False, N_PARTICLES_MMGW)
        regular['est'] = mmgw_estimate_from_particles(particles)
    else:
        regular['est'] = regular['x'].copy()

    # save error and plot estimate
    regular['error'][step_id::steps] += error_and_plotting(regular['est'][M], regular['est'][L], regular['est'][W],
                                                           regular['est'][AL], all_prior[M], all_prior[L], all_prior[W],
                                                           all_prior[AL], meas[M], meas[L], meas[W], meas[AL], gt[M],
                                                           gt[L], gt[W], gt[AL], plot_cond, regular['name'],
                                                           save_path + 'example' + regular['name'] + '%i.svg' % step_id,
                                                           regular['figure'], regular['axes'],
                                                           est_color=regular['color'])


def red_update(red, meas, cov_meas, gt, step_id, steps, plot_cond, save_path, use_mmgw):
    """
    Method utilizing RED. Fuses the four components of the RED with orientation between
    0 and 2pi with those of the measurement RED and applies mixture reduction on the resulting multi modal density. The
    mean of the density is estimated by taking the mean of the highest weighted component or by using the MMGW
    estimator.
    :param red:         The state containing mean, covariance, etc.
    :param meas:        Measurement
    :param cov_meas:    Measurement covariance
    :param gt:          Ground truth (for plotting and error calculation)
    :param step_id:     Current step index (for plotting and error calculation)
    :param steps:       Total number of steps
    :param plot_cond:   Boolean for plotting the current time step
    :param save_path:   Path to save the plots
    :param use_mmgw:    Use the MMGW instead of the highest weight estimate
    """
    # predict
    error_mat = np.array([
        [0.5 * T ** 2, 0.0],
        [0.0, 0.5 * T ** 2],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [T, 0.0],
        [0.0, T],
    ])
    error_cov = np.dot(np.dot(error_mat, np.diag([SIGMA_V1, SIGMA_V2]) ** 2), error_mat.T)
    error_cov[SR, SR] = SIGMA_SHAPE ** 2
    for i in range(len(red['x'])):
        red['x'][i] = np.dot(F, red['x'][i])
        red['cov'][i] = np.dot(np.dot(F, red['cov'][i]), F.T) + error_cov

    prior = red['est']

    # transform prior and measurement into reduced REDs
    meas_mult, meas_cov_mult, meas_weights = turn_mult(meas, cov_meas)

    # calculate posterior RED
    post_mult = np.zeros((len(red['x']) * len(meas_mult), len(red['x'][0])))
    post_cov_mult = np.zeros((len(red['x']) * len(meas_mult), len(red['x'][0]), len(red['x'][0])))
    post_weights = np.zeros(len(red['x']) * len(meas_mult))
    log_prior_weights = np.log(red['comp_weights'])
    for i in range(len(red['x'])):
        for j in range(len(meas_mult)):
            nu = meas_mult[j] - np.dot(H, red['x'][i])
            nu[AL] = (nu[AL] + np.pi) % (2 * np.pi) - np.pi
            nu_cov = np.dot(np.dot(H, red['cov'][i]), H.T) + meas_cov_mult[j]
            post_mult[i * len(meas_mult) + j] = red['x'][i] + np.dot(np.dot(np.dot(red['cov'][i], H.T),
                                                                            np.linalg.inv(nu_cov)), nu)
            post_cov_mult[i * len(meas_mult) + j] = red['cov'][i] - np.dot(np.dot(np.dot(red['cov'][i], H.T),
                                                                                  np.linalg.inv(nu_cov)),
                                                                           np.dot(H, red['cov'][i]))
            post_weights[i * len(meas_mult) + j] = -2.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(nu_cov)) \
                                                   - 0.5 * np.dot(np.dot(nu, np.linalg.inv(nu_cov)), nu)
            post_weights[i * len(meas_mult) + j] += log_prior_weights[i]
    post_weights -= np.log(np.sum(np.exp(post_weights)))
    post_weights = np.exp(post_weights)

    red['x'], red['cov'], red['comp_weights'] = reduce_mult(post_mult, post_cov_mult, post_weights)
    if use_mmgw:
        particles = sample_mult(post_mult, post_cov_mult, post_weights, N_PARTICLES_MMGW)
        red['est'] = mmgw_estimate_from_particles(particles)
    else:
        red['est'] = red['x'][np.argmax(red['comp_weights'])].copy()

    red['error'][step_id::steps] += error_and_plotting(red['est'][M], red['est'][L], red['est'][W], red['est'][AL],
                                                       prior[M], prior[L], prior[W], prior[AL], meas[M], meas[L],
                                                       meas[W], meas[AL], gt[M], gt[L], gt[W], gt[AL], plot_cond,
                                                       red['name'],
                                                       save_path + 'example' + red['name'] + '%i.svg' % step_id,
                                                       red['figure'], red['axes'], est_color=red['color'])


def shape_mean_update(shape_mean, meas, cov_meas, gt, step_id, steps, plot_cond, save_path, tau=1.0):
    """
    Treat ellipse estimates as random matrices having received an equal degree.
    :param shape_mean:  Current estimate (also stores error); will be modified as a result
    :param meas:        Measurement in original state space
    :param cov_meas:    Covariance of measurement in original state space (only m is used)
    :param gt:          Ground truth
    :param step_id:     Current measurement step
    :param steps:       Total measurement steps
    :param plot_cond:   Boolean determining whether to plot the current estimate
    :param save_path:   Path to save the plots
    :param tau:         forget parameter of prediction step
    """
    # store prior for plotting
    m_prior = shape_mean['x'][M]
    al_prior, l_prior, w_prior = get_ellipse_params(shape_mean['shape'])

    # predict
    shape_mean['x'] = np.dot(F[KIN][:, KIN], shape_mean['x'])
    error_mat = np.array([
        [0.5 * T ** 2, 0.0],
        [0.0, 0.5 * T ** 2],
        [T, 0.0],
        [0.0, T],
    ])
    error_cov = np.dot(np.dot(error_mat, np.diag([SIGMA_V1, SIGMA_V2]) ** 2), error_mat.T)
    shape_mean['cov'] = np.dot(np.dot(F[KIN][:, KIN], shape_mean['cov']), F[KIN][:, KIN].T) + error_cov
    shape_mean['gamma'] = 6.0 + np.exp(-T / tau)*(shape_mean['gamma'] - 6.0)

    # convert measurement
    shape_meas = to_matrix(meas[AL], meas[L], meas[W], False)

    # Kalman fusion
    innov_cov_k = np.dot(np.dot(H_SHAPE, shape_mean['cov']), H_SHAPE.T) + cov_meas[KIN_MEAS][:, KIN_MEAS]
    gain_k = np.dot(np.dot(shape_mean['cov'], H_SHAPE.T), np.linalg.inv(innov_cov_k))
    shape_mean['x'] = shape_mean['x'] + np.dot(gain_k, meas[KIN_MEAS] - np.dot(H_SHAPE, shape_mean['x']))
    shape_mean['cov'] = shape_mean['cov'] - np.dot(np.dot(gain_k, innov_cov_k), gain_k.T)
    shape_mean['shape'] = (shape_mean['gamma'] * shape_mean['shape'] + 6.0 * shape_meas) / (shape_mean['gamma'] + 6.0)
    shape_mean['gamma'] += 6.0

    # save error and plot estimate
    al_post, l_post, w_post = get_ellipse_params(shape_mean['shape'])
    shape_mean['error'][step_id::steps] += error_and_plotting(shape_mean['x'][M], l_post, w_post, al_post, m_prior,
                                                              l_prior, w_prior, al_prior, meas[M], meas[L], meas[W],
                                                              meas[AL], gt[M], gt[L], gt[W], gt[AL], plot_cond,
                                                              shape_mean['name'],
                                                              save_path + 'example' + shape_mean['name'] + '%i.svg'
                                                              % step_id, shape_mean['figure'], shape_mean['axes'],
                                                              est_color=shape_mean['color'])
