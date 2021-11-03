"""
Author: Kolja Thormann

Contains test cases for different ellipse fusion methods
"""

import numpy as np
import time

from numpy.random import multivariate_normal as mvn

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tikzplotlib

from FusionMethods.ellipse_fusion_methods import mmgw_mc_update, regular_update, red_update, shape_mean_update
from FusionMethods.ellipse_fusion_support import sample_m, get_ellipse_params, single_particle_approx_gaussian,\
    barycenter, to_matrix, get_ellipse_params_from_sr, mmgw_estimate_from_particles, turn_mult, sample_mult
from FusionMethods.error_and_plotting import gauss_wasserstein, square_root_distance, plot_error_bars,\
    plot_convergence
from FusionMethods.constants import *


state_dtype = np.dtype([
    ('x', 'O'),               # [m1, m2, al, l ,w] or [m1, m2, s11, s12, s22]
    ('cov', 'O'),             # covariance
    ('comp_weights', 'O'),    # weights of components in red fusion
    ('est', 'O'),             # the estimate based on the posterior density
    ('error', 'O'),           # length depends on number of steps
    ('shape', 'f4', (2, 2)),  # for mean of shape matrix only
    ('gamma', 'i4'),          # for RM mean, keep track of number of measurements
    ('name', 'O'),            # name of the fusion method
    ('color', 'O'),           # color for error plotting
    ('figure', 'O'),          # figure for plotting
    ('axes', 'O'),            # axes for plotting
])


def test_convergence(steps, runs, prior, cov_prior, cov_meas, random_param, save_path):
    """
    Test convergence of error for different fusion methods. Creates plot of root mean square error convergence and
    errors at first and last measurement step.
    :param steps:           Number of measurements
    :param runs:            Number of MC runs
    :param prior:           Prior prediction mean (ground truth will be drawn from it each run)
    :param cov_prior:       Prior prediction covariance (ground truth will be drawn from it each run)
    :param cov_meas:        Noise of sensor
    :param random_param:    use random parameter switch to simulate ambiguous parameterization
    :param save_path:       Path for saving figures
    """
    error = np.zeros(steps*2)

    # setup state for various ellipse fusion methods
    mmgw_mc = np.zeros(1, dtype=state_dtype)
    mmgw_mc[0]['error'] = error.copy()
    mmgw_mc[0]['name'] = 'MC-MMGW'
    mmgw_mc[0]['color'] = 'cyan'
    regular = np.zeros(1, dtype=state_dtype)
    regular[0]['error'] = error.copy()
    regular[0]['name'] = 'Regular'
    regular[0]['color'] = 'red'
    regular_mmgw = np.zeros(1, dtype=state_dtype)
    regular_mmgw[0]['error'] = error.copy()
    regular_mmgw[0]['name'] = 'Regular-MMGW'
    regular_mmgw[0]['color'] = 'orange'
    red_mmgw = np.zeros(1, dtype=state_dtype)
    red_mmgw[0]['error'] = error.copy()
    red_mmgw[0]['name'] = 'RED-MMGW'
    red_mmgw[0]['color'] = 'green'
    # red_mmgw_r = np.zeros(1, dtype=state_dtype)
    # red_mmgw_r[0]['error'] = error.copy()
    # red_mmgw_r[0]['name'] = 'RED-MMGW-r'
    # red_mmgw_r[0]['color'] = 'lightgreen'
    # red_mmgw_s = np.zeros(1, dtype=state_dtype)
    # red_mmgw_s[0]['error'] = error.copy()
    # red_mmgw_s[0]['name'] = 'RED-MMGW-s'
    # red_mmgw_s[0]['color'] = 'turquoise'
    shape_mean = np.zeros(1, dtype=state_dtype)
    shape_mean[0]['error'] = error.copy()
    shape_mean[0]['name'] = 'Shape-Mean'
    shape_mean[0]['color'] = 'magenta'

    # rt_red = 0.0
    # rt_red_r = 0.0
    # rt_red_s = 0.0

    for r in range(runs):
        print('Run %i of %i' % (r+1, runs))
        # initialize ===================================================================================================
        # create gt from prior
        gt = sample_m(prior, cov_prior, False, 1)

        # ellipse orientation should be velocity orientation
        vel = np.linalg.norm(gt[V])
        gt[V] = np.array(np.cos(gt[AL]), np.sin(gt[AL])) * vel

        # get prior in square root space
        mmgw_mc[0]['x'], mmgw_mc[0]['cov'], particles_mc = single_particle_approx_gaussian(prior, cov_prior,
                                                                                           N_PARTICLES_MMGW)
        mmgw_mc[0]['est'] = mmgw_mc[0]['x'].copy()
        mmgw_mc[0]['est'][SR] = get_ellipse_params_from_sr(mmgw_mc[0]['x'][SR])
        mmgw_mc[0]['figure'], mmgw_mc[0]['axes'] = plt.subplots(1, 1)

        # get prior for regular state
        regular[0]['x'] = prior.copy()
        regular[0]['cov'] = cov_prior.copy()
        regular[0]['est'] = prior.copy()
        regular[0]['figure'], regular[0]['axes'] = plt.subplots(1, 1)
        regular_mmgw[0]['x'] = prior.copy()
        regular_mmgw[0]['cov'] = cov_prior.copy()
        particles = sample_m(regular_mmgw[0]['x'], regular_mmgw[0]['cov'], False, N_PARTICLES_MMGW)
        regular_mmgw[0]['est'] = mmgw_estimate_from_particles(particles)
        regular_mmgw[0]['figure'], regular_mmgw[0]['axes'] = plt.subplots(1, 1)

        # get prior for red
        red_mmgw[0]['x'], red_mmgw[0]['cov'], red_mmgw[0]['comp_weights'] = turn_mult(prior.copy(), cov_prior.copy())
        particles = sample_mult(red_mmgw[0]['x'], red_mmgw[0]['cov'], red_mmgw[0]['comp_weights'], N_PARTICLES_MMGW)
        red_mmgw[0]['est'] = mmgw_estimate_from_particles(particles)
        red_mmgw[0]['figure'], red_mmgw[0]['axes'] = plt.subplots(1, 1)

        # red_mmgw_r[0]['x'], red_mmgw_r[0]['cov'], red_mmgw_r[0]['comp_weights'] = turn_mult(prior.copy(), cov_prior.copy())
        # particles = sample_mult(red_mmgw_r[0]['x'], red_mmgw_r[0]['cov'], red_mmgw_r[0]['comp_weights'], N_PARTICLES_MMGW)
        # red_mmgw_r[0]['est'] = mmgw_estimate_from_particles(particles)
        # red_mmgw_r[0]['figure'], red_mmgw_r[0]['axes'] = plt.subplots(1, 1)
        #
        # red_mmgw_s[0]['x'], red_mmgw_s[0]['cov'], red_mmgw_s[0]['comp_weights'] = turn_mult(prior.copy(),
        #                                                                                     cov_prior.copy())
        # particles = sample_mult(red_mmgw_s[0]['x'], red_mmgw_s[0]['cov'], red_mmgw_s[0]['comp_weights'],
        #                         N_PARTICLES_MMGW)
        # red_mmgw_s[0]['est'] = mmgw_estimate_from_particles(particles)
        # red_mmgw_s[0]['figure'], red_mmgw_s[0]['axes'] = plt.subplots(1, 1)

        # get prior for RM mean
        shape_mean[0]['x'] = prior[KIN]
        shape_mean[0]['shape'] = to_matrix(prior[AL], prior[L], prior[W], False)
        shape_mean[0]['cov'] = cov_prior[KIN][:, KIN]
        shape_mean[0]['gamma'] = 6.0
        shape_mean[0]['figure'], shape_mean[0]['axes'] = plt.subplots(1, 1)

        # test different methods
        for i in range(steps):
            if i % 10 == 0:
                print('Step %i of %i' % (i + 1, steps))
            plot_cond = (r + 1 == runs) & ((i % 2) == 1)  # & (i + 1 == steps)

            # move ground truth
            gt = np.dot(F, gt)
            error_mat = np.array([
                [0.5 * T ** 2,          0.0],
                [0.0,          0.5 * T ** 2],
                [T,                     0.0],
                [0.0,                     T],
            ])
            kin_cov = np.dot(np.dot(error_mat, np.diag([SIGMA_V1, SIGMA_V2]) ** 2), error_mat.T)
            gt[KIN] += mvn(np.zeros(len(KIN)), kin_cov)
            gt[AL] = np.arctan2(gt[V2], gt[V1])

            # create measurement from gt (using alternating sensors) ===================================================
            k = np.random.randint(0, 4) if random_param else 0
            gt_mean = gt.copy()
            if k % 2 == 1:
                l_save = gt_mean[L]
                gt_mean[L] = gt_mean[W]
                gt_mean[W] = l_save
            gt_mean[AL] = (gt_mean[AL] + 0.5 * np.pi * k + np.pi) % (2 * np.pi) - np.pi
            meas = sample_m(np.dot(H, gt_mean), cov_meas, False, 1)

            # fusion methods ===========================================================================================
            mmgw_mc_update(mmgw_mc[0], meas.copy(), cov_meas.copy(), N_PARTICLES_MMGW, gt, i, steps, plot_cond, save_path)

            regular_update(regular[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path, False)

            regular_update(regular_mmgw[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path, True)

            # tic = time.time()
            red_update(red_mmgw[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path, True,
                       mixture_reduction='salmond')
            # toc = time.time()
            # rt_red += toc - tic

            # tic = time.time()
            # red_update(red_mmgw_r[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path, True,
            #            mixture_reduction='salmond', pruning=False)
            # toc = time.time()
            # rt_red_r += toc - tic

            # tic = time.time()
            # red_update(red_mmgw_s[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path, True,
            #            mixture_reduction='salmond')
            # toc = time.time()
            # rt_red_s += toc - tic

            shape_mean_update(shape_mean[0], meas.copy(), cov_meas.copy(), gt, i, steps, plot_cond, save_path,
                              0.2 if cov_meas[AL, AL] < 0.1*np.pi else 5.0 if cov_meas[AL, AL] < 0.4*np.pi else 10.0)

        plt.close(mmgw_mc[0]['figure'])
        plt.close(regular[0]['figure'])
        plt.close(regular_mmgw[0]['figure'])
        plt.close(red_mmgw[0]['figure'])
        # plt.close(red_mmgw_r[0]['figure'])
        # plt.close(red_mmgw_s[0]['figure'])
        plt.close(shape_mean[0]['figure'])

    mmgw_mc[0]['error'] = np.sqrt(mmgw_mc[0]['error'] / runs)
    regular[0]['error'] = np.sqrt(regular[0]['error'] / runs)
    regular_mmgw[0]['error'] = np.sqrt(regular_mmgw[0]['error'] / runs)
    red_mmgw[0]['error'] = np.sqrt(red_mmgw[0]['error'] / runs)
    # red_mmgw_r[0]['error'] = np.sqrt(red_mmgw_r[0]['error'] / runs)
    # red_mmgw_s[0]['error'] = np.sqrt(red_mmgw_s[0]['error'] / runs)
    shape_mean[0]['error'] = np.sqrt(shape_mean[0]['error'] / runs)

    # print('Runtime RED:')
    # print(rt_red / (runs*steps))
    # print('Runtime RED no pruning:')
    # print(rt_red_r / (runs * steps))
    # print('Runtime RED-S:')
    # print(rt_red_s / (runs * steps))

    # error plotting ===================================================================================================
    plot_error_bars(np.block([regular, regular_mmgw, shape_mean, mmgw_mc, red_mmgw]), steps)
    plot_convergence(np.block([regular, regular_mmgw, shape_mean, mmgw_mc, red_mmgw]), steps, save_path)


def test_mean(orig, cov, n_particles, save_path):
    """
    Compare the mean in original state space with mean in square root space (via MC approximation) in regards of their
    GW and ESR error.
    :param orig:        Mean in original state space
    :param cov:         Covariance for original state space
    :param n_particles: Number of particles for MC approximation of SR space mean
    :param save_path:   Path for saving figures
    """
    # approximate mean in SR space
    vec_mmsr, _, vec_particle = single_particle_approx_gaussian(orig, cov, n_particles, True)
    mat_mmsr = np.array([
        [vec_mmsr[2], vec_mmsr[3]],
        [vec_mmsr[3], vec_mmsr[4]]
    ])
    mat_mmsr = np.dot(mat_mmsr, mat_mmsr)
    al_mmsr, l_mmsr, w_mmsr = get_ellipse_params(mat_mmsr)

    # approximate mean in matrix space
    mat_mat = np.zeros((2, 2))
    vec_mat = np.zeros(2)
    for i in range(len(vec_particle)):
        vec_mat += vec_particle[i, :2]
        mat = np.array([
            [vec_particle[i, 2], vec_particle[i, 3]],
            [vec_particle[i, 3], vec_particle[i, 4]]
        ])
        mat_mat += np.dot(mat, mat)
    vec_mat /= len(vec_particle)
    mat_mat /= len(vec_particle)
    al_mat, l_mat, w_mat = get_ellipse_params(mat_mat)

    # caclulate Barycenter using optimization
    covs_sr = np.zeros((n_particles, 2, 2))
    covs_sr[:, 0, 0] = vec_particle[:, 2]
    covs_sr[:, 0, 1] = vec_particle[:, 3]
    covs_sr[:, 1, 0] = vec_particle[:, 3]
    covs_sr[:, 1, 1] = vec_particle[:, 4]
    covs = np.einsum('xab, xbc -> xac', covs_sr, covs_sr)
    bary_particles = np.zeros((n_particles, 5))
    bary_particles[:, M] = vec_particle[:, M]
    bary_particles[:, 2] = covs[:, 0, 0]
    bary_particles[:, 3] = covs[:, 0, 1]
    bary_particles[:, 4] = covs[:, 1, 1]
    bary = barycenter(bary_particles, np.ones(n_particles) / n_particles, n_particles, particles_sr=vec_particle)
    mat_mmgw = np.array([
        [bary[2], bary[3]],
        [bary[3], bary[4]],
    ])
    al_mmgw, l_mmgw, w_mmgw = get_ellipse_params(mat_mmgw)

    # approximate error
    error_orig_gw = 0
    error_orig_sr = 0
    error_mmsr_gw = 0
    error_mmsr_sr = 0
    error_mat_gw = 0
    error_mat_sr = 0
    error_mmgw_gw = 0
    error_mmgw_sr = 0
    for i in range(n_particles):
        mat_particle = np.array([
            [vec_particle[i, 2], vec_particle[i, 3]],
            [vec_particle[i, 3], vec_particle[i, 4]]
        ])
        mat_particle = np.dot(mat_particle, mat_particle)
        al_particle, l_particle, w_particle = get_ellipse_params(mat_particle)
        error_orig_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, orig[M], orig[L],
                                           orig[W], orig[AL])
        error_orig_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, orig[M],
                                              orig[L], orig[W], orig[AL])
        error_mmsr_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mmsr[M],
                                           l_mmsr, w_mmsr, al_mmsr)
        error_mmsr_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mmsr[M],
                                              l_mmsr, w_mmsr, al_mmsr)
        error_mat_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mat[M],
                                          l_mat, w_mat, al_mat)
        error_mat_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, vec_mat[M],
                                             l_mat, w_mat, al_mat)
        error_mmgw_gw += gauss_wasserstein(vec_particle[i, :2], l_particle, w_particle, al_particle, bary[M], l_mmgw,
                                           w_mmgw, al_mmgw)
        error_mmgw_sr += square_root_distance(vec_particle[i, :2], l_particle, w_particle, al_particle, bary[M], l_mmgw,
                                              w_mmgw, al_mmgw)
    error_orig_gw = np.sqrt(error_orig_gw / n_particles)
    error_orig_sr = np.sqrt(error_orig_sr / n_particles)
    error_mmsr_gw = np.sqrt(error_mmsr_gw / n_particles)
    error_mmsr_sr = np.sqrt(error_mmsr_sr / n_particles)
    error_mat_gw = np.sqrt(error_mat_gw / n_particles)
    error_mat_sr = np.sqrt(error_mat_sr / n_particles)
    error_mmgw_gw = np.sqrt(error_mmgw_gw / n_particles)
    error_mmgw_sr = np.sqrt(error_mmgw_sr / n_particles)

    fig, ax = plt.subplots(1, 1)

    samples = np.random.choice(n_particles, 20, replace=False)
    for i in range(20):
        al_particle, l_particle, w_particle = get_ellipse_params_from_sr(vec_particle[samples[i], SR])
        el_particle = Ellipse((vec_particle[samples[i], X1], vec_particle[samples[i], X2]), 2 * l_particle,
                              2 * w_particle, np.rad2deg(al_particle), fill=True, linewidth=2.0)
        el_particle.set_alpha(0.4)
        el_particle.set_fc('grey')
        ax.add_artist(el_particle)

    el_gt = Ellipse((orig[X1], orig[X2]), 2 * orig[L], 2 * orig[W], np.rad2deg(orig[AL]), fill=False, linewidth=2.0)
    el_gt.set_alpha(0.7)
    el_gt.set_ec('red')
    ax.add_artist(el_gt)
    plt.plot([0], [0], color='red', label='Euclidean')

    ela_final = Ellipse((vec_mat[X1], vec_mat[X2]), 2 * l_mat, 2 * w_mat, np.rad2deg(al_mat), fill=False,
                        linewidth=2.0)
    ela_final.set_alpha(0.7)
    ela_final.set_ec('magenta')
    ax.add_artist(ela_final)
    plt.plot([0], [0], color='magenta', label='Shape mean')

    elb_final = Ellipse((vec_mmsr[X1], vec_mmsr[X2]), 2 * l_mmsr, 2 * w_mmsr, np.rad2deg(al_mmsr), fill=False,
                        linewidth=2.0)
    elb_final.set_alpha(0.7)
    elb_final.set_ec('lightgreen')
    ax.add_artist(elb_final)
    plt.plot([0], [0], color='lightgreen', label='ESR')

    el_res = Ellipse((bary[X1], bary[X2]), 2 * l_mmgw, 2 * w_mmgw, np.rad2deg(al_mmgw), fill=False, linewidth=2.0,
                     linestyle='--')
    el_res.set_alpha(0.7)
    el_res.set_ec('green')
    ax.add_artist(el_res)
    plt.plot([0], [0], color='green', label='GW')

    plt.axis([-10 + orig[0], 10 + orig[0], -10 + orig[1], 10 + orig[1]])
    ax.set_aspect('equal')
    ax.set_title('MMSE Estimates')
    plt.xlabel('x in m')
    plt.ylabel('y in m')
    plt.legend()
    tikzplotlib.save(save_path + 'mmseEstimates.tex', add_axis_environment=False)
    # plt.savefig(save_path + 'mmseEstimates.svg')
    plt.show()

    # print error
    print('RMGW of original:')
    print(error_orig_gw)
    print('RMSR of original:')
    print(error_orig_sr)
    print('RMGW of mmsr:')
    print(error_mmsr_gw)
    print('RMSR of mmsr:')
    print(error_mmsr_sr)
    print('RMGW of Euclidean mmse:')
    print(error_mat_gw)
    print('RMSR of Euclidean mmse:')
    print(error_mat_sr)
    print('RMGW of mmgw_bary:')
    print(error_mmgw_gw)
    print('RMSR of mmgw_bary:')
    print(error_mmgw_sr)

    bars = np.arange(1, 8, 2)
    ticks = np.array(['Euclidean', 'Shape mean', 'MMSR', 'MMGW'])

    plt.bar(bars[0], error_orig_gw, width=0.25, color='red', align='center')
    plt.bar(bars[1], error_mat_gw, width=0.25, color='magenta', align='center')
    plt.bar(bars[2], error_mmsr_gw, width=0.25, color='lightgreen', align='center')
    plt.bar(bars[3], error_mmgw_gw, width=0.25, color='black', align='center')

    plt.xticks(bars, ticks)
    plt.title('GW RMSE')
    plt.savefig(save_path + 'meanTestGW.svg')
    plt.show()
