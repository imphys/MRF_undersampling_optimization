# Author: D.G.J. Heesterbeek
# Delft University of Technology
# Last revision: 11-06-2022
"""
Script for sequence optimisation using the predictions from UEE_phase.py
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.interpolate as interpolate
import scipy.optimize as sc_opt
import nibabel as nib
import pdb
from scipy.optimize import Bounds
from datetime import datetime
from importlib import reload
import concurrent.futures
from tqdm import tqdm
import UEE_phase as UEE
import Optimisation

reload(UEE)
reload(Optimisation)


def jac(Opt_param):
    """
    Function calculating the Jacobian using the JAC class from UEE_phase.py
    :param Opt_param: Currennt flip angle sequence
    :return: Jacobian used for sequence optimisation
    """
    # Base point
    base = Prob.full_run(Opt_param)

    # Determine step size for gradient evaluation
    res_step = np.finfo(Opt_param.dtype).eps ** (0.5)  # See Numerical Recipes ( W. H. Press et. al). Change for 3-point method
    step = (res_step) * np.maximum(1, Opt_param)

    Prob_grad = UEE.JAC(Prob=Prob, Opt_param=Opt_param, base=base, step=step)

    # Initialisation
    Jac = np.zeros(N)
    # ------------------------------------------------------------------------------------------------------------------
    # Uncomment line 47--49 when running on a cluster (and comment 54, 55) for use of multiple CPUs
    # ------------------------------------------------------------------------------------------------------------------
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #    for it, der in enumerate(tqdm(executor.map(Prob_grad.jac_step, range(N)))):
    #       Jac[it] = der

    # ------------------------------------------------------------------------------------------------------------------
    # Comment line 54, 55 when running on a cluster (and uncomment 46--49) for use of multiple CPUs
    # ------------------------------------------------------------------------------------------------------------------
    for it in tqdm(range(N), desc='Calculating one Jacobian for optimisation...'):
        Jac[it] = Prob_grad.jac_step(it)

    return Jac

if __name__ == '__main__':
    startTime = datetime.now()
    # %% Optimisation settings
    # Tolerance for convergence
    ftol = 1e-2
    # Lower bound on flip angle sequence
    alpha_min = 1/18 * np.pi
    # Upper bound on flip angle sequence
    alpha_max = 1/3 * np.pi
    # Upper bound on absolute difference between two consecutive flip angles
    alpha_max_step = 1/180 * np.pi


    # %% Practical settings
    # Boolean for saving the figures in the current directory
    savefig = True
    # Boolean for saving the data in the current directory
    savedat = True
    # Boolean for printing information about the results in the console
    printing = True
    # Boolean for applying density compensation (should be True for realistic outcomes)
    dcf_bool = True
    # Boolean for predicting the undersampling error (if False, no error will be predicted). This boolean is just
    # to check the code
    under_sampling = True
    # Prefix used for saving the data
    save_index = 'INDEX'

    # Numerical value that determines which parameter map is plotted (0 for T1, 1 for T2)
    plot_index = 1

    # %% k-space sampling settings
    # Boolean for including the 'PSF error' term
    PSF_err = True
    # Boolean for golden angle rotation of the k-space sampling pattern. If False a 2*pi/32 incremental rotation is
    # applied
    golden_angle = False
    # String for the spiral name (options: 'Philips_spiral' and # 'var_density')
    spiral = 'Philips_spiral'
    # Numerical value for the offset angle for the first k-space sampling pattern
    off_set = 0
    # Amount of interleaves used for sampling the k-space
    interleaf = 1

    # %% acquisition settings (1)
    # Numerical value for the amount of time indices
    N = 400
    # Type of initialisation of the flip angle sequence for the optimisation problem ('init_1' is more realistic than
    # 'init_2'.
    init = 'init_1'
    # Repetition times
    TR = (15 * np.ones(N - 1))
    # Defines the rotation axis for the flip angle sequence
    phi = np.zeros((1, N))

    # %% acquisition settings (2) (Changing NOT recommended)
    clip_state = N / 2
    Opt_type = 'without_TR'     # Either 'with_TR' or 'without_TR'
    weighting = 'rCRB'          # Either 'manual' or 'rCRB'
    W_T1 = 0
    W_T2 = 0
    W_M0 = 0

    # %% Define tissue parameters (Scan)
    """" 
    Scans used for reference are scaled down using interpolation to decrease calculation times
    """
    # Desired shape of scaled down reference maps
    shape = (113, 113)
    # Load data
    Data = np.load(r'reference_map.npz')
    x = np.arange(0, 224, 1)
    y = np.arange(0, 224, 1)
    T1_full = np.nan_to_num(Data['T1'])
    f_T1 = interpolate.RectBivariateSpline(x=x, y=y, z=T1_full)
    T2_full = np.nan_to_num(Data['T2'])
    f_T2 = interpolate.RectBivariateSpline(x=x, y=y, z=T2_full)
    M0_field_full = Data['M0']
    f_M0 = interpolate.RectBivariateSpline(x=x, y=y, z=M0_field_full)
    angle_full = -np.nan_to_num(Data['phase'])
    f_angle = interpolate.RectBivariateSpline(x=x, y=y, z=angle_full)

    x_new = np.arange(0, 225, 2)
    y_new = np.arange(0, 225, 2)
    T1 = f_T1(x=x_new, y=y_new)
    T2 = f_T2(x=x_new, y=y_new)
    M0_field = f_M0(x=x_new, y=y_new)
    angle = f_angle(x=x_new, y=y_new)
    phase = np.exp(angle * 1j)

    mask = M0_field > 0.01
    N_vox = np.sum(mask)
    rho_0_scale = 1 / N_vox * np.sum(mask * M0_field)

    T1 = mask * T1
    T2 = mask * T2
    M0_field = mask * M0_field
    phase = mask * phase

    rho = rho_0_scale * mask * phase

    T1 = UEE.Cost.Log_scaling(T1, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation
    T2 = UEE.Cost.Log_scaling(T2, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation

    theta_0 = np.array([np.sum(T1) / np.sum(mask), np.sum(T2) / np.sum(mask), 1, 0])
    if printing:
        print('Theta_0 is: ', theta_0)

    shape = T1.shape
    print('The shape is: ', shape)
    shape_extended = (int(shape[0] + 2 * np.floor(shape[0] / 2)),
                      int(shape[1] + 2 * np.floor(shape[1] / 2)))

# ----------------------------------------------------------------------------------------------------------------------
    # DO NOT CHANGE PARAMETERS BELOW HERE

    # %% Define model parameters
    theta_1 = np.array([T1 - theta_0[0], T2 - theta_0[1], M0_field/rho_0_scale - 1 , np.zeros(shape)])
    # Initialise state matrix
    S = np.zeros((3, 2), dtype=np.complex)
    M0 = np.complex(theta_0[2], theta_0[3])
    S[2, 0] = M0
    # Define flip angle initialisations
    if init == 'init_1':
        alpha = 1/3*np.pi * np.abs(np.sin(np.array(range(N))*(1.5*2*np.pi/N)))
    elif init == 'init_2':
        alpha = (1/18 + 1/2*(1/3 - 1/18))*np.pi + (
                    1/3*(1/3 - 1/18)*np.pi * np.abs(np.sin(np.array(range(N))*(4*2*np.pi/N))))
    alpha[0] = np.pi

    # Plot flip angle initialisation
    plt.figure()
    plt.plot(alpha)
    plt.title('Alpha')
    plt.xlabel('Time index')
    plt.ylabel('alpha [rad]')
    if savefig:
        plt.savefig(save_index + '_Alpha_sequence')

    # %% Define sampling spirals P_j
    G_j = UEE.Cost.Coord(shape_extended)

    method = 'FFT'
    P_j, P_j_tot = UEE.Cost.P_all(G_j, N, shape, shape_extended, under_sampling, golden_angle, spiral, off_set,
                                  interleaf, method, dcf_bool, save_index, savefig, printing)
    P_j_resid = P_j - P_j_tot

    rho_0_star = sc.signal.convolve(rho, P_j_tot, mode='same')

    figure = plt.figure()
    plt.title('rho_0_star')
    fig = plt.imshow(np.abs(rho_0_star))
    fig_bar = figure.colorbar(fig)
    plt.savefig(save_index + '_Rho_0_star')

    if printing:
        print("Creating P_j (1):", datetime.now() - startTime)
        print('Center of P_j has ', np.abs(P_j_tot[int(np.floor(shape_extended[0] / 2)),
                                                   int(np.floor(shape_extended[1] / 2))]) ** 2 /
              np.sum(np.abs(P_j_tot) ** 2) * 100, 'percent of the energy')

    # %% Define problem
    Prob = UEE.Cost(savefig=savefig, savedat=False, save_index=save_index,
                    plot_index=plot_index, printing=False, plot=False, startTime=startTime,
                    shape=shape, shape_extended=shape_extended, T1=T1, T2=T2, rho=rho, mask=mask, N=N, PSF_err=PSF_err,
                    theta_0=theta_0, theta_1=theta_1, S=S, M0=M0, sigma=1, alpha=alpha, TR=TR, phi=phi,
                    clip_state=clip_state, Opt_type=Opt_type, weighting=weighting, W_T1=W_T1, W_T2=W_T2, W_M0=W_M0,
                    G_j=G_j, P_j=P_j, P_j_tot=P_j_tot, P_j_resid=P_j_resid, rho_0_star=rho_0_star, method=method)


    #%% Set Optimisation parameters
    Opt_param = np.zeros((1, N))
    Opt_param[0, :] = alpha
    Opt_param = np.squeeze(Opt_param)

    # Create boundaries for optimsiation
    bounds = Bounds(N * [alpha_min], [np.pi] + (N - 1) * [alpha_max])

    # Construct step constraints
    mat = np.zeros((N - 2, N))
    for ii in range(0, N - 2):
        mat[ii, ii + 1] = 1
        mat[ii, ii + 2] = -1

    lb = np.zeros((1, N - 2))
    lb[0, :] = -alpha_max_step
    lb = np.squeeze(lb)

    ub = np.zeros((1, N - 2))
    ub[0, :] = alpha_max_step
    ub = np.squeeze(ub)

    ineq_cons = [{"type": "ineq", "fun": lambda x: -mat @ x + ub},
                 {"type": "ineq", "fun": lambda x: mat @ x - lb}]



    #%% Optimisation
    print('Start optimisation')
    res = sc_opt.minimize(fun=Prob.full_run,  x0=Opt_param, method='SLSQP', options={'ftol': ftol, 'iprint': 2, 'disp': True, 'maxiter': 800},
                          bounds=bounds,  jac=jac, constraints=ineq_cons)

    #%% Results
    Opt_alpha = res.x
    np.save(save_index+'_Opt_alpha', Opt_alpha)

    plt.figure()
    plt.plot(range(N), alpha, label='Initial alpha')
    plt.plot(range(N), Opt_alpha, label='Optimal alpha')
    plt.xlabel("Time index")
    plt.ylabel("alpha [rad]")
    plt.legend()
    plt.title('Alpha optimised')
    plt.ylim([alpha_min-0.02*np.pi, alpha_max+0.02*np.pi])
    plt.savefig(save_index + '_Alpha optimised N = ' + str(N))

    np.savez(save_index + '_Settings', PSF_err=PSF_err, dcf_bool=dcf_bool,
             under_sampling=under_sampling, golden_angle=golden_angle,
             spiral=spiral, plot_index=plot_index, alpha=alpha,
             ftol=ftol, T1=T1, T2=T2, M0_field=M0_field, phase=phase, interleaf=interleaf)
    print('Runtime: ', datetime.now() - startTime)




