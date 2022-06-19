# Author: D.G.J. Heesterbeek
# Delft University of Technology
# Last revision: 11-06-2022
"""
Script to predict the undersampling error.
The derivation of the prediction of the undersampling error is based on the paper:
"Understanding the combined effect of k -space undersampling and transient states excitation in
MR Fingerprinting reconstructions", by C.C. Stolk and A. Sbrizzi.

External input: k-space sampling pattern and (optionally) scan for ground truth.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy as sc
import sigpy as sp
import sigpy.mri as spmri
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

import Optimisation

# %% Acquisition settings (1)
if __name__ == '__main__':
    # Numerical value for the amount of time indices
    N = 400
    # Construction of the conventional sequence with 3 arches
    alpha_conv = 1 / 3 * np.pi * np.abs(np.sin(np.array(range(N)) * (1.5 * 2 * np.pi / N)))
    alpha_conv[0] = np.pi
    # Sequences for which the undersampling error is predicted. The suffix "x_32" with x={1,2,3} determines the
    # undersampling rate
    dict_1 = {'save_index': 'conv_1_32', 'alpha': alpha_conv}
    dict = [dict_1]
    # Repetition times
    TR = (15 * np.ones(N - 1))
    # Defines the rotation axis for the flip angle sequence
    phi = np.zeros((1, N))

    for ii in range(len(dict)):
        startTime = datetime.now()
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
        # String that determines the type of reference data or ground truth (Either 'Checkerboard' or 'Scan')
        tissue_param = 'Checkerboard'
        print('The used tissue parameter is: ', tissue_param)
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

        # ----------------------------------------------------------------------------------------------------------------------
        # DO NOT CHANGE PARAMETERS BELOW HERE
        # %% Acquisition settings (2) (Changing NOT recommended)
        clip_state = N / 2
        Opt_type = 'without_TR'  # Either 'with_TR' or 'without_TR'
        weighting = 'rCRB'  # Either 'manual' or 'rCRB'
        W_T1 = 0
        W_T2 = 0
        W_M0 = 0
        if dict[ii]['save_index'].endswith('_1_32'):
            print('use 1 interleaf')
            interleaf = 1
        elif dict[ii]['save_index'].endswith('_2_32'):
            print('use 2 interleaves')
            interleaf = 2
        elif dict[ii]['save_index'].endswith('_3_32'):
            print('use 3 interleaves')
            interleaf = 3
        save_index = dict[ii]['save_index']


class Cost():
    def __init__(self, savefig, savedat, save_index, plot_index, printing, plot, startTime,
                 shape, shape_extended, T1, T2, rho, mask, N, PSF_err, theta_0, theta_1, S, M0, sigma, alpha, TR, phi,
                 clip_state, Opt_type, weighting, W_T1, W_T2, W_M0, G_j, P_j, P_j_tot, P_j_resid, rho_0_star,
                 method):

        # %% Settings
        self.savefig = savefig
        self.savedat = savedat
        self.PSF_err = PSF_err
        self.save_index = save_index
        self.plot_index = plot_index
        self.printing = printing
        self.plot = plot
        self.startTime = startTime

        # %% Phantom
        self.shape = shape
        self.shape_extended = shape_extended
        self.T1 = T1
        self.T2 = T2
        self.rho = rho
        self.mask = mask

        # %% Variable acquisition
        self.N = N
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.S = S
        self.M0 = M0
        self.sigma = sigma
        self.alpha = alpha
        self.TR = TR
        self.phi = phi
        self.clip_state = clip_state
        self.Opt_type = Opt_type
        self.weighting = weighting
        self.W_T1 = W_T1
        self.W_T2 = W_T2
        self.W_M0 = W_M0

        # %% Define sampling spirals P_j
        self.G_j = G_j
        self.P_j = P_j
        self.P_j_tot = P_j_tot
        self.P_j_resid = P_j_resid
        self.rho_0_star = rho_0_star
        self.method = method

    # %% Functions independent of acquisition parameters (run without using __init__ values)
    @staticmethod
    def Zero_padding(image, shape):
        """
        This function is applied when the image has even dimensions and returns an image with uneven dimension by zero
        padding
        :param image: Image to be zero-padded
        :param shape: Desired shape
        :return: Image with desired shape
        """
        shape_image = np.array(image.shape)
        shape = np.array(shape)
        add_block = (shape - shape_image) / 2
        add_block = add_block.astype(int)
        image_new = np.zeros(shape, dtype=image.dtype)
        image_new[add_block[0]:(add_block[0] + shape_image[0]), add_block[1]:(add_block[1] + shape_image[1])] = image

        return image_new

    @staticmethod
    def Log_scaling(field, indicator):
        """
        Applies logarithmic operation or inverse operation on field
        :param field: variables on which the operation is applied
        :param indicator: 0: apply logarithmic operation. 1: apply inverse operation
        :return: processed field
        """
        field_return = np.zeros(field.shape)
        if indicator == 0:
            field_return[field != 0] = np.log(field[field != 0])

        if indicator == 1:
            field_return[field != 0] = np.exp(field[field != 0])

        return field_return

    @staticmethod
    def Create_mask(width, rad_frac):
        """
        Create circular boolean mask for checkerboard pattern.
        :param width: Image width
        :param rad_frac: Radius of circular mask (between 0 and 1)
        :return: nxn boolean array (n is the image shape in both directions)
        """
        uneven_bool = bool(2 * (width / 2 - np.floor(width / 2)))
        if uneven_bool:
            x_arr, y_arr = np.mgrid[0:width, 0:width]
            center = np.floor(width / 2)
        else:
            x_arr, y_arr = np.mgrid[0:width, 0:width] + 0.5
            center = np.floor(width / 2)
        dist_center = np.sqrt((x_arr - center) ** 2 + (y_arr - center) ** 2)
        radius = np.min([dist_center[0, int(center)], dist_center[int(center), 0]])
        mask = dist_center <= radius * rad_frac

        return mask

    @staticmethod
    def phase_field(shape, order):
        """
        Create phase field for numerical experiment with checkerboard phantom (figure 3)
        :param shape: shape of the phase field
        :param order: determines the amount of variation in the phase pattern
        :return: NxN complex phase array with image dimensions
        """
        border = (int(np.floor(shape[0] / 2)), int(np.floor(shape[1] / 2)))
        A, B = np.mgrid[-border[0]:border[0] + 1, -border[1]:border[1] + 1]
        dist = A ** 2 + B ** 2
        dist_rescale = (dist * order / dist[border[0], 0]) * 2 * np.pi
        phase = np.exp(dist_rescale * 1j)

        return phase

    @staticmethod
    def Spiral_coord(index, bounds, shape, golden_angle, spiral, off_set, interleaf, **kwargs):
        """
        Create k-space sampling coordinates.
        :param index: Determines angle over which the spiral is rotated
        :param bounds (string): Either "2-pi" or "full" to determine the range of the spiral. Full will use the image shape.
        :param shape: Image shape
        :param golden_angle: boolean, if 'True' the k-space sampling pattern is rotated with the golden angle, if
        'False' with the standard 2*pi/32 used by Philips
        :param spiral: Either 'Philips_spiral' or 'var_density'. Determines k-space sampling pattern
        :param off_set: Offset angle for first k-space sampling pattern
        :param interleaf: Amount of interleaves used for one time indices
        :return: kx2 array (k is the amount of sampled k-space points in one spiral)
        """
        if spiral == 'Philips_spiral':

            coord = np.load('Single spiral.npz')
            coord = coord['Coords']
            coord = np.transpose(coord)

            len_one_spiral = coord.shape[0]
            # Scale coordinages (1.45 if edges should be sampled)
            if bounds == '2-pi':
                coord = coord * (1.00 * np.pi / 0.5)
            elif bounds == 'full':
                coord = coord * (1.00 * shape[0] / (2 * 0.5))
        elif spiral == 'var_density':
            # Generate spiral trajectory
            # spiral(fov, N, f_sampling(=1/4), R, ninterleaves, alpha, max_grad_amp, max_slew_rate, gamma=2.678e8):
            coord = spmri.spiral(0.24, 120, 1 / 4, 32, 1, 6, 0.03, 150)
            len_one_spiral = coord.shape[0]
            # Scale coordinages (1.45 if edges should be sampled)
            if bounds == '2-pi':
                coord = coord * (1.00 * np.pi / 256)
            elif bounds == 'full':
                coord = coord * (1.00 * shape[0] / (2 * 256))

        if golden_angle:
            # rotate by index * golden-angle
            gold_ang = -np.pi * (3 - np.sqrt(5))
            realcoord = []
            imgcoord = []
            for ii in range(interleaf):
                comp_coord = (coord[:, 0] + 1j * coord[:, 1]) * np.exp(
                    (1j * gold_ang * (index * interleaf + ii)) + 1j * off_set)
                realcoord = np.append(realcoord, np.real(comp_coord))
                imgcoord = np.append(imgcoord, np.imag(comp_coord))
            output = np.stack((realcoord, imgcoord), axis=1)
        else:
            # rotate by index * angle
            fac = 32
            angle = + 2 * np.pi / fac
            realcoord = []
            imgcoord = []
            for ii in range(interleaf):
                comp_coord = (coord[:, 0] + 1j * coord[:, 1]) * np.exp(
                    (1j * angle * (index + fac // interleaf * ii)) + 1j * off_set)
                realcoord = np.append(realcoord, np.real(comp_coord))
                imgcoord = np.append(imgcoord, np.imag(comp_coord))
            output = np.stack((realcoord, imgcoord), axis=1)

        if bounds == '2-pi':
            mask_x = np.abs(output[:, 0]) > np.pi
            mask_y = np.abs(output[:, 1]) > np.pi
        elif bounds == 'full':
            mask_x = np.abs(output[:, 0]) > shape[1] / 2
            mask_y = np.abs(output[:, 1]) > shape[0] / 2

        bool_mask = np.invert(mask_x + mask_y)

        return output, bool_mask, len_one_spiral

    @staticmethod
    def Spiral_dcf(spiral_coord, dcf_bool):
        """
        Function to generate density compensation weights.
        :param spiral_coord: Spiral coordinates
        :param dcf_bool: if False this function will return an array of ones (no density compensation weighting)
        :return: k array with density compensation weights
        """
        if dcf_bool:
            dcf = np.zeros((spiral_coord.shape[0], 1))
            k_mag = np.abs((spiral_coord[:, 0] ** 2 + spiral_coord[:, 1] ** 2) ** 0.5)
            kdiff = np.append([0], np.diff(k_mag))
            dcf[:, 0] = k_mag * np.abs(kdiff)
        else:
            dcf = np.ones((spiral_coord.shape[0], 1))

        return dcf

    @staticmethod
    def Coord(shape):
        """
        Helper function for image coordinates. Coord[0, :, :] are the x-coordinates and Coord[1, :, :] are the y coordinates
        :param shape: Image shape
        :return: 2xnxn array
        """
        coord_mat = np.zeros([2, shape[0], shape[1]])
        for ii in range(2):
            for iii in range(shape[ii]):
                arr = np.arange(-np.floor(shape[ii] / 2), -np.floor(shape[ii] / 2) + shape[ii])
                if ii == 0:  # y-coordinates
                    coord_mat[1, :, :] = np.repeat(arr[:, None], shape[1], axis=1)
                if ii == 1:  # x-coordinates
                    coord_mat[0, :, :] = np.repeat(arr[None, :], shape[0], axis=0)

        return coord_mat

    @staticmethod
    def P_single(G_j, shape, spiral, spiral_dcf):
        """
        Calculate single point spread function. Inefficient (redundant) implementation of P_single_fft.
        :param G_j: Image coordinates
        :param shape: Image shape
        :param spiral: Spiral coordinates
        :param spiral_dcf: Density compensation values
        :return: (n + 2*np.floor(n/2))x(n + 2*np.floor(n/2)) array
        """
        m1 = shape[0]
        m2 = shape[1]
        P_j = np.zeros((G_j[0, :, :].shape[0], G_j[0, :, :].shape[1]), dtype=complex)
        for ii in tqdm(range(G_j[0, :, :].shape[0]), desc='Wave summation for PSF'):
            for iii in range(G_j[0, :, :].shape[1]):
                P_j[ii, iii] = 1 / (m1 * m2) * np.sum(
                    spiral_dcf * np.exp(1j * np.dot(spiral, G_j[:, ii, iii][:, None])))

        return P_j

    @staticmethod
    def P_single_fft(shape_extended, spiral, spiral_dcf):
        """
        Calculate single point spread function in an efficient FFT manner
        :param shape_extended: Extended image shape
        :param spiral: Spiral coordinates
        :param spiral_dcf: Density compensation values
        :return: (n + 2*np.floor(n/2))x(n + 2*np.floor(n/2)) array
        """
        nufftlinop = sp.linop.NUFFT(shape_extended, spiral)
        P_j = nufftlinop.H * (spiral_dcf)
        return P_j

    @staticmethod
    def P_all(G_j, N, shape, shape_extended, under_sampling, golden_angle, spiral, off_set, interleaf, method, dcf_bool,
              save_index, savefig, printing):
        """
        Calculate all time dependent point spread functions
        :param G_j: Image coordinates
        :param N: Amount of time indices
        :param shape: Image shape
        :param shape_extended: Extended image shape for convolution
        :param under_sampling: Boolean, "True" if undersampling should be taken into account or "False" if it should not
        :param golden_angle: Boolean, "True" if golden angle rotation should be applied or "False" if conventional
        Philips rotation shoud be applied
        :param golden_angle: Boolean, if 'True' the k-space sampling pattern is rotated with the golden angle, if
        'False' with the standard 2*pi/32 used by Philips
        :param spiral: Either 'Philips_spiral' or 'var_density'. Determines k-space sampling pattern
        :param off_set: Offset angle for first k-space sampling pattern
        :param interleaf: Amount of interleaves used for one time indices
        :param method: "Explicit" for an explicit calculation of P, "FFT" for using the Fast Fourier Transform
        :param dcf_bool: Boolean, if 'True' the density compensation weights are taken into account
        :param savefig: Boolean, if 'True' the time averaged point spread function is saved
        :param printing: Boolean, if 'True' scale factor related statistics are printed in the console
        :return: P_j is (n + 2*np.floor(n/2))x(n + 2*np.floor(n/2))xN array and P_j_tot is
        (n + 2*np.floor(n/2))x(n + 2*np.floor(n/2)) array
        """
        P_j = np.zeros([N, shape_extended[0], shape_extended[1]], dtype=complex)
        P_j_tot = np.zeros([shape_extended[0], shape_extended[1]], dtype=complex)

        if under_sampling == True:
            for ii in tqdm(range(N), desc='Making the PSF'):
                if method == "Explicit":
                    spiral_coord, bool_mask, len_one_spiral = Cost.Spiral_coord(ii, bounds='2-pi', shape=shape,
                                                                                golden_angle=golden_angle,
                                                                                spiral=spiral, off_set=off_set,
                                                                                interleaf=interleaf)
                    dcf = Cost.Spiral_dcf(spiral_coord[:len_one_spiral, :], dcf_bool)
                    spiral_coord = spiral_coord[bool_mask]
                    dcf = np.tile(dcf, (interleaf, 1))
                    dcf = dcf[bool_mask]
                    P_j[ii, :, :] = Cost.P_single(G_j, shape, spiral_coord, dcf)
                elif method == "FFT":
                    spiral_coord, bool_mask, len_one_spiral = Cost.Spiral_coord(ii, bounds='full', shape=shape_extended,
                                                                                golden_angle=golden_angle,
                                                                                spiral=spiral, off_set=off_set,
                                                                                interleaf=interleaf)
                    dcf = Cost.Spiral_dcf(spiral_coord[:len_one_spiral, :], dcf_bool)
                    spiral_coord = spiral_coord[bool_mask]
                    dcf = np.tile(dcf, (interleaf, 1))
                    dcf = dcf[bool_mask]
                    P_j[ii, :, :] = Cost.P_single_fft(shape_extended, spiral_coord, dcf)
                else:
                    print('Not a valid argument')
                    exit()

                # %% Plot spirals
                if ii == 0 and printing:
                    plt.figure()
                    plt.plot(spiral_coord[:, 0], spiral_coord[:, 1])
                    plt.title('Spiral k-space sampling')
                    plt.xlim([-1 * np.floor(shape_extended[1] / 2), 1 * np.floor(shape_extended[1] / 2)])
                    plt.ylim([-1 * np.floor(shape_extended[0] / 2), 1 * np.floor(shape_extended[0] / 2)])
                    if savefig:
                        plt.savefig(save_index + '_Spiral_k_space_sampling')

            P_j_tot = 1 / N * np.sum(P_j, axis=0)
            if printing:
                print('Sum of P_j_tot before scaling is: ', np.sum(P_j_tot))

            # %% Scaling
            # scale = np.sqrt(np.sum(np.abs(P_j_tot)**2))
            scale = np.abs(np.sum(P_j_tot))
            P_j = 1 / scale * P_j
            P_j_tot = 1 / scale * P_j_tot

            if printing:
                print('The scale factor is: ', scale)
                print('Sum of P_j_tot after scaling is: ', np.sum(P_j_tot))
        elif under_sampling == False:
            spiral_coord_help = Cost.Coord(shape)
            spiral_coord = np.zeros((shape[0] * shape[1], 2))
            for ii in range(shape[0]):
                for iii in range(shape[1]):
                    spiral_coord[ii * shape[0] + iii, :] = spiral_coord_help[:, ii, iii] / shape[0] * 2 * np.pi

            dcf = np.ones((spiral_coord.shape[0], 1))

            PSF = Cost.P_single(G_j, shape, spiral_coord, dcf)
            P_j[range(N), :, :] = PSF

            plt.figure()
            plt.title('Spiral k-space sampling')
            plt.plot(spiral_coord[:, 0], spiral_coord[:, 1])
            if savefig:
                plt.savefig(save_index + '_Spiral_k_space_sampling')

            P_j_tot = 1 / N * np.sum(P_j, axis=0)
            scale = np.sqrt(np.sum(np.abs(P_j_tot) ** 2))
            P_j = 1 / scale * P_j
            P_j_tot = 1 / scale * P_j_tot

            plt.figure()
            plt.title('$P_{tot}$')
            plt.imshow(np.abs(P_j_tot))
            if savefig:
                plt.savefig(save_index + '_P_tot')

        return P_j, P_j_tot

    # %% Functions dependent on acquisition parameters
    def Evaluation_data(self, Opt_param):
        """
        Simulate the transverse magnetisation and its derivative.
        :param Opt_param: Flip angle pattern
        :return: M (N) is the evolution of the transverse magnetisation and Mat (4, N) are the derivatives.
        """
        Prob = Optimisation.Cost(S_init=self.S, M0=self.M0, N=self.N, phi=self.phi, T1=np.exp(self.theta_0[0]),
                                 T2=np.exp(self.theta_0[1]), sigma=self.sigma, print=False,
                                 clip_state=self.clip_state, TR_set=self.TR, Opt_type=self.Opt_type,
                                 weighting=self.weighting, W_T1=self.W_T1, W_T2=self.W_T2, W_M0=self.W_M0)

        M = Prob.signal(alpha=Opt_param)
        der_m_T1_raw, der_m_T2_raw, der_m_M0 = Prob.dm_dT(alpha=Opt_param)

        der_m_T1 = (der_m_T1_raw[0, :] + 1j * der_m_T1_raw[1, :]) * np.exp(self.theta_0[0])
        der_m_T2 = (der_m_T2_raw[0, :] + 1j * der_m_T2_raw[1, :]) * np.exp(self.theta_0[1])
        der_m_M0 = der_m_M0[0, :] + 1j * der_m_M0[1, :]
        Mat = np.array([der_m_T1, der_m_T2, der_m_M0, der_m_M0 * 1j])

        return M, Mat

    def S_matrices(self, M, Mat):
        """
        Calculate the N and residual S-matrices from the paper (see top of the script)
        :param M: Evolution of the transverse magnetisation
        :param Mat: Evolution of the derivative of the transverse magnetisation
        :return: N (4x4) matrix, S_1_0_resid (extended shape (see P_j_tot)), S_1_1_resid (extended shape (see P_j_tot))
        """
        S_1_0_resid = np.zeros([4, self.shape_extended[0], self.shape_extended[1]], dtype=np.complex64)
        S_1_1_resid = np.zeros([4, 4, self.shape_extended[0], self.shape_extended[1]], dtype=np.complex64)

        N = np.dot(np.conjugate(Mat), Mat.transpose())

        for ii in range(4):
            S_1_0_resid[ii, :, :] = np.sum(self.P_j_resid * np.conjugate(Mat[ii, :][:, None, None]) * M[:, None, None],
                                           axis=0)
            for iii in range(4):
                S_1_1_resid[ii, iii, :, :] = np.sum(
                    self.P_j_resid * np.conjugate(Mat[ii, :][:, None, None]) * Mat[iii, :][:, None, None], axis=0)

        return N, S_1_0_resid, S_1_1_resid

    def Error_matrices(self, S_1_0_resid, S_1_1_resid):
        """
        Calculate auxiliary error matrices E_1 and E_2
        :param S_1_0_resid: Residual matrix
        :param S_1_1_resid: Residual matrix
        :return: E_1 (nxn) and E_2 (nxn)
        """
        E_1 = np.zeros([4, self.shape[0], self.shape[1]])
        E_2 = np.zeros([4, self.shape[0], self.shape[1]])
        sum_term = np.zeros([4, self.shape[0], self.shape[1]], dtype=complex)
        for ii in range(4):
            E_1[ii, :, :] = np.real(
                np.conjugate(self.rho_0_star) * sc.signal.convolve(self.rho, S_1_0_resid[ii, :, :], mode='same'))
            for iii in range(4):
                sum_term[iii, :, :] = sc.signal.convolve(self.rho * self.theta_1[iii, :, :], S_1_1_resid[ii, iii, :, :],
                                                         mode='same')
            E_2[ii, :, :] = np.real(np.sum(np.conjugate(self.rho_0_star) * sum_term, axis=0))

        return E_1, E_2

    def Theta_1_star(self, E_1, E_2, N):
        """
        Calculate theta_1_star (estimation of the deviation from theta_0) and the explicit error terms
        param E_1: Error matrix 1
        param E_2: Error matrix 2
        param N: Fisher information matrix
        return: Error components used for the prediction
        """
        Error_term_1 = np.zeros([4, self.shape[0], self.shape[1]])
        Error_term_2 = np.zeros([4, self.shape[0], self.shape[1]])
        P_theta_1 = np.zeros([4, self.shape[0], self.shape[1]], dtype=complex)

        for ii in range(4):
            P_theta_1[ii, :, :] = sc.signal.convolve(self.rho * self.theta_1[ii, :, :], self.P_j_tot, mode='same')

        NP_theta_1 = np.zeros([4, self.shape[0], self.shape[1]], dtype=complex)
        Ninv_rho_NP_theta_1 = np.zeros([4, self.shape[0], self.shape[1]])

        N_inv = np.linalg.inv(np.real(N))
        for iii in range(self.shape[0]):
            Error_term_1[:, iii, :] = np.dot(N_inv, E_1[:, iii, :])
            Error_term_2[:, iii, :] = np.dot(N_inv, E_2[:, iii, :])

            NP_theta_1[:, iii, :] = np.dot(N, P_theta_1[:, iii, :])
            rho_NP_theta_1 = np.real(np.conjugate(self.rho_0_star) * NP_theta_1)
            Ninv_rho_NP_theta_1[:, iii, :] = np.dot(N_inv, rho_NP_theta_1[:, iii, :])

        if self.PSF_err:
            theta_1_star = 1 / (np.abs(self.rho_0_star) ** 2) * (Ninv_rho_NP_theta_1 + Error_term_1 + Error_term_2)
        else:
            theta_1_star = self.theta_1 + 1 / (np.abs(self.rho_0_star) ** 2) * (Error_term_1 + Error_term_2)

        return theta_1_star, Ninv_rho_NP_theta_1, Error_term_1, Error_term_2

    def full_run(self, Opt_param):

        # %% Calculate evolution data
        M, Mat = Cost.Evaluation_data(self, Opt_param)
        if np.sum(np.real(M) ** 2) != 0:
            print('Transverse magnetisation contains real components!')

        # %% Define matrices used for error calculation
        N, S_1_0_resid, S_1_1_resid = Cost.S_matrices(self, M, Mat)

        # %% Calculate error matrices
        E_1, E_2 = Cost.Error_matrices(self, S_1_0_resid, S_1_1_resid)

        # %% Calculate Theta_1_star
        theta_1_star, Ninv_rho_NP_theta_1, Error_term_1, Error_term_2 = Cost.Theta_1_star(self, E_1, E_2, N)

        # %% Inverse log
        tissue_map = np.zeros([2, self.shape[0], self.shape[1]])
        for ii in range(2):
            tissue_map[ii, :, :] = Cost.Log_scaling(self.mask * (self.theta_0[ii] + theta_1_star[ii, :, :]), 1)
        rho_1 = theta_1_star[2, :, :] + theta_1_star[3, :, :] * 1j
        rho_star = self.rho_0_star * (1 + rho_1)

        Ninv_rho_NP_theta_1 = Cost.Log_scaling(self.mask * 1 / (np.abs(self.rho_0_star) ** 2) * Ninv_rho_NP_theta_1, 1)
        Error_term_1 = Cost.Log_scaling(self.mask * 1 / (np.abs(self.rho_0_star) ** 2) * Error_term_1, 1)
        Error_term_2 = Cost.Log_scaling(self.mask * 1 / (np.abs(self.rho_0_star) ** 2) * Error_term_2, 1)

        T1_ground = Cost.Log_scaling(self.T1, 1)
        T2_ground = Cost.Log_scaling(self.T2, 1)

        # %% Calcute RMS error
        tissue = np.zeros([2, self.shape[0], self.shape[1]])
        tissue[0, :, :] = T1_ground
        tissue[1, :, :] = T2_ground

        T = np.sum(self.mask)

        tissue_div = np.zeros([2, self.shape[0], self.shape[1]])
        frac_error = np.zeros([2, self.shape[0], self.shape[1]])
        RMS_rel_bias = np.zeros([2])

        error = self.mask * (tissue_map[self.plot_index, :, :] - tissue[self.plot_index, :, :])

        for ii in range(2):
            tissue_div[ii, :, :] = np.copy(tissue[ii, :, :])
            tissue_div[ii, tissue_div[ii, :, :] == 0] = 1
            frac_error[ii, :, :] = self.mask * (tissue_map[ii, :, :] - tissue[ii, :, :]) / tissue_div[ii, :, :]

            squared_sum_rel_err = np.sum((frac_error[ii, :, :]) ** 2)
            RMS_rel_bias[ii] = np.sqrt(1 / T * squared_sum_rel_err)

        result = 100 * (np.sum(RMS_rel_bias)) / 2

        if self.printing:
            print('RMS relative error of the bias T1: ', RMS_rel_bias[0])
            print('RMS relative error of the bias T2: ', RMS_rel_bias[1])
        if self.savedat:
            np.savez(self.save_index + '_Data', frac_error=frac_error, tissue_map=tissue_map, rho_star=rho_star,
                     Error_term_1=Error_term_1, Error_term_2=Error_term_2, N=N, theta_1_star=theta_1_star,
                     rho_0_star=rho_0_star, Ninv_rho_NP_theta_1=Ninv_rho_NP_theta_1, RMS_rel_bias=RMS_rel_bias)
        if self.plot:
            Cost.Plot(self, tissue_map, rho_star, RMS_rel_bias, frac_error)

        return result

    def Plot(self, tissue_map, rho_star, RMS_rel_bias, frac_error):
        fig = plt.figure()
        plt.title('Predicted MRF error in T1. RMS = ' + str(np.round(100 * RMS_rel_bias[0], 1)))
        im = plt.imshow(frac_error[0, :, :], cmap='RdBu_r')
        fig_bar = fig.colorbar(im, format=mtick.PercentFormatter(xmax=1))
        plt.clim(-0.4, 0.4)
        fig_bar.set_label('Error')
        if self.savefig:
            plt.savefig(self.save_index + 'T1_Predicted_MRF_error')

        fig = plt.figure()
        plt.title('Predicted MRF error in T2. RMS = ' + str(np.round(100 * RMS_rel_bias[1], 1)))
        im = plt.imshow(frac_error[1, :, :], cmap='RdBu_r')
        fig_bar = fig.colorbar(im, format=mtick.PercentFormatter(xmax=1))
        plt.clim(-0.4, 0.4)
        fig_bar.set_label('Error')
        if self.savefig:
            plt.savefig(self.save_index + 'T2_Predicted_MRF_error')

        fig = plt.figure()
        plt.title(r'$T_1$')
        est_tot = self.mask * (tissue_map[0, :, :])
        im = plt.imshow(est_tot, norm=SymLogNorm(vmin=150, vmax=5000, linthresh=1, base=10))
        fig_bar = fig.colorbar(im)
        fig_bar.set_label('Tissue parameter [ms]')
        if self.savefig:
            plt.savefig(self.save_index + '_T1')

        fig = plt.figure()
        plt.title(r'$T_2$')
        est_tot = self.mask * (tissue_map[1, :, :])
        im = plt.imshow(est_tot, cmap='magma', norm=SymLogNorm(vmin=30, vmax=1000, linthresh=1, base=10))
        fig_bar = fig.colorbar(im)
        fig_bar.set_label('Tissue parameter [ms]')
        if self.savefig:
            plt.savefig(self.save_index + '_T2')

        fig = plt.figure()
        plt.title('Phase $rho^*$')
        im = plt.imshow(self.mask * np.angle(rho_star), cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
        fig_bar = fig.colorbar(im)
        if self.savefig:
            plt.savefig(self.save_index + '_Phase')

        fig = plt.figure()
        plt.title('Magnitude $rho^*$')
        im = plt.imshow(self.mask * np.abs(rho_star), cmap='gray')
        fig_bar = fig.colorbar(im)
        if self.savefig:
            plt.savefig(self.save_index + '_mag_rho')


class JAC():
    """
    This class is used in 'main_UEE_phase.py' to calculate the Jacobian.
    """

    def __init__(self, Prob, Opt_param, base, step):
        self.Prob = Prob
        self.Opt_param = Opt_param
        self.base = base
        self.step = step

    def jac_step(self, ii):
        Opt_one_step_forward = self.Opt_param
        Opt_one_step_forward[ii] = self.Opt_param[ii] + self.step[ii]
        one_step_forward = self.Prob.full_run(Opt_one_step_forward)

        Jac_step = (one_step_forward - self.base) / self.step[ii]
        return Jac_step


if __name__ == '__main__':
    # %% Define tissue parameters (Checkerboard)
    if tissue_param == 'Checkerboard':
        base = (11, 11)  # Checkerboard pattern (15)
        block_size = 11  # Block size (15)
        rad_frac = 0.7
        shape = (block_size * base[0], block_size * base[1])
        shape_extended = (int(block_size * base[0] + 2 * np.floor(block_size * base[0] / 2)),
                          int(block_size * base[1] + 2 * np.floor(block_size * base[1] / 2)))
        checkerboard = np.indices(base).sum(axis=0) % 2
        T1 = 750 + 500 * np.repeat(np.repeat(checkerboard, block_size, axis=0), block_size, axis=1)
        T2 = 70 + 20 * np.repeat(np.repeat(checkerboard, block_size, axis=0), block_size, axis=1)
        mask = Cost.Create_mask(shape[0], rad_frac) * np.ones((T1.shape[0], T1.shape[1]))
        rho = mask * np.ones((T1.shape[0], T1.shape[1]))

        T1 = mask * T1
        T2 = mask * T2

        phase = Cost.phase_field(shape=shape, order=1)
        rho = rho * phase

        theta_0 = np.array([1000, 80, 1, 0], dtype=float)

        T1 = Cost.Log_scaling(T1, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation
        T2 = Cost.Log_scaling(T2, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation
        theta_0[:2] = np.log(theta_0[:2])

        if printing:
            print('Theta_0 is: ', theta_0)

    # %% Define tissue parameters (Scan)
    elif tissue_param == 'Scan':
        shape = (225, 225)
        Data = np.load(r'reference_map.npz')

        M0_field = Cost.Zero_padding(Data['M0'], shape)
        mask = M0_field > 0.01
        N_vox = np.sum(mask)
        rho_0_scale = 1 / N_vox * np.sum(mask * M0_field)

        T1 = mask * Cost.Zero_padding(np.nan_to_num(Data['T1']), shape)
        T2 = mask * Cost.Zero_padding(np.nan_to_num(Data['T2']), shape)
        phase = mask * Cost.Zero_padding(np.exp(-np.nan_to_num(Data['phase']) * 1j), shape)
        rho = rho_0_scale * mask * phase

        T1 = Cost.Log_scaling(T1, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation
        T2 = Cost.Log_scaling(T2, 0)  # arguments: field, [0, 1]. 0: apply logarithm. 1: apply inverse operation

        theta_0 = np.array([np.sum(T1) / np.sum(mask), np.sum(T2) / np.sum(mask), 1, 0])
        if printing:
            print('Theta_0 is: ', theta_0)

        shape = T1.shape
        print('The shape is: ', shape)
        shape_extended = (int(shape[0] + 2 * np.floor(shape[0] / 2)),
                          int(shape[1] + 2 * np.floor(shape[1] / 2)))

    else:
        print("No such tissue parameter available")
        exit()

    # %% Define acquistion parameters
    if tissue_param == 'Checkerboard':
        theta_1 = np.array([T1 - theta_0[0], T2 - theta_0[1], np.zeros(shape), np.zeros(shape)])
    elif tissue_param == 'Scan':
        theta_1 = np.array([T1 - theta_0[0], T2 - theta_0[1], M0_field / rho_0_scale - 1, np.zeros(shape)])

    # Initialise state matrix
    S = np.zeros((3, 2), dtype=complex)
    M0 = 1
    S[2, 0] = M0
    alpha = dict[ii]['alpha']

    # %% Define Point Spread Funtions P_j
    G_j = Cost.Coord(shape_extended)
    method = 'FFT'
    P_j, P_j_tot = Cost.P_all(G_j, N, shape, shape_extended, under_sampling, golden_angle, spiral, off_set, interleaf,
                              method, dcf_bool, save_index, savefig, printing)
    P_j_resid = P_j - P_j_tot
    rho_0_star = sc.signal.convolve(rho, P_j_tot, mode='same')

    if printing:
        print("Creating P_j (1):", datetime.now() - startTime)
        print('Center of P_j has ',
              np.abs(P_j_tot[int(np.floor(shape_extended[0] / 2)), int(np.floor(shape_extended[1] / 2))]) ** 2 / np.sum(
                  np.abs(P_j_tot) ** 2) * 100, 'percent of the energy')

    # %% Define problem
    Prob = Cost(savefig=savefig, savedat=savedat, save_index=save_index,
                plot_index=plot_index,
                printing=printing, plot=True, startTime=startTime, shape=shape,
                shape_extended=shape_extended, T1=T1, T2=T2, rho=rho, mask=mask, N=N, PSF_err=PSF_err,
                theta_0=theta_0, theta_1=theta_1, S=S, M0=M0, sigma=1, alpha=alpha, TR=TR, phi=phi,
                clip_state=clip_state, Opt_type=Opt_type, weighting=weighting, W_T1=W_T1, W_T2=W_T2, W_M0=W_M0, G_j=G_j,
                P_j=P_j, P_j_tot=P_j_tot, P_j_resid=P_j_resid, rho_0_star=rho_0_star, method=method)

    # %% Single run
    np.savez(save_index + '_Settings', PSF_err=PSF_err, dcf_bool=dcf_bool, under_sampling=under_sampling,
             golden_angle=golden_angle, interleaf=interleaf, spiral=spiral, tissue_param=tissue_param,
             plot_index=plot_index, alpha=alpha)

    Opt_param = alpha
    Result = Prob.full_run(Opt_param)
    print('Opt val is: ', Result)
