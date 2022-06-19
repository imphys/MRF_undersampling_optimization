# Author: D.G.J. Heesterbeek
# Delft University of Technology
# Last revision: 11-06-2022
"""
Script to calculate the cost function value for Cramér-Rao Bound (CRB) optimisations. Some functions in this script are
used in UEE_phase.py to calculate the evaluation of the transverse magnetisation and their derivatives to the tissue
parameters using the EPG formalism.
"""

import numpy as np


class Cost:
    def __init__(self, S_init, M0, N, phi, T1, T2, sigma, print, clip_state, TR_set, Opt_type, weighting,
                 **kwargs):
        self.S_init = S_init
        self.M0 = M0
        self.N = N
        self.phi = np.squeeze(phi)
        self.T1 = T1
        self.T2 = T2
        self.sigma = sigma
        self.print = print
        self.clip_state = clip_state
        self.TR_set = TR_set
        self.Opt_type = Opt_type
        self.weighting = weighting
        self.kwargs = kwargs

    def Grad(self, S):
        """
        Perform dephasing step
        :param S: Current state matrix
        :return: State matrix for next time step
        """
        if S.shape[1] < self.clip_state:
            S_new = np.zeros((3, S.shape[1] + 1), dtype=complex)
            S_new[0, 1:] = S[0, :]
            S_new[0, 0] = np.conjugate(S[1, 1])
            S_new[1, :-2] = S[1, 1:]
            S_new[2, :-1] = S[2, :]

        else:
            S_new = np.zeros((3, S.shape[1]), dtype=complex)
            S_new[0, 1:] = S[0, :-1]
            S_new[0, 0] = np.conjugate(S[1, 1])
            S_new[1, :-1] = S[1, 1:]
            S_new[2, :] = S[2, :]
        return S_new

    def RF_matrix(self, alpha, phi):
        """
        Caluculate RF_matrix
        :param alpha: flip angle
        :param phi: angle of the rotation axis with the x-axis
        :return: RF matrix
        """
        R = np.empty((3, 3), dtype=complex)

        e_iphi = np.exp(1j * phi)
        a = np.cos(alpha / 2) ** 2
        b = e_iphi ** 2 * (1 - a)  # np.sin(alpha/2)**2
        c = -1j * e_iphi * np.sin(alpha)

        R[0, 0] = a
        R[0, 1] = b
        R[0, 2] = c

        R[1, 0] = np.conjugate(b)
        R[1, 1] = a
        R[1, 2] = np.conjugate(c)

        R[2, 0] = -0.5 * np.conjugate(c)
        R[2, 1] = -0.5 * c
        R[2, 2] = np.cos(alpha)
        return R

    def Relax_matrix(self, t, ii):
        """
        ii = 0: Calculate relaxation matrix
        ii = 1: Calculate the derivative of relaxation matrix to T1
        ii = 2: Calculate the derivative of relaxation matrix to T2
        :param t: time the system is relaxing for
        :param ii: see explanation above
        :return: Relaxation matrix
        """
        Relax = np.zeros((3, 3))

        Relax[0, 0] = np.exp(-t / self.T2)
        Relax[1, 1] = np.exp(-t / self.T2)
        Relax[2, 2] = np.exp(-t / self.T1)

        if ii == 0:
            result = Relax

        if ii == 1:
            der_R_T1_term = np.zeros((3, 3))
            der_R_T1_term[2, 2] = t / self.T1 ** 2
            dRelax_dT1 = np.dot(Relax, der_R_T1_term)
            result = dRelax_dT1

        if ii == 2:
            der_R_T2_term = np.zeros((3, 3))
            der_R_T2_term[0, 0] = t / self.T2 ** 2
            der_R_T2_term[1, 1] = t / self.T2 ** 2
            dRelax_dT2 = np.dot(Relax, der_R_T2_term)
            result = dRelax_dT2

        return result

    def signal_step(self, S, tr, R, step):
        """
        Calculate next signal with EPG formalism.
        :param S: Current state matrix
        :param tr: Repetition time
        :param R: RF matrix
        :param step: Time step
        :return: updated state matrix
        """

        # First step
        if step == 0:
            # Matrix related to longitudinal relaxation
            Sz = np.zeros((3, S.shape[1]))
            Sz[2, 0] = 1
            S_new = np.linalg.multi_dot([R, S])

        if step != 0:
            # Matrix related to longitudinal relaxation
            if S.shape[1] < self.clip_state:
                Sz = np.zeros((3, S.shape[1] + 1))
                Sz[2, 0] = 1
            else:
                Sz = np.zeros((3, S.shape[1]))
                Sz[2, 0] = 1

            # Relaxation tr - fixed te and gradient influence
            Relax_term = np.dot(Cost.Relax_matrix(self, tr, 0), Cost.Grad(self, S)) + \
                         self.M0 * (1 - np.exp(-tr / self.T1)) * Sz
            # Rotation and echo relaxation
            S_new = np.linalg.multi_dot([R, Relax_term])
        return S_new

    def signal(self, alpha):
        """
        Create signal (not used for CRB cost function calculation but is used for UEE_phase)
        :param alpha: flip angle sequence
        :return: Transverse magnetisation signal
        """
        self.TR = self.TR_set
        signal = np.zeros(self.N, dtype=complex)

        # First step
        R = Cost.RF_matrix(self, alpha[0], self.phi[0])
        S = np.dot(R, self.S_init)
        signal[0] = S[0, 0]

        for step in range(1, self.N):
            R = Cost.RF_matrix(self, alpha[step], self.phi[step])
            S_new = Cost.signal_step(self, S=S, tr=self.TR_set[step - 1], R=R, step=step)
            signal[step] = S_new[0, 0]
            S = S_new

        return signal

    def dm_dT(self, alpha):
        """
        Find dm[n]/dT1 and dm[n]/dT2 using the State matrix.
        :param alpha: flip angle sequence
        :return: derivative of transverse magnetisation to the tissue parameters
        """
        try:
            self.TR
        except AttributeError:
            self.TR = self.TR_set

        der_m_T1 = np.zeros((2, self.N))
        der_m_T2 = np.zeros((2, self.N))
        der_m_M0 = np.zeros((2, self.N))

        der_S_T1_previous = np.zeros((3, 2))
        der_S_T2_previous = np.zeros((3, 2))
        der_S_M0_previous = np.zeros((3, 2))

        S = self.S_init

        for step in range(self.N):
            R = Cost.RF_matrix(self, alpha[step], self.phi[step])
            if step == 0:
                # Matrix related to longitudinal relaxation
                Sz = np.zeros((3, S.shape[1]))
                Sz[2, 0] = 1

                # Derivative to M0
                der_S_M0 = np.dot(R, Sz)
                der_m_M0[0, step] = np.real(der_S_M0[0, 0])
                der_m_M0[1, step] = np.imag(der_S_M0[0, 0])
                der_S_M0_previous = der_S_M0

                S = Cost.signal_step(self, S=S, tr=np.array([]), R=R, step=step)

            if step != 0:
                # Matrix related to longitudinal relaxation
                if S.shape[1] < self.clip_state:
                    Sz = np.zeros((3, S.shape[1] + 1))
                    Sz[2, 0] = 1
                else:
                    Sz = np.zeros((3, S.shape[1]))
                    Sz[2, 0] = 1

                # Derivative to T1
                der_S_T1 = np.linalg.multi_dot([R, Cost.Relax_matrix(self, self.TR[step - 1], 1), Cost.Grad(self, S)]) + \
                           -self.M0 * self.TR[step - 1] / self.T1 ** 2 * np.exp(-self.TR[step - 1] / self.T1) * np.dot(
                    R, Sz) + \
                           np.linalg.multi_dot(
                               [R, Cost.Relax_matrix(self, self.TR[step - 1], 0), Cost.Grad(self, der_S_T1_previous)])
                der_m_T1[0, step] = np.real(der_S_T1[0, 0])
                der_m_T1[1, step] = np.imag(der_S_T1[0, 0])
                der_S_T1_previous = der_S_T1

                # Derivative to T2
                der_S_T2 = np.linalg.multi_dot([R, Cost.Relax_matrix(self, self.TR[step - 1], 2), Cost.Grad(self, S)]) + \
                           np.linalg.multi_dot(
                               [R, Cost.Relax_matrix(self, self.TR[step - 1], 0), Cost.Grad(self, der_S_T2_previous)])
                der_m_T2[0, step] = np.real(der_S_T2[0, 0])
                der_m_T2[1, step] = np.imag(der_S_T2[0, 0])
                der_S_T2_previous = der_S_T2

                # Derivative to M0
                der_S_M0 = np.linalg.multi_dot(
                    [R, Cost.Relax_matrix(self, self.TR[step - 1], 0), Cost.Grad(self, der_S_M0_previous)]) + \
                           (1 - np.exp(-self.TR[step - 1] / self.T1)) * np.dot(R, Sz)
                der_m_M0[0, step] = np.real(der_S_M0[0, 0])
                der_m_M0[1, step] = np.imag(der_S_M0[0, 0])
                der_S_M0_previous = der_S_M0

                S = Cost.signal_step(self, S=S, tr=self.TR[step - 1], R=R, step=step)

        return der_m_T1, der_m_T2, der_m_M0

    def V(self, num_weights):
        """"
        Calculate inverse Fisher Information Matrix (FIM) I and its inverse V
        :param num_weights: The number of parameters in the FIM
        :return: The inverse FIM V
        """
        der_m_T1, der_m_T2, der_m_M0 = Cost.dm_dT(self, self.alpha)
        J_T_J_sum = np.zeros((num_weights, num_weights))

        for step in range(self.N):
            J = np.concatenate((der_m_T1[:, step].reshape(2, 1), der_m_T2[:, step].reshape(2, 1),
                                der_m_M0[:, step].reshape(2, 1)), axis=1)
            J_T = np.transpose(J)
            J_T_J_sum += np.dot(J_T, J)

        I = 1 / (self.sigma ** 2) * J_T_J_sum
        V = np.linalg.inv(I)

        return V

    def Opt(self, Opt_param, return_costs=False):
        """"
        Find cost function value based on the inverse FIM
        :param Opt_param: current acquisition parameters
        :return: Cost function value for CRB optimisation
        """

        if self.Opt_type == 'with_TR':
            self.alpha = np.squeeze(Opt_param[:self.N])
            self.TR = np.squeeze(Opt_param[self.N:])
        elif self.Opt_type == 'without_TR':
            self.alpha = np.squeeze(Opt_param)
            self.TR = self.TR_set
        if return_costs:
            costs = np.zeros((3, 3))
            costs2 = np.zeros((3))
        Opt_return = 0
        num_weights = len(self.kwargs)
        V_val = Cost.V(self, num_weights)

        W = np.zeros((num_weights, num_weights))
        if self.weighting == 'manual':
            step = 0
            for key, val in self.kwargs.items():
                W[step, step] = val
                step += 1
            Opt_val = np.trace(np.dot(W, V_val))
        elif self.weighting == 'rCRB':
            W[0, 0] = 1 / self.T1 ** 2
            W[1, 0] = 1 / (self.T1 * self.T2)
            W[2, 0] = 1 / (self.T1 * self.M0)
            W[0, 1] = 1 / (self.T2 * self.T1)
            W[1, 1] = 1 / self.T2 ** 2
            W[2, 1] = 1 / (self.T2 * self.M0)
            W[0, 2] = 1 / (self.M0 * self.T1)
            W[1, 2] = 1 / (self.M0 * self.T2)
            W[2, 2] = 1 / self.M0 ** 2
            Opt_val = np.trace(np.multiply(W, V_val))
        else:
            print('Not a valid weighting')
            exit()

        Opt_return += Opt_val

        if self.print:
            print('Opt_val is: ', Opt_val)
            if self.weighting == 'manual':
                print('np.dot(W, V_val): \n', np.dot(W, V_val))
            elif self.weighting == 'rCRB':
                print('np.multiply(W, V_val): \n', np.multiply(W, V_val))
            print('The inverse Fisher-information matrix V is: \n', V_val)

        if return_costs:
            return Opt_return, costs, costs2
        else:
            return (Opt_return)  # , V_val


class JAC():
    def __init__(self, Prob, Opt_param, base, step):
        self.Prob = Prob
        self.Opt_param = Opt_param
        self.base = base
        self.step = step

    def jac_step(self, ii):
        Opt_one_step_forward = self.Opt_param
        Opt_one_step_forward[ii] = self.Opt_param[ii] + self.step[ii]
        one_step_forward = self.Prob.Opt(Opt_one_step_forward)

        Jac_step = (one_step_forward - self.base) / self.step[ii]

        return Jac_step
