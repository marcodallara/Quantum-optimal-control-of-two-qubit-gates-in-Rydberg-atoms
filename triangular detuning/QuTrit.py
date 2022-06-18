# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy

from operators import hamiltonian, _get_had_2
import numpy as np
from scipy.linalg import expm, norm
from quocslib.utils.AbstractFoM import AbstractFoM
import os
import qutip

class QuTrit(AbstractFoM):

    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.psi_target = np.asarray(eval(args_dict.setdefault("target_state", "[0.0 , 1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2), 0.0, 0.0, 0.0, 0.0]")),
                                     dtype="complex")
        self.psi_0 = np.asarray(eval(args_dict.setdefault("initial_state", "[0.0 , 1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2), 0.0, 0.0, 0.0, 0.0]")), dtype="complex")

        # # Noise in the figure of merit
        # self.is_noisy = args_dict.setdefault("is_noisy", False)

        # # Drifting FoM
        # self.include_drift = args_dict.setdefault("include_drift", False)

        # Maximization or minimization
        # Minimization -1.0
        # Maximization 1.0
        self.optimization_factor = args_dict.setdefault("optimization_factor", -1.0)

        self.FoM_list = []
        self.save_path = ""
        self.FoM_save_name = "FoM.txt"

        self.FoM_eval_number = 0

    # def __del__(self):
    #     np.savetxt(os.path.join(self.save_path, self.FoM_save_name), self.FoM_list)

    def save_FoM(self):
        np.savetxt(os.path.join(self.save_path, self.FoM_save_name), self.FoM_list)

    def set_save_path(self, save_path: str = ""):
        self.save_path = save_path

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:
        #pulse duration
        t = parameters[2]
        N = 100
        dt = t/N
        #detuning with triangular shape, with linear a+bx
        b = parameters[0]
        a = parameters[1]
        #first half detuning
        f = [a]
        for i in range (N):
            f.append(a + b*(i+1)*dt)
        #rabifreq
        rabi = 10

        psi_01 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        psi_f = self.psi_0

        #first half evolution
        U_1 = self._time_evolution(f , dt, 1)
        psi_f = np.matmul(U_1, psi_f)
        psi_01 = np.matmul(U_1, psi_01)
        print("afer first")
        print(psi_f)
        #second half of triangle
        f = np.flip(f)

        U_2 = self._time_evolution(f , dt, 1)
        psi_f = np.matmul(U_2, psi_f)
        psi_01 = np.matmul(U_2, psi_01)

        phi_01 = numpy.angle(psi_01[1])
        print("afer 01")
        print(psi_01)
        print("afer second")
        print(psi_f)
        psi_targ = [0.0 , 1.0/np.sqrt(2), 0.0, 0.0, -np.exp(1j*phi_01)/np.sqrt(2), 0.0, 0.0, 0.0, 0.0]
        #print(norm(psi_f))
        #psi_f = np.matmul(1/np.sqrt(2)*_get_had_2(), psi_f)
        #print(norm(psi_f))
        #print(self.get_test(U_1, U_2))
        infidelity = self._get_infidelity(psi_targ, psi_f)
        # std = 1e-4
        #print(np.abs(infidelity))
        self.FoM_list.append(np.abs(infidelity))
        self.FoM_eval_number += 1

        return {"FoM": np.abs(infidelity)}#, "std": std}

    @staticmethod
    def get_test(u_1, u_2):
        psi_0 = np.array([1.0/2 , -1.0/2, 0.0, -1.0/2, 1.0/2, 0.0, 0.0, 0.0, 0.0])
        psi_f = np.matmul(u_1, psi_0)
        psi_f = np.matmul(u_2, psi_f)
        psi_f = np.matmul(_get_had_2(), psi_f)
        C = np.array(([1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]))
        psi_0 = np.matmul(C, psi_0)
        #psi_0 = np.matmul(1/np.sqrt(2)*_get_had_2(), psi_0)

        return psi_f,psi_0

    @staticmethod
    def _time_evolution(fc, dt, ph):
        U = np.identity(9)
        for ii in range(len(fc) - 1):
            ham_t = hamiltonian( (fc[ii + 1] + fc[ii]) / 2, ph)
            U_temp = U
            U = np.matmul(expm(-1j * ham_t * dt), U_temp)
        return U

    @staticmethod
    def _get_infidelity(psi1, psi2):
        return 1 - np.abs(np.dot(np.conj(psi1), psi2)) ** 2 / (norm(psi1) * norm(psi2))
