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
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from scipy.linalg import expm

# hamiltonian of a perfect Ryd Blockade
def hamiltonian( detuning_t, ph):
    phase = ph
    rabi_module = - 10
    # The controls
    ham_t = +detuning_t/2 * _get_gg() - detuning_t/2 * _get_ee() - rabi_module/2 * (np.conj(phase) * _get_eg() + phase * _get_ge())
    return ham_t

#e->excited, g->ground
def _get_eg():
    sigma_plus = np.array(np.outer([0, 1], [1, 0]), dtype="complex")
    return sigma_plus

def _get_ge():
    sigma_minus = np.array(np.outer([1, 0], [0, 1]), dtype="complex")
    return sigma_minus

def _get_ee():
    n_op = np.array(np.outer([0, 1], [0, 1]), dtype="complex")
    return n_op

def _get_gg():
    n_op = np.array(np.outer([1, 0], [1, 0]), dtype="complex")
    return n_op


def _time_evolution(fc, dt, ph):
    U = np.identity(4)
    ham_t = hamiltonian(fc, ph)
    U = np.matmul(expm(-1j * ham_t * dt), U)
    return U
