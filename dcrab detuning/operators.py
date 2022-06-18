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
import numpy as np

# hamiltonian of a perfect Ryd Blockade
def hamiltonian( detuning_t, ph):
    phase = ph
    rabi_module = - 10
    V_R = -211
    sigma_plus_1 = np.kron(_get_sigma_plus(), np.identity(3))
    sigma_minus_1 = np.kron(_get_sigma_minus(), np.identity(3))
    sigma_plus_2 = np.kron(np.identity(3), _get_sigma_plus())
    sigma_minus_2 = np.kron(np.identity(3), _get_sigma_minus())
    n_op_1 = np.kron(_get_n_op(), np.identity(3))
    n_op_2 = np.kron(np.identity(3), _get_n_op())
    h_int = V_R * np.array(np.outer([0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 1]), dtype="complex")
    # The controls
    ham_t = -rabi_module/2 * (np.conj(phase) * (sigma_plus_1 + sigma_plus_2) + phase*(sigma_minus_1 + sigma_minus_2)) - detuning_t * (n_op_1 + n_op_2) -h_int
    return ham_t

def _get_sigma_plus():
    sigma_plus = np.array(np.outer([0, 0, 1], [0, 1, 0]), dtype="complex")
    return sigma_plus


def _get_sigma_minus():
    sigma_minus = np.array(np.outer([0, 1, 0], [0, 0, 1]), dtype="complex")
    return sigma_minus


def _get_n_op():
    n_op = np.array(np.outer([0, 0, 1], [0, 0, 1]), dtype="complex")
    return n_op


def _get_sigma_x():
    sigma_x = 1/np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="complex")
    return sigma_x


def _get_sigma_y():
    sigma_y = 1/np.sqrt(2)*np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype="complex")
    return sigma_y


def _get_sigma_z():
    sigma_z = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]], dtype="complex")
    return sigma_z

def _get_had():
    had = np.array([[1, 1, 0],[1, -1, 0], [0, 0, 1]], dtype="complex")
    return had

def _get_had_2():
    had_2 = np.array(np.kron(np.identity(3), _get_had()), dtype="complex")
    return had_2
