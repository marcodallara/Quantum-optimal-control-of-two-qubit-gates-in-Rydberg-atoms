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

import os, platform
import matplotlib.pyplot as plt
import numpy as np
import math as math
from QuTrit import QuTrit
from operators import hamiltonian
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
import qutip
from scipy.linalg import expm, norm

def _time_evolution(fc, dt, ph):
    U = np.identity(9)
    ham_t = hamiltonian(fc, ph)
    U = np.matmul(expm(-1j * ham_t * dt), U)
    return U


def plot_FoM(result_path, FoM_filename):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    file_path = os.path.join(result_path, FoM_filename)
    save_name = "FoM_" + opt_name

    FoM = [line.rstrip('\n') for line in open(file_path)]
    FoM = [float(f) for f in FoM]
    iterations = range(1, len(FoM) + 1)
    # print('\nInitial FoM: %.4f' % FoM[0])
    # print('Final FoM: %.4f \n' % FoM[-1])
    min_FoM = min(FoM)
    max_FoM = max(FoM)
    difference = abs(max_FoM - min_FoM)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(iterations, FoM, color='darkblue', linewidth=1.5, zorder=10)
    # plt.scatter(x, y, color='k', s=15)

    plt.grid(True, which="both")
    plt.ylim(min_FoM - 0.05 * difference, max_FoM + 0.05 * difference)
    # plt.yscale('log')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('FoM', fontsize=20)
    # plt.savefig(os.path.join(folder, save_name + '.pdf'))
    plt.savefig(os.path.join(result_path, save_name + '.png'))


def plot_controls(result_path):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    for file in os.listdir(result_path):
        if file.endswith('best_controls.npz'):
            file_path = os.path.join(result_path, file)

    save_name = "Controls_" + opt_name

    controls = np.load(file_path)

    time_grid = []
    pulse = []

    for data_name in controls.files:
        if "time" in data_name:
            time_grid = controls[data_name]
        elif "pulse" in data_name:
            pulse = np.append(pulse, controls[data_name])

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(time_grid, pulse[:101], linewidth=1.5, zorder=10)
    plt.grid(True, which="both")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Control', fontsize=20)
    # plt.savefig(os.path.join(folder, save_name + '.pdf'))
    plt.savefig(os.path.join(result_path, save_name + '.png'))


def main(optimization_dictionary: dict, args_dict: dict):

    # Create FoM object
    FoM_object = QuTrit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    # Execute the optimization and save the FoM
    FoM_object.set_save_path(optimization_obj.results_path)
    optimization_obj.execute()
    FoM_object.save_FoM()

    # Plot the results
    plot_FoM(FoM_object.save_path, FoM_object.FoM_save_name)
    plot_controls(FoM_object.save_path)
    opt_alg_obj = optimization_obj.get_optimization_algorithm()

    best_parameters = opt_alg_obj.get_best_controls()["parameters"]
    det = best_parameters[0]
    print(det)
    t = best_parameters[1]
    dt = t/100 #pulse duration
    rabi = 10
    #phase formula for simmetry of 01 after the second pulse
    y = det/rabi
    s = rabi * t
    a = np.sqrt(y**2+1)
    b = s*a/2
    ph = (a*np.cos(b) + 1j*y*np.sin(b)) / (-a*np.cos(b) + 1j*y*np.sin(b))
    ph = np.conj(ph)

    U_1 = _time_evolution(det ,dt, 1)
    U_2 = _time_evolution(det ,dt, ph)
    b = qutip.Bloch(view=[23, 0]) #11 1r+r1
    c = qutip.Bloch(view=[20, 0]) #01 0r
    psi_b = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    psi_c = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for ii in range(100):
        psi_b = np.matmul(U_1, psi_b)
        psi_c = np.matmul(U_1, psi_c)

        vec = ((psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0) + psi_b[4] * qutip.basis(2,1)).unit()
        b.add_states(vec, kind='point')

        vec = ( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit()
        c.add_states(vec, kind='point')

    b.add_states(((psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0) + psi_b[4] * qutip.basis(2,1)).unit())
    c.add_states(( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit())
    for ii in range(100):
        psi_b = np.matmul(U_2, psi_b)
        psi_c = np.matmul(U_2, psi_c)

        vec = ( (psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0)  + psi_b[4] * qutip.basis(2,1)).unit()
        b.add_states(vec, kind='point')

        vec = ( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit()
        c.add_states(vec, kind='point')

    b.add_states(psi_b[4] * qutip.basis(2,1))
    c.add_states(psi_c[1] * qutip.basis(2,1))
    #f = np.flip(f)
    print("psi_b:")
    print(psi_b)
    print("psi_c:")
    print(psi_c)
    #b.add_states((psi_f[2] * qutip.basis(2,0) + psi_f[1] * qutip.basis(2,1)).unit())
    b.render()
    b.save("b")
    c.render()
    c.save("c")

# --------------------------------------------------------------------

if __name__ == '__main__':
    # get the optimization settings from the json dictionary
    optimization_dictionary = readjson(os.path.join(os.getcwd(), "dCRAB_QuTrit.json"))
    # define some parameters for the optimization
    args_dict = {"initial_state": "[0.0 , 1.0/np.sqrt(2), 0.0, 0.0, 1.0/np.sqrt(2), 0.0, 0.0, 0.0, 0.0]",
                 "target_state": "[0.0 , 1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2), 0.0, 0.0, 0.0, 0.0]"}
    main(optimization_dictionary, args_dict)

   # plot_controls(os.path.join(os.getcwd(), "QuOCS_Results/20220504_124607_Optimization_dCRAB_Fourier_NM_QuTrit/"))


