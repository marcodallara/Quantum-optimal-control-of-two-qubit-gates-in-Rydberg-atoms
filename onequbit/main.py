import os
import platform
import numpy as np
import qutip
from operators import hamiltonian
from scipy.linalg import expm, norm

def main():
    #f is detuning
    f = 3.77371
    f = 10

    t = 0.628 #2 pigreco
    t = 0.445/2
    dt = t/100 #pulse duration

    b = qutip.Bloch(view=[-90, 15]) #11 1r+r1
    b.point_marker = ['o']
    b.figsize = [7, 6]
    colors = []
    for i in range(100):
        colors.append('b')
    for i in range(100):
        colors.append('r')
    b.point_color = colors
    b.point_size = [15]
    b.font_size = 18
    b.zlpos = [1.2, -1.2]
    b.zlabel = ['$|0>$', '$|1>$']

    psi_0 = [1.0, 0.0]


    vec = ( psi_0[0] * qutip.basis(2,0)  + psi_0[1] * qutip.basis(2,1)).unit()
    #b.add_states(vec, kind='point')
    #b.add_states( psi_0[0] * qutip.basis(2,0)  + psi_0[1] * qutip.basis(2,1).unit())


    U_1 = _time_evolution(f ,dt, 1)
    for ii in range(100):
        psi_0 = np.matmul(U_1, psi_0)

        vec = ( psi_0[0] * qutip.basis(2,0)  + psi_0[1] * qutip.basis(2,1)).unit()
        b.add_states(vec, kind='point')


    #b.add_states( psi_0[0] * qutip.basis(2,0)  + psi_0[1] * qutip.basis(2,1).unit())
    print("tet")
    b.render()
    b.save("b")

def _time_evolution(fc, dt, ph):
    U = np.identity(2)
    ham_t = hamiltonian(fc, ph)
    U = np.matmul(expm(-1j * ham_t * dt), U)
    return U

main()
