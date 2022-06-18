import os
import platform
import numpy as np
import qutip
import QuTrit
from operators import hamiltonian
from scipy.linalg import expm, norm

def main():
    f = 3.77371
    rabi = 10
    #ph = 3.90242
    t = 0.429
    dt = t/100 #pulse duration

    #phase formula for simmetry of 01 after the second pulse
    y = f/rabi
    s = rabi * t
    a = np.sqrt(y**2+1)
    b = s*a/2
    ph = (a*np.cos(b) + 1j*y*np.sin(b)) / (-a*np.cos(b) + 1j*y*np.sin(b))
    ph = np.conj(ph)


    b = qutip.Bloch(view=[23, 0]) #11 1r+r1
    c = qutip.Bloch(view=[20, 0]) #01 0r
    print("ciao")
    psi_b = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    psi_c = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    psi_d = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    U_1 = _time_evolution(f ,t/2, 0)
    #b.add_states(psi_b[4] * qutip.basis(2,1))
    #c.add_states(psi_c[1] * qutip.basis(2,1))

    vec = ( (psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0)  + psi_b[4] * qutip.basis(2,1)).unit()
    b.add_states(vec, kind='point')
    vec = ( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit()
    c.add_states(vec, kind='point')

    U_1 = _time_evolution(f ,dt, 1)
    for ii in range(100):
        psi_b = np.matmul(U_1, psi_b)
        psi_c = np.matmul(U_1, psi_c)
        psi_d = np.matmul(U_1, psi_d)

        vec = ((psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0) + psi_b[4] * qutip.basis(2,1)).unit()
        b.add_states(vec, kind='point')

        vec = ( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit()
        c.add_states(vec, kind='point')

    b.add_states(((psi_b[5] + psi_b[7])/np.sqrt(2) * qutip.basis(2,0) + psi_b[4] * qutip.basis(2,1)).unit())
    c.add_states(( psi_c[2] * qutip.basis(2,0)  + psi_c[1] * qutip.basis(2,1)).unit())
    print("psi_b:")
    print(psi_b)
    print("psi_c:")
    print(psi_c)
    print("psi_d:")
    print(psi_d)

    #f = np.flip(f)
    U_2 = _time_evolution(f ,dt, ph)
    for ii in range(100):
        psi_b = np.matmul(U_2, psi_b)
        psi_c = np.matmul(U_2, psi_c)
        psi_d = np.matmul(U_2, psi_d)

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
    print("psi_d:")
    print(psi_d)
    #b.add_states((psi_f[2] * qutip.basis(2,0) + psi_f[1] * qutip.basis(2,1)).unit())
    b.render()
    b.save("b")
    c.render()
    c.save("c")
    #psi_f = np.matmul(1/np.sqrt(2)*_get_had_2(), psi_f)
    #print(norm(psi_f))
    #print(self.get_test(U_1, U_2))
    #infidelity = self._get_infidelity(self.psi_target, psi_f)
    # std = 1e-4
    #print(np.abs(infidelity))
    #self.FoM_list.append(np.abs(infidelity))
    #self.FoM_eval_number += 1
    #print(ph)
    #print(f)

def _time_evolution(fc, dt, ph):
    U = np.identity(9)
    ham_t = hamiltonian(fc, ph)
    U = np.matmul(expm(-1j * ham_t * dt), U)
    return U

main()
