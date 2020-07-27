import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIModel
from tenpy.models.lattice import Triangular
from tenpy.algorithms import dmrg

import matplotlib.pyplot as plt


def plot_triangular(Lx, Ly):
    plt.figure(figsize=(4, 5))
    ax = plt.gca()
    lat = Triangular(Lx=Lx, Ly=Ly, site=None, bc_MPS='finite')
    lat.plot_coupling(ax, linewidth=3.)
    lat.plot_order(ax, linestyle=':')
    lat.plot_sites(ax)
    ax.set_aspect('equal')
    plt.show()


def DMRG_tf_ising_finite(Lx, Ly, g, verbose=True):
    print("finite DMRG, transverse field Ising model")
    print("Lx={Lx:d}, Ly={Ly:d}, g={g:.2f}".format(Lx=Lx, Ly=Ly, g=g))
    M = TFIModel(dict(g=g, J=1., lattice='Triangular', bc_MPS='finite', Lx=Lx, Ly=Ly, conserve=None, verbose=verbose))
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
        'combine': True
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    return E, psi, M


def onesite_DMRG_tf_ising_finite(Lx, Ly, g, verbose=True):
    print("single-site finite DMRG, transverse field Ising model")
    print("Lx={Lx:d}, Ly={Ly:d}, g={g:.2f}".format(Lx=Lx, Ly=Ly, g=g))
    M = TFIModel(dict(g=g, J=1., lattice='Triangular', bc_MPS='finite', Lx=Lx, Ly=Ly, conserve=None, verbose=verbose))
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'verbose': verbose,
        'combine': False,
        'active_sites': 1  # single-site
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    mag_x = np.sum(psi.expectation_value("Sigmax"))
    mag_z = np.sum(psi.expectation_value("Sigmaz"))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    return E, psi, M
    


if __name__ == "__main__":
    Lx=3
    Ly=3
    g=1.
    plot_triangular(Lx=Lx, Ly=Ly)
    DMRG_tf_ising_finite(Lx=Lx, Ly=Ly, g=g, verbose=True)
    print("-" * 100)
    onesite_DMRG_tf_ising_finite(Lx=Lx, Ly=Ly, g=g, verbose=True)