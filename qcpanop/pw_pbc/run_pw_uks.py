
from qcpanop.pw_pbc.pseudopotential import get_nonlocal_pseudopotential_matrix_elements
from qcpanop.pw_pbc.basis import plane_wave_basis
from qcpanop.pw_pbc.scf import uks

from pyscf import dft, scf, pbc
from pyscf.pbc import gto, scf

import ase
from ase.build import bulk
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import numpy as np

def main():

    # define unit cell 
    
    #a = np.eye(3) * 4.0
    #atom = 'Mn 0 0 0'

    #ase_atom = bulk('Si', 'diamond', a = 10.26)
    ase_atom = bulk('C', 'diamond', a = 6.74)
    #ase_atom = bulk('He', 'diamond', a = 4.0)
    #ase_atom = bulk('H', 'diamond', a = 8.88)
    #ase_atom = bulk('Ne', 'diamond', a = 10.26)

    atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    a = ase_atom.cell 
    
    cell = gto.M(a = a,
                 atom = atom,
                 unit = 'bohr',
                 basis = 'cc-pvdz',
                 pseudo = 'gth-pbe',
                 #verbose = 100,
                 #ke_cutoff = 1000 / 27.21138602,
                 precision = 1.0e-8,
                 charge = 0,
                 spin = 0,
                 dimension = 3)

    cell.build()

    cutoff = 1000.0 

    # get plane wave basis information
    basis = plane_wave_basis(cell, 
        ke_cutoff = cutoff / 27.21138602, 
        n_kpts = [1, 1, 1])

    #mf = dft.UKS(cell) #, kpts=cell.make_kpts([1, 1, 1]))
    #mf.xc = 'lda'
    #energy = mf.kernel()
    #print(energy)

    # run plane wave scf 
    en, ca, cb = uks(cell, basis, xc = 'pbe', guess_mix = True)

    # C / diamond / pbe / gth-pbe / 1000 ev cutoff
    assert np.isclose(en, -10.281221451484)

    # hf / 300 ev cutoff
    #assert np.isclose(en, -10.453404153698)

if __name__ == "__main__":
    main()
