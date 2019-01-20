#! /usr/bin/env python

"""
Self-made atomic basis.
Zhihao Cui <zcui@caltech.edu>
"""

import os, sys
import copy
import numpy as np
import scipy.linalg as la

from pyscf import lo, tools
from pyscf.pbc import gto, scf, df
from pyscf.tools import molden

def atom_spin(name):
    """
    Given atom name, return spin for atomic calculation
    """
    # FIXME more atom choice
    if   (name == 'H' ): return 1
    elif (name == 'C' ): return 2
    elif (name == 'O' ): return 2
    elif (name == 'N' ): return 3
    elif (name == 'Si'): return 2
    else:
        raise ValueError("Undefined spin for atom: %s"%name)

def atom_core(name):
    """
    Given atom name, return number of core orbitals for atomic calculation
    """
    # FIXME PP treatment.
    if (name == 'H' ): return 0
    if (name == 'C' ): return 0
    if (name == 'O' ): return 0
    if (name == 'N' ): return 0
    if (name == 'Si'): return 0
    else:
        raise ValueError("Undefined core for atom: %s"%name)

def atom_norb(name, ref_basis='minao'):
    """
    Given atom name, return number of orb used for basis.
    """
    # FIXME PP treatment.
    assert(ref_basis in ['minao', 'gth-szv'])
    if ref_basis == 'gth-szv':
        pseudo = 'gth-pade'
    else:
        pseudo = None
    pmol          = gto.Cell()
    pmol.verbose  = 1
    pmol.atom     = [[name, (0.0, 0.0, 0.0)]]
    pmol.a        = np.eye(3) * 5.0
    pmol.basis    = ref_basis
    pmol.pseudo   = pseudo
    pmol.build()
    return pmol.nao_nr(), pmol.ao_labels()    

class Atom_Info(object):
    def __init__(self, mol, **kwargs):
        self.names, self.pos= zip(*mol._atom)
        self.names_uniq, self.idx_inv = np.unique(self.names, return_inverse=True)
        self.spins = [atom_spin(name) for name in self.names_uniq]
        self.cores = [atom_core(name) for name in self.names_uniq] # FIXME core info
        self.norbs = []
        self.ao_labels = []
        for name in self.names_uniq:
            if "ref_basis" in kwargs:
                ref_basis = kwargs["ref_basis"]
            else:
                ref_basis = 'minao'
            norbs, ao_labels = atom_norb(name, ref_basis=ref_basis)
            self.norbs.append(norbs)
            self.ao_labels.append(ao_labels)
        self.basis = mol._basis
        self.pseudo = mol._pseudo
        self.basis_partition = {}
        for i, (sh0, sh1, ao0, ao1) in enumerate(mol.offset_nr_by_atom()):
            name = self.names[i]
            if name not in self.basis_partition:
                self.basis_partition[name] = [range(ao0, ao1)]
            else:
                self.basis_partition[name].append(range(ao0, ao1))

def build_atom_cell(name, basis, pseudo, spin, charge=0, box_length=10.0):
    """
    Build cell (with vacuum) for atom calculation.
    """
    pmol          = gto.Cell()
    pmol.atom     = [[name, (0.0, 0.0, 0.0)]]
    pmol.a        = np.eye(3) * box_length
    pmol.basis    = basis
    pmol.charge   = charge
    pmol.pseudo   = pseudo
    pmol.spin     = spin
    pmol.build()
    return pmol

def atom_scf(mol, method='ROHF', **kwargs):
    """
    Atom calculation to make new basis.
    """
    info = Atom_Info(mol, **kwargs)
    for i, name in enumerate(info.names_uniq):
        basis = {name: info.basis[name]}
        pseudo = {name: info.pseudo[name]}
        spin = info.spins[i]
        pmol = build_atom_cell(name, basis, pseudo, spin)
        pmol.verbose = 1
        print "name: ", name
        print "spin: ", spin

        pmf = scf.KROHF(pmol)
        pmf.max_cycle  = 5000
        pmf.conv_tol   = 1e-6
        pmf = pmf.density_fit()
        pmf.scf()
        
        nocc = np.sum(pmf.mo_occ > 1e-12)
        assert(nocc <= info.norbs[i])

        print pmf.mo_occ
        print info.norbs[i]
        print pmf.mo_coeff

### --------- test functions ---------- ###

def build_diamond_cell():
    cell = gto.Cell()
    cell.a = '''
    3.5668  0       0
    0       3.5668  0
    0       0       3.5668'''
    cell.atom='''C 0.      0.      0.
       C 0.8917  0.8917  0.8917
       C 1.7834  1.7834  0.
       C 2.6751  2.6751  0.8917
       C 1.7834  0.      1.7834
       C 2.6751  0.8917  2.6751
       C 0.      1.7834  1.7834
       C 0.8917  2.6751  2.6751'''

    cell.ke_cutoff = 100
    #cell.basis = '6311g'
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.build(unit='Angstrom')
    return cell

def build_random_cell():
    cell = gto.Cell()
    cell.a = '''
    3.5668  0       0
    0       3.5668  0
    0       0       3.5668'''
    cell.atom='''C 0.      0.      0.
       H 0.8917  0.8917  0.8917
       H 1.7834  1.7834  0.
       O 2.6751  2.6751  0.8917
       C 1.7834  0.      1.7834
       C 2.6751  0.8917  2.6751
       H 0.      1.7834  1.7834
       Si 0.8917  2.6751  2.6751'''

    cell.ke_cutoff = 100
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 4
    cell.build(unit='Angstrom')
    return cell

if __name__ == '__main__':
    #cell = build_diamond_cell()
    cell = build_random_cell()
    #atom_info = Atom_Info(cell, ref_basis='gth-szv')
    atom_scf(cell, ref_basis='gth-szv')
