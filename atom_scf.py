#! /usr/bin/env python

"""
Self-made atomic basis.
Zhihao Cui <zcui@caltech.edu>
"""

import os, sys
import copy
from functools import reduce
import numpy as np
import scipy.linalg as la

from pyscf import lo, tools, lib
from pyscf.pbc import gto, scf, df
from pyscf.lib import logger as log
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
    # FIXME PP treatment, remove the warning
    assert(ref_basis in ['minao', 'gth-szv'])
    if ref_basis == 'gth-szv':
        pseudo = 'gth-pade'
    else:
        pseudo = None
    pmol          = gto.Cell()
    pmol.verbose  = 0
    pmol.atom     = [[name, (0.0, 0.0, 0.0)]]
    pmol.a        = np.eye(3) * 5.0
    pmol.basis    = ref_basis
    pmol.pseudo   = pseudo
    pmol.spin     = atom_spin(name)
    pmol.build()
    return pmol.nao_nr(), pmol.ao_labels()    

class Atom_Info(object):
    """
    Atomic information.
    mol             :   input mol object
    names           :   atom names
    pos             :   position of each atom
    names_uniq      :   unique atom names (species)
    idx_inv         :   idx to reconstruct names from names_uniq
    atom_num        :   number of atoms of each species
    spins           :   spin of each species
    cores           :   core of each species
    norbs           :   number of minimal orbitals of each species
    ao_labels       :   corresponding orbital labels
    basis           :   basis, dict, use atomic name as key
    pseufo          :   pseudo, dict, use atomic name as key
    basis_idx_old   :   basis idx in original mol/cell, dict, use atomic name as key
    basis_idx_new   :   basis idx in new basis, dict, use atomic name as key
    mfs             :   mf for atomic calculation
    """
    def __init__(self, mol, **kwargs):
        self.mol = mol
        self.names, self.pos= zip(*mol._atom)
        self.names_uniq, self.idx_inv, self.atom_num = \
                np.unique(self.names, return_inverse=True, return_counts=True)
        self.name_dict = dict(zip(self.names_uniq, range(len(self.names_uniq))))
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
        self.basis_idx_old = {}
        self.basis_idx_new = {}
        start = 0
        for i, (sh0, sh1, ao0, ao1) in enumerate(mol.offset_nr_by_atom()):
            name = self.names[i]
            idx = self.name2idx(name)
            norb = self.norbs[idx]
            if name not in self.basis_idx_old:
                self.basis_idx_old[name] = [range(ao0, ao1)]
            else:
                self.basis_idx_old[name].append(range(ao0, ao1))
            if name not in self.basis_idx_new:
                self.basis_idx_new[name] = [range(start, start+norb)]
            else:
                self.basis_idx_new[name].append(range(start, start+norb))
            start += norb
        self.mfs = [] # to be calculate

    def nao_nr(self):
        return np.sum([self.norbs[i] * self.atom_num[i] \
                for i, name in enumerate(self.names_uniq)])

    def name2idx(self, name):
        return self.name_dict[name]

def build_atom_cell(name, basis, pseudo, spin, charge=0, box_length=25.0):
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
    pmol.precision = 1e-12
    pmol.build()
    return pmol

def build_atom_cell_full_basis(info, order):
    mol = info.mol
    atom_new = [(name, pos) if i == order else ("Ghost:%s"%name, pos)\
            for i, (name, pos) in enumerate(mol._atom)]
    names_old = info.names
    names_new = zip(*atom_new)[0]
    atom_name = names_new[order]
    basis_new = {name : mol._basis.get(name, mol._basis[names_old[i]])
        for i, name in enumerate(names_new)}
    # FIXME: check if pseudo potential is correctly treated.
    pseudo_new = {atom_name : mol._pseudo[atom_name]}
    spin = atom_spin(atom_name)
    pmol          = gto.Cell()
    pmol.a        = mol.a
    pmol.atom     = atom_new
    pmol.basis    = basis_new
    pmol.charge   = 0
    pmol.pseudo   = pseudo_new
    pmol.spin     = spin
    pmol.precision = 1e-12
    pmol.build()
    return pmol

def scf_run(pmol, method='KROHF'):
    if method == 'KROHF':
        pmf = scf.KROHF(pmol)
    else:
        raise NotImplementedError
    pmf.max_cycle  = 5000
    pmf.conv_tol   = 1e-12
    pmf = pmf.density_fit()
    pmf.scf()
    return pmf

def atom_scf(mol, method='KROHF', full_basis=False, **kwargs):
    """
    Atom calculation to make new basis.
    """
    info = Atom_Info(mol, **kwargs)
    if not full_basis:
        for i, name in enumerate(info.names_uniq):
            basis = {name: info.basis[name]}
            pseudo = {name: info.pseudo[name]}
            spin = info.spins[i]
            log.debug(mol, "\nAtom calculation")
            log.debug(mol, 'atom = %s [%s / %s]', name, i+1, len(info.names_uniq))
            log.debug(mol, 'spin = %s', spin)
            log.debug(mol, 'basis = \n%s', basis)
            log.debug(mol, 'pseudo = \n%s', pseudo)
            
            pmol = build_atom_cell(name, basis, pseudo, spin)
            pmol.verbose = 3
            pmf = scf_run(pmol, method=method)
            info.mfs.append(pmf)
        
            nocc = np.sum(np.asarray(pmf.mo_occ) > 0.0)
            assert(pmf.converged)
            assert(nocc <= info.norbs[i])
            log.debug(mol, "mo_occ: %s", pmf.mo_occ)
            log.debug(mol, "nocc: %s", nocc)
            log.debug(mol, "norbs: %s", info.norbs[i])
            log.debug(mol, "ao label \n %s", info.ao_labels[i])
    else:
        for i, name in enumerate(info.names):
            log.debug(mol, "\nAtom calculation")
            log.debug(mol, 'atom = %s [%s / %s]', name, i+1, len(info.names))
            pmol = build_atom_cell_full_basis(info, i)
            pmol.verbose = 5
            pmf = scf_run(pmol, method=method)
            info.mfs.append(pmf)
        
            nocc = np.sum(np.asarray(pmf.mo_occ) > 0.0)
            assert(pmf.converged)
            assert(nocc <= info.norbs[i])
            log.debug(mol, "mo_occ: %s", pmf.mo_occ)
            log.debug(mol, "nocc: %s", nocc)
            log.debug(mol, "norbs: %s", info.norbs[i])
            log.debug(mol, "ao label \n %s", info.ao_labels[i])
    return info

def tile_ao_basis(info):
    """
    Tile new calculated minimal AO basis to form a basis expanded in 
    Cell's large basis.
    """
    assert(info.mfs)
    mol = info.mol
    mo_coeff_B2 = [mf.mo_coeff[0][:, :info.norbs[i]] \
            for i, mf in enumerate(info.mfs)]
    nbas_B2 = info.nao_nr()
    nbas_B1 = info.mol.nao_nr()
    B2 = np.zeros((nbas_B1, nbas_B2))
    for i, name in enumerate(info.names_uniq):
        bas_idx_old = info.basis_idx_old[name]
        bas_idx_new = info.basis_idx_new[name]
        assert(len(bas_idx_old) == len(bas_idx_new))
        for j in range(len(bas_idx_old)):
            B2[np.ix_(bas_idx_old[j], bas_idx_new[j])] = \
                    mo_coeff_B2[i]
    # normalize
    S1 = mol.pbc_intor('cint1e_ovlp_sph', kpt=mol.make_kpts([1,1,1]))
    val = np.diag(reduce(np.dot, (B2.T.conj(), S1, B2)))
    B2 /= np.sqrt(val)
    return B2

def get_B2(info):
    return tile_ao_basis(info)

def get_S1_S2_S12(B2, mol):
    S1 = mol.pbc_intor('cint1e_ovlp_sph', kpt=mol.make_kpts([1,1,1]))
    S2 = reduce(np.dot, (B2.T.conj(), S1, B2))
    S12 = S1.dot(B2)
    return S1, S2, S12

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
    cell.verbose = 5
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
    cell.verbose = 5
    cell.spin = 1
    cell.build(unit='Angstrom')
    return cell

def build_simple_cell():
    cell = gto.Cell()
    cell.a = '''
    3.5668  0       0
    0       4.5668  0
    0       0       3.5668'''
    cell.atom='''C 0.      0.      0.
       H 0.8917  0.8917  0.8917
       H 1.7834  1.7834  0. '''

    cell.ke_cutoff = 100
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 5
    cell.spin = 1
    cell.build(unit='Angstrom')
    return cell

if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000)
    cell = build_diamond_cell()
    #cell = build_random_cell()
    #cell = build_simple_cell()
    info = Atom_Info(cell, ref_basis='gth-szv')
    #info = atom_scf(cell, ref_basis='gth-szv')
    #order = 0
    #build_atom_cell_full_basis(info, order)
    info = atom_scf(cell, full_basis=True)
    exit()
    B2 = tile_ao_basis(info)
    s1, s2, s12 = get_S1_S2_S12(B2, cell)
