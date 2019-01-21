#!/usr/bin/env python
"""
Visualization code.
Written by Tianyu Zhu <tyzhu@caltech.edu>
"""
import sys
sys.path.append('../')
import iao # modified IAO
import numpy
from pyscf import lo, tools
from pyscf.pbc import gto, scf
from pyscf.tools import molden

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
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build(unit='Angstrom')
mf = scf.RHF(cell)
mf.kernel()

'''
generates IAOs
'''
mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
#a = lo.iao.iao(cell, mo_occ, minao='gth-szv')
a = iao.iao(cell, mo_occ, minao='gth-szv', scf_basis=True, ref_basis='gth-szv') # scf basis
print a.shape
molden.from_mo(cell, 'diamondiao_szv.molden', a)

# Orthogonalize IAO
a = lo.vec_lowdin(a, mf.get_ovlp())
molden.from_mo(cell, 'diamondiao_szv_ortho.molden', a)

loc_orb = lo.PM(cell, a).kernel()
molden.from_mo(cell, 'diamondiao_szv_PM.molden', loc_orb)

#ibo must take the orthonormalized IAOs
ibo = lo.ibo.ibo(cell, mo_occ, a)
print ibo.shape
molden.from_mo(cell, 'diamondibo_szv_ibo.molden', ibo)
