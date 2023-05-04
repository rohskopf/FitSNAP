"""
Python script demonstrating how to load XYZ file and calculate custom descriptors.
Usage:
    python example.py --fitsnap_in input.in
"""

import numpy as np
from mpi4py import MPI
import argparse
import ase.io
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz
import itertools
import pickle
import torch
from fitsnap3lib.lib.neural_networks.descriptors.site_basis import *

def calc_fitting_data(atoms, pt):
    """
    Function to calculate fitting data from FitSNAP.
    Input: ASE atoms object for a single configuration of atoms.
    """

    # make a data dictionary for this config

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'Displaced_BCC'
    data['File'] = None
    data['Stress'] = None #atoms.get_stress(voigt=False)
    data['Positions'] = atoms.get_positions()
    data['Energy'] = None #atoms.get_total_energy()
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = None #atoms.get_forces()
    data['QMLattice'] = atoms.cell[:]
    data['test_bool'] = 0
    data['Lattice'] = atoms.cell[:]
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = 1.0
    data['fweight'] = 1.0
    data['vweight'] = 1.0

    # data must be a list of dictionaries

    data = [data]

    pt.create_shared_array('number_of_atoms', 1, tm=config.sections["SOLVER"].true_multinode)
    pt.shared_arrays["number_of_atoms"].array = np.array([len(atoms)])

    # Create arrays specific to custom calculator

    snap.pt.create_shared_array('number_of_neighs_scrape', len(data), dtype='i')
    snap.pt.slice_array('number_of_neighs_scrape')

    # preprocess configs

    snap.calculator.preprocess_allocate(len(data))
    for i, configuration in enumerate(data):
        snap.calculator.preprocess_configs(configuration, i)

    # calculate A matrix for the list of configs in data: 

    snap.data = data
    snap.calculator.shared_index=0
    snap.calculator.distributed_index=0 
    snap.process_configs()

    # return the A matrix for this config
    # we can also return other quantities (reference potential, etc.) associated with fitting

    # Create list of Configuration objects.
    
    snap.solver.create_datasets()
    
    # Save a pickled list of Configuration objects.

    configs_file = snap.config.sections['EXTRAS'].configs_file
    with open(configs_file, 'wb') as f:
        pickle.dump(snap.solver.configs, f)

# parse command line args

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
args = parser.parse_args()
print("FitSNAP input script:")
print(args.fitsnap_in)

comm = MPI.COMM_WORLD

# import parallel tools and create pt object

from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm=comm)

# config class reads the input settings

from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])

# create a fitsnap object

from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# tell ParallelTool not to create SharedArrays, optional depending on your usage of MPI during fits.
#pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
#pt.check_fitsnap_exist = False

# tell FitSNAP not to delete the data object after processing configs

snap.delete_data = False

# read configs and make a single ASE atoms object 

frames = ase.io.read("xyz-water/latte_cell_40.xyz", ":")

for atoms in frames:

    calc_fitting_data(atoms, pt)

# Load pickled list of configs.

with open(r"configs.pickle", "rb") as file:
    configs = pickle.load(file)

# Extract data needed for descriptor calculation.

x = torch.tensor(configs[0].x).requires_grad_(True) # NOTE: requires_grad_(True) turns on autograd w.r.t. this tensor.
transform_x = torch.tensor(configs[0].transform_x)
atom_types = torch.tensor(configs[0].types).long()
num_atoms = torch.tensor(configs[0].natoms)
neighlist = torch.tensor(configs[0].neighlist).long()
# Extract atoms i and neighbors j
# E.g. https://github.com/FitSNAP/FitSNAP/blob/master/fitsnap3lib/tools/dataloader/pairwise.py
unique_i = neighlist[:,0] # atoms i lined up with neighbors j.
unique_j = neighlist[:,1] # neighbors j lined up with atoms i.

# Calculate distances (rij) and displacements (xij, yij, zij).
# NOTE: Only do this once so we don't bloat the computational graph.
# See `forward` function in https://github.com/FitSNAP/FitSNAP/blob/master/fitsnap3lib/lib/neural_networks/pairwise.py

# loop through unique atom indices and build local list for it?
#print ('unique_i',unique_i)
#ui = torch.unique(unique_i)
#print ('ui',ui)


xneigh = transform_x + x[unique_j,:]
diff = x[unique_i] - xneigh # size (numneigh, 3)
diff_norm = torch.nn.functional.normalize(diff, dim=1) # size (numneigh, 3)
rij = torch.linalg.norm(diff, dim=1).unsqueeze(1)  # size (numneigh,1)
neigh_types = atom_types[unique_j]

ui = torch.unique(unique_i)
#print (neighlist[(unique_i == ui[0]).nonzero().squeeze()])

print ((unique_i == ui[0]).nonzero())
nediffs = diff[(unique_i == ui[0]).nonzero().squeeze()]
#nediffs = xneigh[(unique_i == ui[0]).nonzero().squeeze()]
#netypes = neigh_types[(unique_i == ui[0]).nonzero().squeeze()]
netypes = neigh_types[(unique_i == ui[0]).squeeze()]
itypes = atom_types[(ui).squeeze()]
print (nediffs)
print (netypes)
print ('neigh dists shape',nediffs.shape)
print (itypes)

print (neigh_types)
# Declare settings needed for ACE descriptors.
# From example:
# coupling_standalone/blob/FitSNAP_pairing/all_examples/FitSNAP_standard/selected_avg.py
ranks = [1,2,3,4] # ranks of basis functions to be evaluated
rc = 5.0 # from fitsnap input
lmbda = 2.0 # exponential factor for sampling of r << r_c in g_k(r)
nradbase = 2 # maximum k in g_k expansion of R_nl
lmax_dict =    {1:0,        2: 2, 3:2, 4:2}
nradmax_dict = {1:nradbase, 2: 2, 3:2, 4:2}
elements = ['H','O']

from get_ccs import *

#ccs_per_orb_nl
#orb_nl = '0_1,1,1,1,1,1_'
orb_nl = '0_0,2,0_'
# Make radial basis object for a particular nl:
mu0,mu,n,l,inter = get_mu_n_l(orb_nl,return_L=True)
rank = len(l)
# NOTE these functions expect distances for all neighbor atomsj to central atom i
#rb = RadialBasis(diff, rc, nradbase, nradmax_dict[rank], lmax_dict[rank], lmbda)
rb = RadialBasis(nediffs, rc, nradbase, nradmax_dict[rank], lmax_dict[rank], lmbda)
#rb.set_species(specs=neigh_types)
rb.set_species(specs=netypes,mu0=mu0,mui=itypes[ui[0]])
#ab = AngularBasis(diff,lmax_dict[rank])
ab = AngularBasis(nediffs,lmax_dict[rank])
Abas = A_basis(rb,ab)
ccsi = ccs_per_orb_nl[orb_nl]
Bbas = B_basis(mu,n,l,Abas,ccsi)
bi = Bbas.get_B()
print (Bbas.B)
print (Bbas.B.shape)
print ('bi',bi)
#rb = RadialBasis(rc, nradbase, nradmax_dict[rank], lmax_dict[rank], lmbda)

"""
TODO: Use this radial basis object to calculate descriptors given the geometry tensors above (rij, diff, diff_norm).
"""
