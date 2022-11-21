from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


#config = Config()
#pt = ParallelTools()


class LammpsSnap(LammpsBase):

    def __init__(self, name):
        super().__init__(name)
        self.pt = ParallelTools()
        self.config = Config()
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

    def get_width(self):
        if (self.config.sections["SOLVER"].solver == "PYTORCH"):
            a_width = self.config.sections["BISPECTRUM"].ncoeff #+ 3
        else:
            num_types = self.config.sections["BISPECTRUM"].numtypes
            a_width = self.config.sections["BISPECTRUM"].ncoeff * num_types
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                a_width += num_types
        return a_width
    
    def _prepare_lammps(self):
        self._set_structure()
        # this is super clean when there is only one value per key, needs reworking
        #        self._set_variables(**_lammps_variables(config.sections["BISPECTRUM"].__dict__))

        # needs reworking when lammps will accept variable 2J
        self._lmp.command(f"variable twojmax equal {max(self.config.sections['BISPECTRUM'].twojmax)}")
        self._lmp.command(f"variable rcutfac equal {self.config.sections['BISPECTRUM'].rcutfac}")
        self._lmp.command(f"variable rfac0 equal {self.config.sections['BISPECTRUM'].rfac0}")
        #        self._lmp.command(f"variable rmin0 equal {config.sections['BISPECTRUM'].rmin0}")

        for i, j in enumerate(self.config.sections["BISPECTRUM"].wj):
            self._lmp.command(f"variable wj{i + 1} equal {j}")

        for i, j in enumerate(self.config.sections["BISPECTRUM"].radelem):
            self._lmp.command(f"variable radelem{i + 1} equal {j}")

        for line in self.config.sections["REFERENCE"].lmp_pairdecl:
            self._lmp.command(line.lower())

        self._set_computes()
        self._set_neighbor_list()

    def _set_box(self):
        self._set_box_helper(numtypes=self.config.sections['BISPECTRUM'].numtypes)

    def _create_atoms(self):
        self._create_atoms_helper(type_mapping=self.config.sections["BISPECTRUM"].type_mapping)

    def _set_computes(self):
        numtypes = self.config.sections['BISPECTRUM'].numtypes
        radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        kw_options = {
            k: self.config.sections["BISPECTRUM"].__dict__[v]
            for k, v in
            {
                "rmin0": "rmin0",
                "bzeroflag": "bzeroflag",
                "quadraticflag": "quadraticflag",
                "switchflag": "switchflag",
                "chem": "chemflag",
                "bnormflag": "bnormflag",
                "wselfallflag": "wselfallflag",
                "bikflag": "bikflag",
                "switchinnerflag": "switchinnerflag",
                "switchflag": "switchflag",
                "sinner": "sinner",
                "dinner": "dinner",
                "dgradflag": "dgradflag",
            }.items()
            if v in self.config.sections["BISPECTRUM"].__dict__
        }

        # remove input dictionary keywords if they are not used, to avoid version problems

        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        if kw_options["bikflag"] == 0:
            kw_options.pop("bikflag")
        if kw_options["switchinnerflag"] == 0:
            kw_options.pop("switchinnerflag")
        if kw_options["dgradflag"] == 0:
            kw_options.pop("dgradflag")
        kw_options["rmin0"] = self.config.sections["BISPECTRUM"].rmin0
        kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)

    def calculate_descriptors(self,x,neighlist,xneigh):
        """
        Calculator rij for the empirical potential solver.
        NOTE: Might add angles here for future empirical potential possibilities.
              Can also include custom-coded descriptors.
              Therefore nice to keep as a separate function for now.

        Args:
            x (ndarray): Array of positions with size (num_neigh,3), these are atoms i with repeated 
                indices to match the dimensions in `neighlist`. 
            neighlist (ndarray): Neighbor list of integers with size (num_neigh, 2); first column is 
                atom i and second column is atom j. 
            xneigh (ndarray): Array of positions for each atom, indicies correspond with `neighlist`, 
                with size (num_neigh, 3).

        Returns:
            Array of rij for each pair with size (num_neigh, ).
        """

        diff = x[neighlist[:,0]] - xneigh
        rij = np.linalg.norm(diff, axis=1) #[:,None]
        #rij = np.reshape(rij, (len(rij),-1)) # transform to (num_neigh,1)

        return rij

    def _collect_lammps_nonlinear(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]
        filename = self._data["File"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        #print(lmp_pos[0,0])

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # extract SNAP data, including reference potential data

        bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        ncols_bispectrum = n_coeff

        # number of columns in the snap array, add 3 to include indices and Cartesian components.

        ncols_snap = n_coeff + 3
        ncols_reference = 0
        nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1
        nrows_snap = nrows_energy + nrows_dgrad + 1
        assert nrows_snap == np.shape(lmp_snap)[0]
        index = self.shared_index # Index telling where to start in the shared arrays on this proc.
                                  # Currently this is an index for the 'a' array (natoms*nconfigs rows).
                                  # This is also an index for the 't' array of types (natoms*nconfigs rows).
                                  # Also made indices for:
                                  # - the 'b' array (3*natoms+1)*nconfigs rows.
                                  # - the 'dgrad' array (natoms+1)*nneigh*3*nconfigs rows.
                                  # - the 'dgrad_indices' array which has same number of rows as 'dgrad'
        dindex = self.distributed_index
        index_b = self.shared_index_b
        index_c = self.shared_index_c
        index_dgrad = self.shared_index_dgrad
        index_neighlist = self.shared_index_neighlist

        # extract the useful parts of the snap array

        bispectrum_components = lmp_snap[0:bik_rows, 3:n_coeff+3]
        #print(bispectrum_components)
        ref_forces = lmp_snap[0:bik_rows, 0:3].flatten()
        dgrad = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 3:n_coeff+3]
        dgrad_indices = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 0:3].astype(np.int32)
        ref_energy = lmp_snap[-1, 0]

        # strip zero dgrad components (equivalent to pruning neighborlist)
         
        nonzero_rows = lmp_snap[bik_rows:(bik_rows+nrows_dgrad),3:(n_coeff+3)] != 0.0
        nonzero_rows = np.any(nonzero_rows, axis=1)
        dgrad = dgrad[nonzero_rows, :]
        nrows_dgrad = np.shape(dgrad)[0]
        nrows_snap = np.shape(dgrad)[0] + nrows_energy + 1
        dgrad_indices = dgrad_indices[nonzero_rows, :]

        # populate the bispectrum array 'a'

        self.pt.shared_arrays['a'].array[index:index+bik_rows] = bispectrum_components
        self.pt.shared_arrays['t'].array[index:index+bik_rows] = lmp_types
        index += num_atoms

        # populate the truth array 'b' and weight array 'w'

        self.pt.shared_arrays['b'].array[index_b] = (energy - ref_energy)/num_atoms
        self.pt.shared_arrays['w'].array[index_b,0] = self._data["eweight"]
        self.pt.shared_arrays['w'].array[index_b,1] = self._data["fweight"]
        index_b += 1

        if (self.config.sections['CALCULATOR'].force):
            # populate the force truth array 'c'

            self.pt.shared_arrays['c'].array[index_c:(index_c + (3*num_atoms))] = self._data["Forces"].ravel() - ref_forces
            index_c += 3*num_atoms

            # populate the dgrad arrays 'dgrad' and 'dbdrindx'

            self.pt.shared_arrays['dgrad'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad
            self.pt.shared_arrays['dbdrindx'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad_indices
            index_dgrad += nrows_dgrad

        if (self.config.sections['CALCULATOR'].empiricalflag):
            ptr_pos = self._lmp.extract_atom('x')

            # look up the neighbor list

            nlidx = self._lmp.find_pair_neighlist('zero')
            nl = self._lmp.numpy.get_neighlist(nlidx)
            tags = self._lmp.extract_atom('id')
            number_of_neighs = 0 # number of neighs for this config
            num_neighs_per_atom = []
            neighlist = []
            xneighs = []
            transform_x = []
            for i in range(0,nl.size):
                idx, nlist  = nl.get(i)
                num_neighs_i = 0
                if nlist.size > 0:
                    for n in np.nditer(nlist):
                        num_neighs_i += 1
                        neighlist.append([tags[idx], tags[n]])
                        xneighs.append([ptr_pos[n][0], ptr_pos[n][1], ptr_pos[n][2]])
                        transform_x.append([ptr_pos[n][0]-lmp_pos[tags[n]-1,0], \
                                            ptr_pos[n][1]-lmp_pos[tags[n]-1,1], \
                                            ptr_pos[n][2]-lmp_pos[tags[n]-1,2]])

                num_neighs_per_atom.append(num_neighs_i)
                number_of_neighs += num_neighs_i

            num_neighs_per_atom = np.array(num_neighs_per_atom)
            assert(np.sum(num_neighs_per_atom) == number_of_neighs)
            neighlist = np.array(neighlist, dtype=int) - 1 # subtract 1 to get indices starting from 0
            xneighs = np.array(xneighs)
            transform_x = np.array(transform_x)

            rij = self.calculate_descriptors(self._data["Positions"], neighlist, xneighs)
            assert(np.shape(rij)[0] == number_of_neighs)

            # populate the neighlist arrays

            nrows_neighlist = number_of_neighs
            self.pt.shared_arrays['neighlist'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = neighlist
            self.pt.shared_arrays['rij'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = rij
            index_neighlist += nrows_neighlist

        # populate the fitsnap dicts
        # these are distributed lists and therefore have different size per proc, but will get 
        # gathered later onto the root proc in calculator.collect_distributed_lists
        # we use fitsnap dicts for NumAtoms and NumDgradRows here because they are organized differently 
        # than the corresponding shared arrays. 

        dindex = dindex+1
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])]
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])]
        self.pt.fitsnap_dict['NumAtoms'][self.distributed_index:dindex] = ['{}'.format(self._data['NumAtoms'])]
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])]
        
        if (self.config.sections['CALCULATOR'].force):
            self.pt.fitsnap_dict['NumDgradRows'][self.distributed_index:dindex] = ['{}'.format(nrows_dgrad)]

        if (self.config.sections['CALCULATOR'].empiricalflag):
            self.pt.fitsnap_dict['NumNeighs'][self.distributed_index:dindex] = ['{}'.format(nrows_neighlist)]

        # reset indices since we are stacking data in the shared arrays

        self.shared_index = index
        self.distributed_index = dindex
        self.shared_index_b = index_b
        self.shared_index_c = index_c
        self.shared_index_dgrad = index_dgrad
        self.shared_index_neighlist = index_neighlist

    def _collect_lammps(self):

        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_snap = nrows_energy + nrows_force + nrows_virial
        ncols_bispectrum = n_coeff * num_types
        ncols_reference = 1
        ncols_snap = ncols_bispectrum + ncols_reference
        # index = pt.fitsnap_dict['a_indices'][self._i]
        index = self.shared_index
        dindex = self.distributed_index

        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))

        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))
        irow = 0
        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if self.config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow:irow+bik_rows, :ncols_bispectrum] / num_atoms

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            EPS = 1.0e-10
            b000sum0 = 0.0
            nstride = n_coeff
            if not self.config.sections['BISPECTRUM'].bikflag:
                if not self.config.sections["BISPECTRUM"].bzeroflag:
                    b000sum0 = 1.0
                if self.config.sections["BISPECTRUM"].chemflag:
                    nstride //= num_types*num_types*num_types
                    if self.config.sections["BISPECTRUM"].wselfallflag:
                        b000sum0 *= num_types*num_types*num_types
                b000sum = sum(b_sum_temp[0, ::nstride])
                if abs(b000sum - b000sum0) < EPS:
                    print("WARNING: Configuration has no SNAP neighbors")

            if not self.config.sections["BISPECTRUM"].bzeroflag:
                if self.config.sections['BISPECTRUM'].bikflag:
                    raise NotImplementedError("per atom energy is not implemented without bzeroflag")
                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.zeros((num_types, 1))
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[self.config.sections["BISPECTRUM"].type_mapping[atom]-1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                b_sum_temp.shape = (num_types * n_coeff + num_types)

            self.pt.shared_arrays['a'].array[index:index+bik_rows] = \
                b_sum_temp * self.config.sections["BISPECTRUM"].blank2J[np.newaxis, :]
            ref_energy = lmp_snap[irow, icolref]
            self.pt.shared_arrays['b'].array[index:index+bik_rows] = 0.0
            self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
            # create an atom types list for the energy rows, if bikflag=1
            if self.config.sections['BISPECTRUM'].bikflag:
                types_energy = [int(i) for i in lmp_types]
                self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + bik_rows] = types_energy
            else:
                self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + bik_rows] = [0]

            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if self.config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
            self.pt.shared_arrays['a'].array[index:index+nrows_force] = \
                np.matmul(db_atom_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J))
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            self.pt.shared_arrays['b'].array[index:index+nrows_force] = \
                self._data["Forces"].ravel() - ref_forces
            self.pt.shared_arrays['w'].array[index:index+nrows_force] = \
                self._data["fweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            # create a types list for the force rows
            types_force = [] 
            for typ in lmp_types:
                for a in range(0,3):
                    types_force.append(typ)
            self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + nrows_force] = types_force
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if self.config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            self.pt.shared_arrays['a'].array[index:index+ndim_virial] = \
                np.matmul(vb_sum_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J))
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            self.pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            self.pt.shared_arrays['w'].array[index:index+ndim_virial] = \
                self._data["vweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            index += ndim_virial
            dindex += ndim_virial

        length = dindex - self.distributed_index
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
        self.shared_index = index
        self.distributed_index = dindex

    def _collect_lammps_preprocess(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # extract SNAP data, including reference potential data

        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)

        ncols_bispectrum = n_coeff + 3
        ncols_reference = 0
        nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1 #6
        dgrad = lmp_snap[num_atoms:(num_atoms+nrows_dgrad), 3:(n_coeff+3)]

        # strip zero dgrad components (almost equivalent to pruning neighborlist)
         
        nonzero_rows = lmp_snap[num_atoms:(num_atoms+nrows_dgrad),3:(n_coeff+3)] != 0.0
        nonzero_rows = np.any(nonzero_rows, axis=1)
        dgrad = dgrad[nonzero_rows, :]
        nrows_dgrad = np.shape(dgrad)[0]
        
        #self.dgradrows[self._i] = nrows_dgrad # no need to store this in a single proc, use a shared array instead

        # check that number of atoms here is equal to number of atoms in the sliced array

        natoms_sliced = self.pt.shared_arrays['number_of_atoms'].sliced_array[self._i]
        assert(natoms_sliced==num_atoms)
        self.pt.shared_arrays['number_of_dgrad_rows'].sliced_array[self._i] = nrows_dgrad

        # pre-processing for empirical potential

        if self.config.sections['CALCULATOR'].empiricalflag:

            # look up the neighbor list

            nlidx = self._lmp.find_pair_neighlist('zero')
            nl = self._lmp.numpy.get_neighlist(nlidx)
            tags = self._lmp.extract_atom('id')
            number_of_neighs = 0
            num_neighs_per_atom = []
            for i in range(0,nl.size):
                idx, nlist  = nl.get(i)
                num_neighs_i = 0
                if nlist.size > 0:
                    for n in np.nditer(nlist):
                        num_neighs_i += 1
                num_neighs_per_atom.append(num_neighs_i)
                number_of_neighs += num_neighs_i

            self.pt.shared_arrays['number_of_neighs_scrape'].sliced_array[self._i] = number_of_neighs