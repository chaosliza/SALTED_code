import os.path as osp

import numpy as np
from scipy import sparse

from salted import sph_utils
from salted.sys_utils import ParseConfig, get_atom_idx, get_feats_projs, read_system


class arraylist:
    def __init__(self):
        self.data = np.zeros((100000,))
        self.capacity = 100000
        self.size = 0

    def update(self, row):
        n = row.shape[0]
        self.add(row, n)

    def add(self, x, n):
        if self.size + n >= self.capacity:
            self.capacity *= 2
            newdata = np.zeros((self.capacity,))
            newdata[: self.size] = self.data[: self.size]
            self.data = newdata

        self.data[self.size : self.size + n] = x
        self.size += n

    def finalize(self):
        return self.data[: self.size]


class PsiBuilder:
    """Per-structure RKHS feature vectors.

    Holds everything that does not depend on the structure, so the caller pays for
    it once instead of once per structure. `build` reproduces the body of the
    rkhs_vector loop exactly; both rkhs_vector and train_fused go through here so
    the two cannot drift apart.
    """

    def __init__(self, rank: int = 0, system=None, atom_info=None):
        inp = ParseConfig().parse_input()
        (self.saltedname, self.saltedpath, saltedtype,
         self.filename, species, average,
         path2qm, qmcode, qmbasis, dfbasis,
         filename_pred, predname, predict_data, alpha_only,
         self.rep1, rcut1, sig1, self.nrad1, self.nang1, self.neighspe1,
         self.rep2, rcut2, sig2, self.nrad2, self.nang2, self.neighspe2,
         self.sparsify, nsamples, self.ncut,
         zeta, Menv, Ntrain, trainfrac, regul, eigcut,
         gradtol, restart, trainsel,
         self.nspe1, self.nspe2, self.HP1, self.HP2) = ParseConfig().get_all_params()

        if saltedtype != "density":
            raise NotImplementedError(f"PsiBuilder supports saltedtype='density', got {saltedtype!r}")

        self.rank = rank
        self.zeta = zeta
        (self.species, self.lmax, self.nmax, self.lmax_max, nnmax, self.ndata,
         self.atomic_symbols, self.natoms, natmax) = system if system else read_system()
        self.atom_idx, self.natom_dict = atom_info if atom_info else get_atom_idx(
            self.ndata, self.natoms, self.species, self.atomic_symbols
        )

        if self.sparsify:
            self.vfps = {
                lam: np.load(osp.join(
                    self.saltedpath, f"equirepr_{self.saltedname}", f"fps{self.ncut}-{lam}.npy"
                ))
                for lam in range(self.lmax_max + 1)
            }

        self.Vmat, self.Mspe, self.power_env_sparse = get_feats_projs(self.species, self.lmax)

        self.cuml_Mcut = {}
        self.totsize = 0
        for spe in self.species:
            for lam in range(self.lmax[spe] + 1):
                for n in range(self.nmax[(spe, lam)]):
                    self.cuml_Mcut[(spe, lam, n)] = self.totsize
                    self.totsize += self.Vmat[(lam, spe)].shape[1]

        # For zeta == 1 get_feats_projs has already folded Vmat into
        # power_env_sparse and build() never reads it again, so holding it costs
        # 216 MB per rank for nothing. Every rank pays that, and minimize_loss
        # runs 45 of them on 186 GB nodes.
        if self.zeta == 1:
            self.Vmat = None

        # lam-dependent only: hoisted out of the per-structure loop, where the
        # wigner loadtxt was costing one shared-filesystem open per structure per lam.
        self.per_lam = {}
        for lam in range(self.lmax_max + 1):
            llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, self.nang1, self.nang2)
            wigner3j = np.loadtxt(osp.join(
                self.saltedpath, "wigners",
                f"wigner_lam-{lam}_lmax1-{self.nang1}_lmax2-{self.nang2}.dat",
            ))
            c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]
            self.per_lam[lam] = (llmax, llvec, wigner3j, c2r)

        self.reps_equivalent = sph_utils.reps_equivalent(
            self.rep1, self.neighspe1, self.HP1, self.rep2, self.neighspe2, self.HP2
        )

    def build(self, iconf: int, structure) -> sparse.coo_matrix:
        natoms = self.natoms[iconf]

        omega1 = sph_utils.get_representation_coeffs(
            structure, self.rep1, self.HP1, self.rank, self.neighspe1,
            self.species, self.nang1, self.nrad1, natoms)
        if self.reps_equivalent:
            omega2 = omega1
        else:
            omega2 = sph_utils.get_representation_coeffs(
                structure, self.rep2, self.HP2, self.rank, self.neighspe2,
                self.species, self.nang2, self.nrad2, natoms)

        v1 = np.transpose(omega1, (1, 3, 0, 2)).copy()
        v2 = np.transpose(omega2, (1, 3, 0, 2)).copy()

        power = {}
        for lam in range(self.lmax_max + 1):
            llmax, llvec, wigner3j, c2r = self.per_lam[lam]

            if self.sparsify:
                featsize = self.nspe1 * self.nspe2 * self.nrad1 * self.nrad2 * llmax
                nfps = len(self.vfps[lam])
                p = sph_utils.equicombsparse_numba(
                    natoms, self.nang1, self.nang2, self.nspe1 * self.nrad1,
                    self.nspe2 * self.nrad2, v1, v2, wigner3j, llmax, llvec, lam, c2r,
                    featsize, nfps, self.vfps[lam])
                featsize = self.ncut
            else:
                featsize = self.nspe1 * self.nspe2 * self.nrad1 * self.nrad2 * llmax
                p = sph_utils.equicomb_numba(
                    natoms, self.nang1, self.nang2, self.nspe1 * self.nrad1,
                    self.nspe2 * self.nrad2, v1, v2, wigner3j, llmax, llvec, lam, c2r,
                    featsize)

            if lam == 0:
                power[lam] = p.reshape(natoms, featsize)
            else:
                power[lam] = p.reshape(natoms, 2 * lam + 1, featsize)

        Psi = {}
        ispe = {}
        Tsize = 0
        for spe in self.species:
            ispe[spe] = 0
            nat_spe = self.natom_dict[(iconf, spe)]

            if self.zeta == 1:
                kernel0_nm = np.dot(
                    power[0][self.atom_idx[(iconf, spe)]], self.power_env_sparse[(0, spe)].T)
                Psi[(spe, 0)] = kernel0_nm
            else:
                kernel0_nm = np.dot(
                    power[0][self.atom_idx[(iconf, spe)]], self.power_env_sparse[(0, spe)].T)
                kernel_nm = kernel0_nm**self.zeta
                Psi[(spe, 0)] = np.real(np.dot(kernel_nm, self.Vmat[(0, spe)]))

            Tsize += nat_spe * self.nmax[(spe, 0)]

            for lam in range(1, self.lmax[spe] + 1):
                if self.zeta == 1:
                    Psi[(spe, lam)] = np.dot(
                        power[lam][self.atom_idx[(iconf, spe)]].reshape(
                            nat_spe * (2 * lam + 1), power[lam].shape[-1]),
                        self.power_env_sparse[(lam, spe)].T)
                else:
                    kernel_nm = np.dot(
                        power[lam][self.atom_idx[(iconf, spe)]].reshape(
                            nat_spe * (2 * lam + 1), power[lam].shape[-1]),
                        self.power_env_sparse[(lam, spe)].T)
                    kernel_nm_blocks = kernel_nm.reshape(
                        nat_spe, 2 * lam + 1, self.Mspe[spe], 2 * lam + 1)
                    kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (self.zeta - 1)
                    kernel_nm = kernel_nm_blocks.reshape(
                        nat_spe * (2 * lam + 1), self.Mspe[spe] * (2 * lam + 1))
                    Psi[(spe, lam)] = np.real(np.dot(kernel_nm, self.Vmat[(lam, spe)]))

                Tsize += nat_spe * self.nmax[(spe, lam)] * (2 * lam + 1)

        srows = arraylist()
        scols = arraylist()
        psi_nonzero = arraylist()

        i = 0
        for iat in range(natoms):
            spe = self.atomic_symbols[iconf][iat]
            for l in range(self.lmax[spe] + 1):
                i1 = ispe[spe] * (2 * l + 1)
                i2 = ispe[spe] * (2 * l + 1) + 2 * l + 1
                x = Psi[(spe, l)][i1:i2]
                nz = np.nonzero(x)
                vals = x[nz]
                for n in range(self.nmax[(spe, l)]):
                    psi_nonzero.update(vals)
                    srows.update(nz[0] + i)
                    scols.update(nz[1] + self.cuml_Mcut[(spe, l, n)])
                    i += 2 * l + 1
            ispe[spe] += 1

        ij = np.vstack((srows.finalize(), scols.finalize()))
        return sparse.coo_matrix(
            (psi_nonzero.finalize(), ij), shape=(Tsize, self.totsize))
