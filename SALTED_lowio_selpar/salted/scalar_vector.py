import os
import os.path as osp

from ase.io import read
import h5py
import numpy as np

from salted.sys_utils import read_system, ParseConfig
from salted import sph_utils


def _load_training_indices(inp):
    path = osp.join(
        inp.salted.saltedpath,
        f"regrdir_{inp.salted.saltedname}",
        f"training_set_N{inp.gpr.Ntrain}.txt",
    )
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"Training-set file not found: {path}. Run 'python -m salted.data_selection' first."
        )
    return np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist()


def build():
    inp = ParseConfig().parse_input()
    (
        saltedname, saltedpath, saltedtype,
        filename, species, average,
        path2qm, qmcode, qmbasis, dfbasis,
        filename_pred, predname, predict_data, alpha_only,
        rep1, rcut1, sig1, nrad1, nang1, neighspe1,
        rep2, rcut2, sig2, nrad2, nang2, neighspe2,
        sparsify, nsamples, ncut,
        zeta, Menv, Ntrain, trainfrac, regul, eigcut,
        gradtol, restart, trainsel, nspe1, nspe2, HP1, HP2,
    ) = ParseConfig().get_all_params()

    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")
    if not sparsify:
        os.makedirs(sdir, exist_ok=True)

    train_indices = _load_training_indices(inp)
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        conf_indices=train_indices
    )

    if sparsify:
        vfps = {
            lam: np.load(osp.join(sdir, f"fps{ncut}-{lam}.npy"))
            for lam in range(lmax_max + 1)
        }

    all_frames = read(filename, ":", parallel=False)
    frames = [all_frames[i] for i in train_indices]
    natoms_total = sum(natoms[i] for i in train_indices)

    lam = 0
    llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)
    wigner3j = np.loadtxt(
        os.path.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat")
    )
    omega1 = sph_utils.get_representation_coeffs(
        frames, rep1, HP1, 0, neighspe1, species, nang1, nrad1, natoms_total
    )
    if sph_utils.reps_equivalent(rep1, neighspe1, HP1, rep2, neighspe2, HP2):
        omega2 = omega1
    else:
        omega2 = sph_utils.get_representation_coeffs(
            frames, rep2, HP2, 0, neighspe2, species, nang2, nrad2, natoms_total
        )

    v1 = np.transpose(omega1, (1, 3, 0, 2)).copy()
    v2 = np.transpose(omega2, (1, 3, 0, 2)).copy()
    c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

    if sparsify:
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        nfps = len(vfps[lam])
        p = sph_utils.equicombsparse_numba(
            natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
            v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize, nfps, vfps[lam],
        )
        featsize = ncut
    else:
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        p = sph_utils.equicomb_numba(
            natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
            v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize,
        )

    p = p.reshape(natoms_total, featsize)
    pvec = np.zeros((ndata, natmax, featsize))

    j = 0
    for ilocal, iconf in enumerate(train_indices):
        for iat in range(natoms[iconf]):
            pvec[ilocal, iat] = p[j]
            j += 1

    with h5py.File(osp.join(sdir, "FEAT-0.h5"), "w") as h5f:
        h5f.create_dataset("descriptor", data=pvec)
        h5f.create_dataset("configuration_indices", data=np.asarray(train_indices, dtype=int))


if __name__ == "__main__":
    build()
