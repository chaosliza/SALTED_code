import os.path as osp

import h5py
import numpy as np

from salted.sys_utils import ParseConfig, read_system, do_fps


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
    train_indices = _load_training_indices(inp)

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        conf_indices=train_indices
    )

    M = inp.gpr.Menv
    sdir = osp.join(inp.salted.saltedpath, f"equirepr_{inp.salted.saltedname}")

    species_idx = {spe: ispe for ispe, spe in enumerate(species)}
    species_array = np.zeros((ndata, natmax), int)
    natoms_total = 0
    for ilocal, iconf in enumerate(train_indices):
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            species_array[ilocal, iat] = species_idx[spe]
            natoms_total += 1
    species_array = species_array.reshape(ndata * natmax)

    with h5py.File(osp.join(sdir, "FEAT-0.h5"), "r") as h5f:
        power = h5f["descriptor"][:]
        stored_indices = h5f["configuration_indices"][:].astype(int).tolist()
    if stored_indices != train_indices:
        raise ValueError("FEAT-0.h5 configuration order does not match training_set_N*.txt.")

    nfeat = power.shape[-1]
    power_dense = np.zeros((natoms_total, nfeat))
    dense_species = np.zeros(natoms_total, dtype=int)
    idx = 0
    for ilocal, iconf in enumerate(train_indices):
        n = natoms[iconf]
        power_dense[idx : idx + n] = power[ilocal, :n]
        dense_species[idx : idx + n] = species_array[ilocal * natmax : ilocal * natmax + n]
        idx += n

    if M > natoms_total:
        raise ValueError(
            f"Menv ({M}) cannot exceed the {natoms_total} atomic environments in the selected Ntrain set."
        )

    fps_idx = np.array(do_fps(power_dense, M, verbose=inp.salted.verbose), int)
    fps_species = dense_species[fps_idx]
    sparse_set = np.vstack((fps_idx, fps_species)).T
    print("Computed sparse set made of ", M, "environments")
    np.savetxt(osp.join(sdir, f"sparse_set_{M}.txt"), sparse_set, fmt="%i")


if __name__ == "__main__":
    build()
