import os
import os.path as osp

import numpy as np

from salted.sys_utils import ParseConfig, read_system


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

    spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        conf_indices=train_indices
    )

    avcoefs = {spe: np.zeros(nmax[(spe, 0)], float) for spe in spelist}
    nat_per_species = {spe: 0 for spe in spelist}

    print(f"computing averages from selected Ntrain={len(train_indices)} configurations...")
    for iconf in train_indices:
        atoms = atomic_symbols[iconf]
        coefs = np.load(
            os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy")
        )
        i = 0
        for iat in range(natoms[iconf]):
            spe = atoms[iat]
            nat_per_species[spe] += 1
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    for _im in range(2 * l + 1):
                        if l == 0:
                            avcoefs[spe][n] += coefs[i]
                        i += 1

    adir = os.path.join(inp.salted.saltedpath, "coefficients", "averages")
    os.makedirs(adir, exist_ok=True)

    for spe in spelist:
        if nat_per_species[spe] == 0:
            raise ValueError(
                f"No atoms of species {spe!r} are present in the selected training set; cannot compute averages."
            )
        avcoefs[spe] /= nat_per_species[spe]
        np.save(os.path.join(adir, f"averages_{spe}.npy"), avcoefs[spe])


if __name__ == "__main__":
    build()
