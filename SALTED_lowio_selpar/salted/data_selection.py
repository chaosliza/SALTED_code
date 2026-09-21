import os
import os.path as osp

import numpy as np
from ase.io import read

from salted.selection_utils import (
    DEFAULT_SELECTION_SEED,
    parse_model_setup,
    select_configurations,
)
from salted.sys_utils import ParseConfig


def build():
    inp = ParseConfig().parse_input()

    filename = inp.system.filename
    Ntrain = int(inp.gpr.Ntrain)
    trainsel = inp.gpr.trainsel
    saltedpath = inp.salted.saltedpath
    saltedname = inp.salted.saltedname

    model_setup = osp.join(os.getcwd(), "model_setup")
    groups = parse_model_setup(model_setup)
    frames = read(filename, ":", parallel=False)
    ndata = len(frames)

    expected = sum(len(indices) for indices in groups.values())
    if expected != ndata:
        raise ValueError(
            f"model_setup describes {expected} conformers but {filename!r} contains {ndata} frames."
        )
    if Ntrain > ndata:
        raise ValueError(
            f"More training structures requested (Ntrain={Ntrain}) than available ({ndata})."
        )

    rdir = osp.join(saltedpath, f"regrdir_{saltedname}")
    os.makedirs(rdir, exist_ok=True)

    candidates = list(range(ndata))

    print("=== Ntrain selection ===")
    print(f"Training-data selection mode: {trainsel}")

    selected = select_configurations(
        candidates,
        groups,
        Ntrain,
        trainsel,
        seed=DEFAULT_SELECTION_SEED,
        frames=frames if trainsel == "equal_rmsd_fps" else None,
        selection_label="Ntrain",
    )

    outfile = osp.join(rdir, f"training_set_N{Ntrain}.txt")
    np.savetxt(outfile, np.asarray(selected, dtype=int), fmt="%i")

    print(f"Full dataset size: {ndata}")
    print(f"Selected training configurations: {len(selected)}")
    print(f"Selection seed: {DEFAULT_SELECTION_SEED}")
    print(f"Wrote: {outfile}")

    if trainsel != "random":
        selected_set = set(selected)
        print("Per-structure selected counts:")
        for name, indices in groups.items():
            nsel = sum(i in selected_set for i in indices)
            print(f"  {name}: {nsel}/{len(indices)}")


if __name__ == "__main__":
    build()
