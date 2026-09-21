import csv
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from ase.build import minimize_rotation_and_translation


SELECTION_MODES = ("random", "equal_random", "equal_rmsd_fps")
DEFAULT_SELECTION_SEED = 3


def _positive_int_env(name):
    """Return a positive integer environment value, or None if unavailable."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(str(raw).split("(", 1)[0])
    except ValueError:
        return None
    return value if value > 0 else None


def rmsd_selection_worker_count(n_groups):
    """Choose RMSD-FPS workers from model groups and the SLURM allocation.

    There is no benefit in starting more workers than independent source
    structures.  Under SLURM, use the CPUs allocated on the current node (or
    CPUs per task for a conventional one-task job).  Outside SLURM, stay
    conservative and use one worker unless SALTED_SELECTION_WORKERS is set.
    """
    n_groups = int(n_groups)
    if n_groups <= 1:
        return 1

    override = _positive_int_env("SALTED_SELECTION_WORKERS")
    if override is not None:
        return max(1, min(n_groups, override))

    slurm_limits = [
        value
        for value in (
            _positive_int_env("SLURM_CPUS_PER_TASK"),
            _positive_int_env("SLURM_CPUS_ON_NODE"),
        )
        if value is not None
    ]
    if slurm_limits:
        allocated = max(slurm_limits)
        return max(1, min(n_groups, allocated))

    return 1


def _rmsd_fps_group(task):
    """Process one independent source structure for equal RMSD-FPS."""
    name, indices, quota, local_frames = task
    local_selected = rmsd_fps_indices(local_frames, quota)
    return name, [indices[i] for i in local_selected]


def training_set_path(inp):
    return os.path.join(
        inp.salted.saltedpath,
        f"regrdir_{inp.salted.saltedname}",
        f"training_set_N{inp.gpr.Ntrain}.txt",
    )


def load_training_indices(inp):
    path = training_set_path(inp)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training-set file not found: {path}. Run 'python -m salted.data_selection' first."
        )
    indices = np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist()
    if len(indices) != int(inp.gpr.Ntrain):
        raise ValueError(
            f"Training-set file contains {len(indices)} indices, expected Ntrain={inp.gpr.Ntrain}."
        )
    if len(indices) != len(set(indices)):
        raise ValueError(f"Training-set file contains duplicate configuration indices: {path}")
    return indices


def parse_model_setup(path="model_setup"):
    """Read ordered ``name:count`` entries from model_setup.

    The cumulative counts define the ORIGINAL/global XYZ frame ranges belonging
    to each source structure.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model_setup file: {path}")

    groups = OrderedDict()
    start = 0
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid model_setup line {lineno}: {raw.rstrip()!r}; expected 'name:count'."
                )
            name, count_text = line.split(":", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Empty structure name in model_setup line {lineno}.")
            try:
                count = int(count_text.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid conformer count in model_setup line {lineno}: {count_text!r}"
                ) from exc
            if count <= 0:
                raise ValueError(
                    f"Conformer count must be > 0 in model_setup line {lineno}, got {count}."
                )
            if name in groups:
                raise ValueError(f"Duplicate structure name {name!r} in model_setup.")

            groups[name] = list(range(start, start + count))
            start += count

    if not groups:
        raise ValueError("model_setup contains no structures.")
    return groups


def restrict_groups(groups, candidate_indices):
    candidate_set = {int(i) for i in candidate_indices}
    restricted = OrderedDict()
    for name, indices in groups.items():
        kept = [int(i) for i in indices if int(i) in candidate_set]
        if kept:
            restricted[name] = kept
    return restricted


def allocate_equal_quotas(groups, nselect, seed=DEFAULT_SELECTION_SEED):
    """Equal water-filling allocation subject to each structure's capacity."""
    capacities = OrderedDict((name, len(indices)) for name, indices in groups.items())
    total_capacity = sum(capacities.values())
    if nselect < 0:
        raise ValueError(f"nselect must be non-negative, got {nselect}.")
    if nselect > total_capacity:
        raise ValueError(
            f"Requested {nselect} configurations, but only {total_capacity} candidates are available."
        )

    quotas = OrderedDict((name, 0) for name in capacities)
    remaining = int(nselect)
    active = [name for name, cap in capacities.items() if cap > 0]

    while remaining > 0 and active:
        base = remaining // len(active)
        if base == 0:
            break

        saturated = [
            name for name in active
            if capacities[name] - quotas[name] < base
        ]

        if not saturated:
            for name in active:
                quotas[name] += base
            remaining -= base * len(active)
            break

        for name in saturated:
            available = capacities[name] - quotas[name]
            quotas[name] += available
            remaining -= available
        active = [name for name in active if name not in saturated]

    if remaining > 0:
        eligible = [name for name in active if quotas[name] < capacities[name]]
        if remaining > len(eligible):
            raise RuntimeError(
                "Internal quota-allocation error: remainder exceeds eligible structures."
            )
        rng = random.Random(seed)
        for name in rng.sample(eligible, remaining):
            quotas[name] += 1

    if sum(quotas.values()) != nselect:
        raise RuntimeError(
            f"Internal quota-allocation error: allocated {sum(quotas.values())}, expected {nselect}."
        )
    return quotas


def calc_rmsd(ref_mol, test):
    """RMSD definition used by the user's established conformer workflow.

    Coordinates are expected to have been aligned beforehand.
    """
    if isinstance(test, list):
        ref_pos = ref_mol.get_positions()
        test_pos = np.array([mol.get_positions() for mol in test])
        return np.sqrt(
            np.power(ref_pos - test_pos, 2).sum(axis=1).sum(axis=1) / ref_pos.shape[0]
        )

    return np.sqrt(
        np.sum((ref_mol.get_positions() - test.get_positions()) ** 2)
        / ref_mol.get_positions().shape[0]
    )


def calc_rmsd_matrix(molecules, only_heavy_atoms=False):
    """All-vs-all RMSD matrix matching the established conformer code."""
    if not molecules:
        return np.zeros((0, 0), dtype=float)

    symbols0 = molecules[0].get_chemical_symbols()
    nat0 = len(molecules[0])
    for mol in molecules:
        if len(mol) != nat0 or mol.get_chemical_symbols() != symbols0:
            raise ValueError(
                "RMSD-FPS requires identical atom counts and atom ordering within each source structure."
            )

    if only_heavy_atoms:
        from ase.filters import Filter

        include = [atom.index for atom in molecules[0] if atom.number != 1]
        molecules_tmp = [Filter(mol, indices=include) for mol in molecules]
    else:
        molecules_tmp = molecules

    n = len(molecules_tmp)
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        rmsd_matrix[i, i:] = calc_rmsd(molecules_tmp[i], molecules_tmp[i:])
        rmsd_matrix[i:, i] = rmsd_matrix[i, i:]
    return rmsd_matrix


def _aligned_copies(molecules):
    """Align copies to the first conformer, as in the established workflow."""
    aligned = deepcopy(molecules)
    if not aligned:
        return aligned
    reference = aligned[0]
    for mol in aligned:
        minimize_rotation_and_translation(reference, mol)
    return aligned


def rmsd_fps_indices(
    molecules, nselect, only_heavy_atoms=False
):
    """Fixed-size RMSD FPS using the user's established selection rule.

    First point: largest total RMSD to all conformers (equivalent to largest
    mean RMSD because every row has the same length). Subsequent points:
    maximize the minimum RMSD to the already-selected set.
    """
    nframes = len(molecules)
    if nselect < 0 or nselect > nframes:
        raise ValueError(f"Cannot select {nselect} RMSD-FPS points from {nframes} frames.")
    if nselect == 0:
        return []

    aligned = _aligned_copies(molecules)
    rmsd_matrix = calc_rmsd_matrix(aligned, only_heavy_atoms=only_heavy_atoms)

    first_scores = rmsd_matrix.sum(axis=0)
    first = int(first_scores.argmax())
    selected = [first]

    while len(selected) < nselect:
        min_to_selected = rmsd_matrix[selected, :].T.min(axis=1)
        min_to_selected[selected] = -np.inf
        next_id = int(min_to_selected.argmax())
        selected.append(next_id)

    return selected


def select_configurations(
    candidate_indices,
    groups,
    nselect,
    mode,
    seed=DEFAULT_SELECTION_SEED,
    frames=None,
    selection_label=None,
):
    """Select ORIGINAL/global configuration indices."""
    candidate_indices = [int(i) for i in candidate_indices]
    if len(candidate_indices) != len(set(candidate_indices)):
        raise ValueError("candidate_indices contains duplicates.")
    if nselect > len(candidate_indices):
        raise ValueError(
            f"Requested {nselect} configurations from only {len(candidate_indices)} candidates."
        )
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode {mode!r}; choose from {SELECTION_MODES}.")

    if mode == "random":
        selected = candidate_indices.copy()
        random.Random(seed).shuffle(selected)
        return selected[:nselect]

    restricted = restrict_groups(groups, candidate_indices)
    if sum(len(v) for v in restricted.values()) != len(candidate_indices):
        covered = {i for values in restricted.values() for i in values}
        missing = [i for i in candidate_indices if i not in covered]
        raise ValueError(
            f"{len(missing)} candidate configuration(s) are not represented by model_setup; "
            f"first missing indices: {missing[:10]}"
        )

    quotas = allocate_equal_quotas(restricted, nselect, seed=seed)
    selected = []

    if mode == "equal_random":
        rng = random.Random(seed)
        for name, indices in restricted.items():
            quota = quotas[name]
            if quota:
                selected.extend(rng.sample(indices, quota))

    elif mode == "equal_rmsd_fps":
        if frames is None:
            raise ValueError("frames must be supplied for equal_rmsd_fps selection.")

        tasks = []
        for name, indices in restricted.items():
            quota = quotas[name]
            if not quota:
                continue
            tasks.append((name, indices, quota, [frames[i] for i in indices]))

        n_workers = rmsd_selection_worker_count(len(tasks))
        if n_workers > 1:
            prefix = f"{selection_label} " if selection_label else ""
            print(
                f"{prefix}RMSD-FPS: processing {len(tasks)} source structures with "
                f"{n_workers} parallel workers.",
                flush=True,
            )
            # executor.map preserves task order, so concatenation remains
            # deterministic regardless of which worker finishes first.
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(_rmsd_fps_group, tasks)
                for _name, group_selected in results:
                    selected.extend(group_selected)
        else:
            prefix = f"{selection_label} " if selection_label else ""
            print(
                f"{prefix}RMSD-FPS: processing {len(tasks)} source structures sequentially.",
                flush=True,
            )
            for task in tasks:
                _name, group_selected = _rmsd_fps_group(task)
                selected.extend(group_selected)

    if len(selected) != nselect:
        raise RuntimeError(
            f"Internal selection error: selected {len(selected)}, expected {nselect}."
        )
    return selected
