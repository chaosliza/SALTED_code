import os
import os.path as osp
import time

import numpy as np
from ase.io import read
from scipy import sparse

from salted import get_averages
from salted.psi_builder import PsiBuilder
from salted.selection_utils import load_training_indices
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    read_system,
)


def build():

    inp = ParseConfig().parse_input()
    (
        saltedname,
        saltedpath,
        saltedtype,
        filename,
        species,
        average,
        path2qm,
        qmcode,
        qmbasis,
        dfbasis,
        filename_pred,
        predname,
        predict_data,
        alpha_only,
        rep1,
        rcut1,
        sig1,
        nrad1,
        nang1,
        neighspe1,
        rep2,
        rcut2,
        sig2,
        nrad2,
        nang2,
        neighspe2,
        sparsify,
        nsamples,
        ncut,
        zeta,
        Menv,
        Ntrain,
        trainfrac,
        regul,
        eigcut,
        gradtol,
        restart,
        trainsel,
        nspe1,
        nspe2,
        HP1,
        HP2,
    ) = ParseConfig().get_all_params()

    comm, size, rank, parallel = detect_mpi()

    # gpr.fast_minimizer (default true): Jacobi-preconditioned CG with
    # buffer-based collectives. Same linear system, same gradtol, same
    # solution - different iterate path, so set it false to bit-reproduce a
    # model built before this existed.
    fast_minimizer = bool(inp.gpr.fast_minimizer)
    if parallel:
        from mpi4py import MPI

    fdir = f"rkhs-vectors_{saltedname}"
    rdir = f"regrdir_{saltedname}"

    # data_selection.py is the single authority for Ntrain selection.
    # The file stores ORIGINAL/global combined.xyz configuration indices.
    trainrangetot = load_training_indices(inp)
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = (
        read_system(conf_indices=trainrangetot)
    )

    atom_per_spe, natoms_per_spe = get_atom_idx(
        ndata, natoms, species, atomic_symbols, conf_indices=trainrangetot
    )

    # load average density coefficients if needed
    if average:
        # compute average density coefficients
        if rank == 0:
            get_averages.build()
        if parallel:
            comm.Barrier()
        # load average density coefficients
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(
                os.path.join(
                    saltedpath, "coefficients", "averages", f"averages_{spe}.npy"
                )
            )

    dirpath = os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}")
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    if parallel:
        comm.Barrier()

    # training_set_N*.txt was generated before initialize by data_selection.py.

    # Distribute structures to tasks
    ntraintot = int(trainfrac * Ntrain)

    if parallel:
        check_MPI_tasks_count(comm, ntraintot, "training structures")
        trainrange = distribute_jobs(comm, trainrangetot[:ntraintot])
        if inp.salted.verbose:
            print(f"Task {rank} handles the following structures: {format_index_ranges(trainrange,True)}", flush=True)
    else:
        trainrange = trainrangetot[:ntraintot]
    ntrain = int(len(trainrange))

    def loss_func(weights, ovlp_list, psi_list, coef_list):
        """Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function."""

        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        if saltedtype=="density":

            loss = 0.0
            # loop over training structures
            for iconf in range(ntrain):

                ref_coefs = coef_list[iconf]

                if average:
                    Av_coeffs = np.zeros(ref_coefs.shape[0])
                i = 0
                for iat in range(natoms[trainrange[iconf]]):
                    spe = atomic_symbols[trainrange[iconf]][iat]
                    for l in range(lmax[spe] + 1):
                        for n in range(nmax[(spe, l)]):
                            if average and l == 0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2 * l + 1

                # rebuild predicted coefficients
                pred_coefs = sparse.csr_matrix.dot(psi_list[iconf], weights)
                if average:
                    pred_coefs += Av_coeffs

                # compute predicted density projections
                ovlp = ovlp_list[iconf]
                ref_projs = np.dot(ovlp, ref_coefs)
                pred_projs = np.dot(ovlp, pred_coefs)

                # collect gradient contributions
                loss += sparse.csc_matrix.dot(
                    pred_coefs - ref_coefs, pred_projs - ref_projs
                )

        elif saltedtype=="density-response":

            loss = 0.0
            # loop over training structures
            itot = 0
            for iconf in range(ntrain):

                ovlp = ovlp_list[iconf]

                for icart in ["x","y","z"]:

                    ref_coefs = np.load(
                        osp.join(
                            saltedpath,
                            f"coefficients/{icart}/",
                            f"coefficients_conf{trainrange[iconf]}.npy",
                        )
                    )

                    # rebuild predicted coefficients
                    pred_coefs = sparse.csr_matrix.dot(psi_list[itot], weights)

                    # compute predicted density projections
                    ref_projs = np.dot(ovlp, ref_coefs)
                    pred_projs = np.dot(ovlp, pred_coefs)

                    # collect gradient contributions
                    loss += sparse.csc_matrix.dot(
                        pred_coefs - ref_coefs, pred_projs - ref_projs
                    )
                    itot += 1

        loss *= norm
        if parallel:
            loss = comm.allreduce(loss)

        # add regularization term
        loss += regul * np.dot(weights, weights)

        return loss

    def grad_func(weights, ovlp_list, psi_list, coef_list):
        """
        Given the weight-vector of the RKHS, compute the gradient of the electron-density loss function.
        """

        #        global totsize
        totsize = psi_list[0].shape[1]

        # init gradient
        gradient = np.zeros(totsize)

        if saltedtype=="density":

            # loop over training structures
            for iconf in range(ntrain):

                ref_coefs = coef_list[iconf]

                if average:
                    Av_coeffs = np.zeros(ref_coefs.shape[0])
                i = 0
                for iat in range(natoms[trainrange[iconf]]):
                    spe = atomic_symbols[trainrange[iconf]][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if average and l==0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2*l+1

                # rebuild predicted coefficients
                pred_coefs = sparse.csr_matrix.dot(psi_list[iconf],weights)
                if average:
                    pred_coefs += Av_coeffs

                # compute predicted density projections
                ovlp = ovlp_list[iconf]
                ref_projs = np.dot(ovlp,ref_coefs)
                pred_projs = np.dot(ovlp,pred_coefs)

                # collect gradient contributions
                gradient += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,pred_projs-ref_projs)
        
        elif saltedtype=="density-response":

            # loop over training structures
            itot = 0
            for iconf in range(ntrain):

                ovlp = ovlp_list[iconf]
 
                for icart in ["x","y","z"]:

                    # load reference QM data
                    ref_coefs = np.load(osp.join(
                        saltedpath, "coefficients", f"{icart}/coefficients_conf{trainrange[iconf]}.npy"
                    ))

                    # rebuild predicted coefficients
                    pred_coefs = sparse.csr_matrix.dot(psi_list[itot],weights)

                    # compute predicted density projections
                    ref_projs = np.dot(ovlp,ref_coefs)
                    pred_projs = np.dot(ovlp,pred_coefs)

                    # collect gradient contributions
                    gradient += 2.0 * sparse.csc_matrix.dot(psi_list[itot].T,pred_projs-ref_projs)
                    
                    itot += 1

        if parallel:
            if fast_minimizer:
                # Allreduce needs a contiguous float64 buffer; the sparse dots
                # above can hand back a matrix type. Normalise before reducing.
                gradient = np.ascontiguousarray(gradient, dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, gradient, op=MPI.SUM)
            else:
                gradient = comm.allreduce(gradient)
            gradient = gradient * norm + 2.0 * regul * weights
        else:
            gradient *= norm
            gradient += 2.0 * regul * weights
        return gradient

    PRECOND_BLK = 2048  # rows of psi^T per chunk; caps the dense temporary

    def precond_func(ovlp_list, psi_list):
        """Diagonal (Jacobi) preconditioner: diag of 2 * sum_conf psi^T S psi."""

        #        global totsize
        totsize = psi_list[0].shape[1]
        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):

            # psi_vector = psi_list[iconf].toarray()
            # ovlp_times_psi = np.dot(ovlp_list[iconf],psi_vector)
            # diag_hessian += 2.0*np.sum(np.multiply(ovlp_times_psi,psi_vector),axis=0)

            # BLOCKED over the weight dimension.
            #
            # The original line here was
            #     ovlp_times_psi = csc.dot(psi_list[iconf].T, ovlp_list[iconf])
            # which materialises a DENSE (totsize x n_aux) array - ~790 MB per
            # structure at n_aux~2000, on every rank. That OOM-killed the node
            # (job 393399, "Detected 3 oom_kill events") and is almost certainly
            # why this function was written but never called.
            #
            # Same arithmetic, same per-row summation order, but the dense
            # intermediate is only (BLK x n_aux) - a few MB.
            psiT = psi_list[iconf].T.tocsr()
            for beg in range(0, totsize, PRECOND_BLK):
                end = min(beg + PRECOND_BLK, totsize)
                blk = psiT[beg:end]
                if blk.nnz == 0:
                    continue
                tmp = blk.dot(ovlp_list[iconf])
                diag_hessian[beg:end] += 2.0 * np.asarray(
                    blk.multiply(tmp).sum(axis=1)
                ).ravel()

        # del psi_vector

        return diag_hessian

    def curv_func(cg_dire, ovlp_list, psi_list):
        """Compute curvature on the given CG-direction."""

        totsize = psi_list[0].shape[1]

        Ad = np.zeros((totsize))

        if saltedtype=="density":

            for iconf in range(ntrain):
                psi_x_dire = sparse.csr_matrix.dot(psi_list[iconf],cg_dire)
                Ad += 2.0 * sparse.csc_matrix.dot(psi_list[iconf].T,np.dot(ovlp_list[iconf],psi_x_dire))

        elif saltedtype=="density-response":

            itot = 0
            for iconf in range(ntrain):
                for icart in ["x","y","z"]:
                    psi_x_dire = sparse.csr_matrix.dot(psi_list[itot],cg_dire)
                    Ad += 2.0 * sparse.csc_matrix.dot(psi_list[itot].T,np.dot(ovlp_list[iconf],psi_x_dire))
                    itot += 1
        
        if parallel:
            if fast_minimizer:
                # Buffer-based Allreduce, not the pickle-based lowercase one.
                # This is the ONE collective per CG iteration - ~394 kB, ~4500
                # times - so pickling it every iteration is pure overhead.
                # Same arithmetic, but MPI may sum in a different order, hence
                # this sits behind the flag.
                Ad = np.ascontiguousarray(Ad, dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, Ad, op=MPI.SUM)
            else:
                Ad = comm.allreduce(Ad)
            Ad = Ad * norm + 2.0 * regul * cg_dire
        else:
            Ad *= norm
            Ad += 2.0 * regul * cg_dire

        return Ad

    psi_builder = None
    if inp.gpr.psi_in_memory:
        if saltedtype != "density":
            raise NotImplementedError(
                f"gpr.psi_in_memory requires saltedtype='density', got {saltedtype!r}"
            )
        psi_builder = PsiBuilder(
            rank,
            system=(
                species, lmax, nmax, llmax, nnmax, ndata,
                atomic_symbols, natoms, natmax,
            ),
            atom_info=(atom_per_spe, natoms_per_spe),
        )
        frames = read(filename, ":")

    if rank == 0:
        print("loading matrices...")
    ovlp_list = []
    psi_list = []
    coef_list = []
    for iconf in trainrange:
        ovlp_list.append(
            np.load(osp.join(saltedpath, "overlaps", f"overlap_conf{iconf}.npy"))
        )
        # load feature vector as a scipy sparse object
        if saltedtype=="density":
            # coo, matching what load_npz returns: csr would reorder the matvec sums
            psi_list.append(psi_builder.build(iconf, frames[iconf]) if psi_builder
                            else sparse.load_npz(osp.join(
              saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            )))
            coef_list.append(np.load(osp.join(
              saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"
            )))
        elif saltedtype=="density-response":
            for icart in ["x","y","z"]:
                psi_list.append(sparse.load_npz(osp.join(
                  saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                )))

    totsize = psi_list[0].shape[1]
    norm = 1.0 / float(ntraintot)

    if rank == 0:
        print(f"problem dimensionality: {totsize}")

    start = time.time()

    # preconditioner
    #
    # Stock SALTED defines precond_func() and then never calls it: P was
    # np.ones(totsize), i.e. plain unpreconditioned CG, which is why this takes
    # ~4500 iterations. Each iteration costs exactly one allreduce, so the
    # iteration count IS the scaling problem.
    #
    # precond_func() sums over range(ntrain), and ntrain is the LOCAL slice from
    # distribute_jobs - so its result is a rank-local partial sum. Using it
    # as-is would give every rank a DIFFERENT preconditioner while w is
    # replicated, which is inconsistent rather than merely slow. It needs the
    # allreduce below. That is a one-off collective, not a per-iteration one.
    if fast_minimizer:
        _tp = time.time()
        diag_hessian = precond_func(ovlp_list, psi_list)
        if parallel:
            diag_hessian = np.ascontiguousarray(diag_hessian, dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, diag_hessian, op=MPI.SUM)
        diag_hessian = diag_hessian * norm + 2.0 * regul
        # Guard the inversion: a zero or negative diagonal would poison the
        # search direction. Fall back to 1.0 for any such entry rather than
        # producing inf/nan and a silently wrong model.
        bad = ~(diag_hessian > 0.0)
        if bad.any() and rank == 0:
            print(f"WARNING: {int(bad.sum())} of {totsize} preconditioner "
                  f"diagonal entries were non-positive; using 1.0 for those.",
                  flush=True)
        P = np.where(bad, 1.0, 1.0 / np.where(bad, 1.0, diag_hessian))
        if rank == 0:
            spread = diag_hessian[~bad].max() / diag_hessian[~bad].min()
            print(f"Jacobi preconditioner active (gpr.fast_minimizer): "
                  f"built in {time.time()-_tp:.1f} s, diag range "
                  f"[{diag_hessian[~bad].min():.3e}, {diag_hessian[~bad].max():.3e}], "
                  f"spread {spread:.1f}x", flush=True)
    else:
        P = np.ones(totsize)

    reg_log10_intstr = str(int(np.log10(regul)))  # for consistency

    init = True
    if restart == True:
        wpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        dpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        rpath = osp.join(
            saltedpath,
            rdir,
            f"M{Menv}_zeta{zeta}",
            f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        if osp.exists(wpath) and osp.exists(dpath) and osp.exists(rpath):
            init = False
            w = np.load(wpath)
            d = np.load(dpath)
            r = np.load(rpath)
            s = np.multiply(P, r)
            delnew = np.dot(r, s)
            loss = loss_func(w, ovlp_list, psi_list, coef_list)
        else:
            # Print a warning and revert to the else behavior
            print(
                "Warning: One or more required files to restart do not exist. Reverting to default initialization."
            )

    if init:
        w = np.ones(totsize) * 1e-04
        loss = loss_func(w, ovlp_list, psi_list, coef_list)
        r = -grad_func(w, ovlp_list, psi_list, coef_list)
        d = np.multiply(P, r)
        delnew = np.dot(r, d)

    if rank == 0:
        print("minimizing...")
    for i in range(100000):
        #loss = loss_func(w, ovlp_list, psi_list)
        Ad = curv_func(d, ovlp_list, psi_list)
        curv = np.dot(d, Ad)
        alpha = delnew / curv
        w = w + alpha * d
        if (i + 1) % 50 == 0 and rank == 0:
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                w,
            )
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                d,
            )
            np.save(
                osp.join(
                    saltedpath,
                    rdir,
                    f"M{Menv}_zeta{zeta}",
                    f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                r,
            )
        if (i+1)%50==0:
            loss_old = loss.copy()
            loss = loss_func(w, ovlp_list, psi_list, coef_list)
            if loss>loss_old:
                if rank == 0:
                    print(f"WARNING: loss function increased, search direction reset as the steepest descent.")
                r = -grad_func(w, ovlp_list, psi_list, coef_list)
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}, loss: {loss:.3e}", flush=True)
                if np.linalg.norm(r) < gradtol:
                    break
                d = np.multiply(P, r)
                delnew = np.dot(r, d)
            else:
                r -= alpha * Ad
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}, loss: {loss:.3e}", flush=True)
                if np.linalg.norm(r) < gradtol:
                    break
                else:
                    s = np.multiply(P, r)
                    delold = delnew.copy()
                    delnew = np.dot(r, s)
                    beta = delnew / delold
                    d = s + beta * d
        else:
            r -= alpha * Ad
            if np.linalg.norm(r) < gradtol:
                if rank == 0:
                    print(f"step {i+1}, gradient norm: {np.linalg.norm(r):.3e}", flush=True)
                break
            else:
                s = np.multiply(P, r)
                delold = delnew.copy()
                delnew = np.dot(r, s)
                beta = delnew / delold
                d = s + beta * d

    if rank == 0:
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            w,
        )
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            d,
        )
        np.save(
            osp.join(
                saltedpath,
                rdir,
                f"M{Menv}_zeta{zeta}",
                f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            r,
        )
        print("minimization completed succesfully!")
        print(f"minimization time: {((time.time()-start)/60):.2f} minutes")


if __name__ == "__main__":
    build()
