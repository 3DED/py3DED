#!/usr/bin/env python
"""Old vs new Bloch-wave metric treatment, in both solution paths.

abTEM's Bloch waves map the symmetrized eigenproblem back to beam amplitudes
with a metric M = diag(1 / sqrt(1 + g_z / K)). Before the fix (abTEM branch
claude/bloch-metric-consistency) this was applied inconsistently: M on the
structure-matrix diagonal instead of M**2, the eigenvector matrix's *diagonal
elements* divided by M instead of its rows multiplied by M, and M applied even
for use_wave_eq=True/"exact", whose equations (the ones multislice solves)
have no metric at all. After the fix:

    use_wave_eq=True / "exact"  M = 1       conserves sum |psi_g|^2 = 1
    use_wave_eq=False           M as above  conserves sum (1 + g_z/K) |psi_g|^2 = 1

This script needs the NEW abTEM installed (it reproduces the old code paths
itself, verbatim), and compares, for a periodic Si crystal tilted off [001]
(beam || [0 1 k], --tilt-index k; k=8 is 7.1 deg), for each use_wave_eq:

  - old vs new, eigendecomposition path (what calculate_diffraction_patterns
    and therefore py3DED use)
  - old vs new, matrix-exponential path (calculate_scattering_matrix)
  - eigendecomposition vs matrix exponential, old and new (must agree)
  - flux conservation, old and new
  - optionally (--ms) R factors against multislice on the same periodic,
    tilted crystal: order=1 for use_wave_eq=True/False, "exact" for "exact"

Usage (CPU; add --device gpu on europa):
    python scripts/compare_bloch_metric.py
    python scripts/compare_bloch_metric.py --energies 100e3 300e3 --ms
    python scripts/compare_bloch_metric.py --tilt-index 3 --thickness 1000 --ms --device gpu

Intensities and R factors exclude 000; R = sum |I_a - I_b| / sum I_b (%).
"""
import argparse
import time

import numpy as np

import abtem
import abtem.bloch.dynamical as dyn
from abtem.bloch import BlochWaves, StructureFactor
from abtem.bloch.utils import (
    calculate_g_vec,
    excitation_errors,
    retrieve_structure_factor_values,
)
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.constants import kappa
from abtem.core.energy import energy2sigma, energy2wavelength

if not hasattr(dyn, "_metric"):
    raise SystemExit(
        "This abTEM has no Bloch-wave metric fix (abtem.bloch.dynamical._metric); "
        "install the fixed branch first."
    )


# --------------------------------------------------------------------------
# The OLD code paths, verbatim from before the fix
# --------------------------------------------------------------------------


def old_structure_matrix(structure_factor, hkl, hkl_selected, cell, energy, gpts, use_wave_eq):
    xp = get_array_module(structure_factor)
    g = xp.asarray(calculate_g_vec(hkl_selected, cell))
    Mii = dyn.calculate_M_matrix(hkl_selected, cell, energy)
    hkl_selected = np.asarray(hkl_selected)
    gmh = (hkl_selected[None] - hkl_selected[:, None]).reshape(-1, 3)
    A = retrieve_structure_factor_values(structure_factor, hkl, gmh, gpts)
    A = A.reshape((len(hkl_selected),) * 2)
    prefactor = energy2sigma(energy) / (kappa * energy2wavelength(energy) * np.pi)
    Mii = xp.asarray(Mii)
    A = A * prefactor * Mii[None] * Mii[:, None]
    sg = xp.asarray(excitation_errors(g, energy, use_wave_eq=use_wave_eq))
    diag = 2 * 1 / energy2wavelength(energy) * sg
    diag *= Mii
    xp.fill_diagonal(A, diag)
    return A


def old_dynamical_scattering(structure_matrix, hkl, cell, energy, thicknesses):
    xp = get_array_module(structure_matrix)
    thicknesses = np.asarray(thicknesses)
    Mii = xp.asarray(dyn.calculate_M_matrix(hkl, cell, energy))
    v, C = xp.linalg.eigh(structure_matrix)
    gamma = v * energy2wavelength(energy) / 2.0
    C = C.copy()
    idx = xp.arange(C.shape[0])
    C[idx, idx] = C[idx, idx] / Mii  # np.fill_diagonal(C, np.diag(C) / Mii)
    C_inv = xp.conjugate(C.T)
    initial = dyn.plane_wave_coefficients(hkl, xp)
    alpha = C_inv @ initial
    array = xp.zeros(shape=(len(thicknesses), len(hkl)), dtype=complex)
    for i, thickness in enumerate(thicknesses):
        array[i] = C @ (xp.exp(2.0j * xp.pi * thickness * gamma) * alpha)
    return array


def old_scattering_matrix(A, hkl, cell, z, energy):
    xp = get_array_module(A)
    S = dyn.expm(1.0j * xp.pi * z * A * energy2wavelength(energy))
    Mii = dyn.calculate_M_matrix(hkl, cell, energy)
    M = xp.asarray(np.diag(Mii))
    M_inv = xp.asarray(np.diag(1 / Mii))
    return xp.dot(M, xp.dot(S, M_inv))


# --------------------------------------------------------------------------


def tilted_setup(k):
    """Cubic Si, and the rotation R (R @ v_crystal = v_lab) putting [0 1 k] on
    the beam axis z; also the orthogonal periodic supercell along it."""
    from ase.build import bulk, make_supercell

    uc = bulk("Si", cubic=True)
    P = np.array([[1, 0, 0], [0, k, -1], [0, 1, k]])
    R = P / np.linalg.norm(P, axis=1)[:, None]
    sc = make_supercell(uc, P)
    sc.set_cell(np.diag(np.linalg.norm(sc.cell, axis=1)), scale_atoms=True)
    return uc, R, sc


def bloch_paths(bw, thicknesses):
    """Complex amplitudes (thickness, beam) from the four paths."""
    sfa = bw._get_structure_factor_array(lazy=False)
    args = dict(
        structure_factor=sfa._eager_array, hkl=sfa.hkl, hkl_selected=bw.hkl,
        cell=bw.cell, energy=bw.energy, gpts=sfa.gpts,
    )
    i0 = np.flatnonzero(np.all(bw.hkl == 0, axis=1))[0]
    out = {}

    out["new eig"] = asnumpy(
        bw.calculate_diffraction_patterns(thicknesses, return_complex=True, lazy=False).array
    )
    # the module function with an eagerly built matrix, rather than
    # BlochWaves.calculate_scattering_matrix, which fails on GPU before abTEM
    # 42846a5b (it built the matrix lazily)
    A_new = bw.calculate_structure_matrix(lazy=False)
    out["new expm"] = np.stack(
        [asnumpy(dyn.calculate_scattering_matrix(
            A=A_new, hkl=bw.hkl, cell=bw.cell, z=z, energy=bw.energy,
            use_wave_eq=bw.use_wave_eq))[:, i0]
         for z in thicknesses]
    )
    A_old = old_structure_matrix(use_wave_eq=bw.use_wave_eq, **args)
    out["old eig"] = asnumpy(
        old_dynamical_scattering(A_old, bw.hkl, bw.cell, bw.energy, thicknesses)
    )
    out["old expm"] = np.stack(
        [asnumpy(old_scattering_matrix(A_old, bw.hkl, bw.cell, z, bw.energy))[:, i0]
         for z in thicknesses]
    )
    return out


def r_factor(a, b):
    return 100 * np.abs(a - b).sum() / b.sum()


def multislice_reference(sc, energy, thickness, order, args):
    """Diffraction pattern of the periodic tilted supercell, and a lookup of
    the intensity at a lab-frame g. Read directly from the pixel g_perp falls
    on: index_diffraction_spots with the supercell would hand each pixel to
    whichever of the many supercell reflections stacked along g_z there has
    the smallest excitation error, often not the actual crystal reflection."""
    from abtem.multislice import FourierMultislice

    reps = max(1, int(round(args.lateral_size / sc.cell[0, 0])))
    sup = sc.repeat((reps, 1, max(1, int(round(thickness / sc.cell[2, 2])))))
    pot = abtem.Potential(sup, sampling=args.sampling, slice_thickness=args.slice_thickness,
                          parametrization="lobato")
    dp = (abtem.PlaneWave(energy=energy)
          .multislice(pot, algorithm=FourierMultislice(order=order))
          .diffraction_patterns(max_angle=None)
          .compute())
    array = asnumpy(dp.array)
    shape = np.array(array.shape[-2:])
    sampling = np.array(dp.sampling)

    def lookup(g_lab):
        pixel = np.round(g_lab[:, :2] / sampling).astype(int) + shape // 2
        return array[..., pixel[:, 0], pixel[:, 1]]

    return lookup, pot.thickness


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--energies", type=float, nargs="+", default=[100e3, 300e3])
    p.add_argument("--tilt-index", type=int, default=8, help="beam || [0 1 k] (default 8: 7.1 deg)")
    p.add_argument("--thickness", type=float, default=300.0)
    p.add_argument("--g-max", type=float, default=6.0)
    p.add_argument("--sg-max", type=float, default=1.0)
    p.add_argument("--g-max-store", type=float, default=2.0, help="compare reflections with |g| < this")
    p.add_argument("--ms", action="store_true", help="also compare against multislice (slower)")
    p.add_argument("--sampling", type=float, default=0.04)
    p.add_argument("--slice-thickness", type=float, default=0.25)
    p.add_argument("--lateral-size", type=float, default=20.0, help="approx. MS supercell size along x [Å]")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="float64")
    args = p.parse_args()

    abtem.config.set({"device": args.device, "precision": args.precision})
    uc, R, sc = tilted_setup(args.tilt_index)
    print(f"Si, beam || [0 1 {args.tilt_index}] ({np.degrees(np.arccos(R[2, 2])):.2f} deg off [001]), "
          f"thickness {args.thickness} Å, g_max {args.g_max}, sg_max {args.sg_max}, "
          f"{args.device}/{args.precision}")

    sf = StructureFactor(uc, g_max=args.g_max, parametrization="lobato", centering="F")

    for energy in args.energies:
        K = 1 / energy2wavelength(energy)
        ms = {}
        if args.ms:
            for order in (1, "exact"):
                t0 = time.time()
                ms[order], z_ms = multislice_reference(sc, energy, args.thickness, order, args)
                print(f"  [{energy/1e3:.0f} keV] multislice order={order}: {time.time()-t0:.0f}s")
            thicknesses = np.array([z_ms])
        else:
            thicknesses = np.linspace(args.thickness / 3, args.thickness, 3)

        print(f"\n=== {energy/1e3:.0f} keV (thicknesses {np.round(thicknesses, 1)} Å) ===")
        for w in (False, True, "exact"):
            bw = BlochWaves(structure_factor=sf, energy=energy, sg_max=args.sg_max,
                            use_wave_eq=w, orientation_matrix=R)
            t0 = time.time()
            paths = bloch_paths(bw, thicknesses)
            g = bw.g_vec
            keep = ~np.all(bw.hkl == 0, axis=1) & (np.linalg.norm(g, axis=1) < args.g_max_store)
            I = {k: np.abs(v) ** 2 for k, v in paths.items()}
            flux_w = np.ones(len(g)) if w else 1 + g[:, 2] / K
            print(f"use_wave_eq={w!s:5s}  N={len(bw.hkl)} beams  ({time.time()-t0:.1f}s)")
            print(f"   R(old eig,  new eig)  = {r_factor(I['old eig'][:, keep],  I['new eig'][:, keep]):8.4f} %   <- effect on calculate_diffraction_patterns / py3DED")
            print(f"   R(old expm, new expm) = {r_factor(I['old expm'][:, keep], I['new expm'][:, keep]):8.4f} %")
            print(f"   max|psi eig - psi expm|: old {np.abs(paths['old eig'] - paths['old expm']).max():.2e}   "
                  f"new {np.abs(paths['new eig'] - paths['new expm']).max():.2e}")
            flux = {k: (flux_w * I[k]).sum(-1) for k in ("old eig", "new eig")}
            print(f"   conserved flux ({'sum |psi|^2' if w else 'sum (1+g_z/K)|psi|^2'}): "
                  f"old eig {np.round(flux['old eig'], 7)}   new eig {np.round(flux['new eig'], 7)}")
            if args.ms:
                order = "exact" if w == "exact" else 1
                I_ms = ms[order](g[keep]).reshape(len(thicknesses), -1)
                print(f"   R(MS order={order}, BW): old {r_factor(I_ms, I['old eig'][:, keep]):7.3f} %   "
                      f"new {r_factor(I_ms, I['new eig'][:, keep]):7.3f} %   ({keep.sum()} reflections)")


if __name__ == "__main__":
    main()
