"""Step 10: molecular ion candidates, and the final result.

The molecular ion may or may not be among the measured masses. For each
spectrum, starting from the maximal fragments of that spectrum:

- with the (CI) adduct removed or not, a maximal fragment with an integer
  DBE fulfils condition (i) of the SENIOR rules; if it also fulfils condition
  (ii) (sum of valences >= 2 * max valence), it is a molecular ion candidate;
- a fragment with a half-integer DBE (odd sum of valences) gets one more
  mono-valent atom, of each mono-valent element present in the maximal
  fragments; the result is a candidate if it fulfils condition (ii).

This assumes all multi-valent atoms of the molecule were detected in some
fragment, and that at most one mono-valent atom was lost by the maximal
fragments (Guillevic et al. 2021). Constructed candidates are added to the
candidate set (without measured peak) and ranked with the others.

Molecular ions are singly charged: a maximal doubly charged fragment gives the
singly charged ion of the same formula.
"""

from __future__ import annotations

import numpy as np

from alpinac.elements import ADDUCT_TO_ELEMENT, N_ELEMENTS, VALENCE
from alpinac.formula import (
    dbe,
    format_formula,
    ion_mass,
    remove_adduct_elements,
    senior_rule_ii,
)
from alpinac.results import Fragment, IdentificationResult, MolecularIon, PeakAssignment

MONO_VALENT = np.flatnonzero(VALENCE == 1)


def _sub_adducts(adduct: np.ndarray) -> list[np.ndarray]:
    """All non-zero sub-formulas of an adduct (pseudo-element counts)."""
    idx = np.flatnonzero(adduct)
    if len(idx) == 0:
        return []
    out = []
    grids = np.meshgrid(*[np.arange(adduct[i] + 1) for i in idx], indexing="ij")
    for combo in zip(*[g.ravel() for g in grids]):
        if sum(combo) == 0:
            continue
        a = np.zeros(N_ELEMENTS, dtype=np.int64)
        a[idx] = combo
        out.append(a)
    return out


def _contains(rows: list[np.ndarray], f: np.ndarray) -> bool:
    return any(np.array_equal(r, f) for r in rows)


def step10(state) -> None:
    """Find or construct the molecular ion candidates of each spectrum.

    Fills ``state.mol_ions``: for each spectrum, the candidate indices.
    """
    comp = state.compound
    state.mol_ions = [[] for _ in range(comp.n_spectra)]
    if not state.alive.any():
        return
    for s in range(comp.n_spectra):
        members = [i for i in np.flatnonzero(state.alive) if state.isos[i].belongs[s]]
        if not members:
            continue
        max_nodes = [int(i) for i in state.maximal(members)]
        variants = [None] + _sub_adducts(comp.spectrum_adduct[s])

        mol_frags: list[np.ndarray] = []
        mol_nodes: list[int | None] = []
        odd_frags: list[np.ndarray] = []
        for node in max_nodes:
            for adduct in variants:
                cand = state.cands.formulas[node].copy()
                if adduct is not None:
                    for i_ad in np.flatnonzero(adduct):
                        el = ADDUCT_TO_ELEMENT[i_ad]
                        cand[el] -= min(cand[el], adduct[i_ad])
                d = dbe(cand)
                if d < 0:
                    continue
                if d == int(d):
                    if senior_rule_ii(cand) and not _contains(mol_frags, cand):
                        mol_frags.append(cand)
                        as_is = adduct is None and state.cands.charge[node] == 1
                        mol_nodes.append(node if as_is else None)
                elif not _contains(odd_frags, cand):
                    odd_frags.append(cand)

        if odd_frags:
            frag_max = np.max(np.array(odd_frags + mol_frags), axis=0)
            frag_max[MONO_VALENT] += (frag_max[MONO_VALENT] > 0).astype(np.int64)
            for cand in odd_frags:
                for el in MONO_VALENT:
                    if cand[el] < frag_max[el]:
                        new = cand.copy()
                        new[el] += 1
                        if not _contains(mol_frags, new) and senior_rule_ii(new):
                            mol_frags.append(new)
                            mol_nodes.append(None)

        for f, node in zip(mol_frags, mol_nodes):
            if node is None:
                found = state.cands.find(f)
                if found is not None and state.alive[found]:
                    node = found
                else:
                    node = state.add_candidate(f)
            if node not in state.mol_ions[s]:
                state.mol_ions[s].append(node)
    state.rank()


def charge_suffix(charge: int) -> str:
    """Suffix of the formulas of multiply charged ions (``"++"``), empty for a single charge."""
    return "+" * charge if charge > 1 else ""


def build_result(state, compound_label) -> IdentificationResult:
    """Convert the final state into an :class:`IdentificationResult`."""
    comp = state.compound
    alive = [i for i in state.ranking if state.alive[i]]
    maximal = set(int(i) for i in state.maximal()) if alive else set()
    charge = {i: int(state.cands.charge[i]) for i in alive}
    names = {
        i: format_formula(state.cands.formulas[i]) + charge_suffix(charge[i])
        for i in alive
    }

    fragments = [
        Fragment(
            rank=r + 1,
            formula=names[i],
            mass=ion_mass(state.cands.formulas[i], charge[i]),
            dbe=dbe(state.cands.formulas[i]),
            likelihood=float(state.likelihood[i]),
            signal_percent=float(state.signal_percent[i]),
            intensity=state.isos[i].total,
            is_maximal=i in maximal,
            measured=bool(state.cands.peaks[i]),
            charge=charge[i],
        )
        for r, i in enumerate(alive)
    ]

    mol_ions = []
    for s in range(comp.n_spectra):
        nodes = set(getattr(state, "mol_ions", [[]] * comp.n_spectra)[s])
        rank = 0
        for i in alive:
            if i in nodes:
                rank += 1
                mol_ions.append(
                    MolecularIon(
                        rank=rank,
                        formula=names[i],
                        mass=ion_mass(state.cands.formulas[i]),
                        dbe=dbe(state.cands.formulas[i]),
                        likelihood=float(state.likelihood[i]),
                        signal_percent=float(state.signal_percent[i]),
                        spectrum=comp.spectra[s],
                        adduct=comp.spectrum_adduct_formula[s],
                        measured=bool(state.cands.peaks[i]),
                    )
                )

    peaks: list[PeakAssignment] = []
    explained = set()
    for i in alive:
        iso = state.isos[i]
        labels = None
        for s in range(comp.n_spectra):
            for j in np.flatnonzero(iso.peak[s] >= 0):
                p = int(iso.peak[s, j])
                labels = labels or [
                    lab + charge_suffix(iso.charge) for lab in iso.labels()
                ]
                explained.add(p)
                peaks.append(
                    PeakAssignment(
                        peak=p,
                        mass=float(comp.mass[p]),
                        area=float(comp.area[p]),
                        spectrum=comp.spectra[s],
                        formula=labels[j],
                        fragment=names[i],
                        exact_mass=float(iso.mass[j]),
                        mass_error_ppm=float(
                            (iso.mass[j] - comp.mass[p]) / comp.mass[p] * 1e6
                        ),
                        assigned_area=float(iso.opt[s, j]),
                        charge=iso.charge,
                    )
                )
    for p in range(comp.n_peaks):
        if p not in explained:
            peaks.append(
                PeakAssignment(
                    peak=p,
                    mass=float(comp.mass[p]),
                    area=float(comp.area[p]),
                    spectrum=comp.spectra[comp.spectrum_id[p]],
                    formula="",
                    fragment="",
                    exact_mass=float("nan"),
                    mass_error_ppm=float("nan"),
                    assigned_area=0.0,
                    charge=0,
                )
            )
    peaks.sort(key=lambda a: (a.peak, -a.assigned_area))

    _, pct, _, _ = state.signal_stats()
    return IdentificationResult(
        compound=compound_label,
        status="ok" if alive else "no_solution",
        message="" if alive else "No candidate left after the optimisation.",
        rt=comp.rt,
        n_peaks=comp.n_peaks,
        signal_percent=float(pct),
        molecular_ions=mol_ions,
        fragments=fragments,
        peaks=peaks,
    )


__all__ = ["build_result", "charge_suffix", "step10", "remove_adduct_elements"]
