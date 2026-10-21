"""The identification of one compound (Guillevic et al. 2021, Fig. 1).

1. knapsack: candidate formulas for each measured mass
2. candidate set: sub-formula relations, singletons removed
3. isotopologues of each candidate, matched to the measured peaks
4. initial contribution of each candidate
5-8. iterative optimisation: the most likely candidates (and their
     sub-fragments) are added and their isotopologue profiles fitted, until
     enough of the signal and of the masses are explained
9. candidates never added are removed
10. molecular ion candidates (see :mod:`alpinac.molecular_ion`)
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from alpinac.candidates import CandidateSet, remove_singletons
from alpinac.config import IdentificationConfig
from alpinac.fitting import fit_profiles, mass_axis, pseudo_voigt
from alpinac.isotopes import Isotopologues
from alpinac.knapsack import knapsack
from alpinac.peaks import Compound
from alpinac.results import IdentificationResult
from alpinac.subformulas import count_valid_subformulas

logger = logging.getLogger(__name__)


class NoSolution(Exception):
    """No candidate formula matches the measured masses."""


class State:
    """Working state of the identification of one compound.

    Candidates keep their index for the whole identification (removed
    candidates are only flagged, like the node ids of the legacy graph).
    """

    def __init__(self, compound: Compound, config: IdentificationConfig, cands: CandidateSet):
        self.compound = compound
        self.config = config
        self.cands = cands
        n = len(cands)
        self.alive = np.ones(n, dtype=bool)
        self.min_mass = min(float(compound.neutral_mass.min()), config.min_detect_mass)
        self.isos: list[Isotopologues] = [
            Isotopologues.build(cands.formulas[i], cands.peaks[i], compound, config.isotope_pruning_factor)
            for i in range(n)
        ]
        self.n_sub = np.array(
            [count_valid_subformulas(cands.formulas[i], self.min_mass) for i in range(n)],
            dtype=np.int64,
        )
        self.validated: list[int] = []
        self.fitted_groups: list[tuple[int, ...]] = []
        self.likelihood = np.zeros(n)
        self.signal_percent = np.zeros(n)
        self.ranking: list[int] = []
        self.n_below_lod = 0
        self.n_singletons = 0

    # --- candidates -----------------------------------------------------
    def add_candidate(self, formula: np.ndarray) -> int:
        """Add a constructed candidate (molecular ion without measured peak)."""
        i = self.cands.add(formula)
        self.alive = np.append(self.alive, True)
        self.isos.append(Isotopologues.build(formula, [], self.compound, self.config.isotope_pruning_factor))
        self.n_sub = np.append(self.n_sub, count_valid_subformulas(formula, self.min_mass))
        self.likelihood = np.append(self.likelihood, 0.0)
        self.signal_percent = np.append(self.signal_percent, 0.0)
        return i

    def descendants(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.cands.below[i] & self.alive)

    def maximal(self, among: np.ndarray | None = None) -> np.ndarray:
        idx = np.flatnonzero(self.alive) if among is None else np.asarray(among)
        return self.cands.maximal(idx)

    def remove(self, indices) -> None:
        for i in indices:
            self.alive[i] = False
            if i in self.validated:
                self.validated.remove(i)

    def remove_singletons(self) -> list[int]:
        """Remove singletons among the alive candidates (current counts per peak)."""
        alive = np.flatnonzero(self.alive)
        sub = self.cands.subset(alive)
        _, removed = remove_singletons(sub)
        # peaks of kept twins may have been reduced
        for k, i in enumerate(alive):
            self.cands.peaks[i] = sub.peaks[k]
        removed = [int(alive[r]) for r in removed]
        self.remove(removed)
        self.n_singletons += len(removed)
        return removed

    def matched_peaks(self, i: int) -> set[int]:
        p = self.isos[i].peak
        return set(p[p >= 0].tolist())

    # --- step 5 / 8: likelihood ------------------------------------------
    def rank(self) -> list[int]:
        """Order the alive candidates by decreasing likelihood.

        ``likelihood = (100 - |100 - S|) * (found + n_desc) / n_sub`` where S
        is the percentage of the signal explained by the candidate and its
        sub-fragments, ``found`` is 1 if it was proposed for a measured peak,
        ``n_desc`` its number of sub-fragments among the candidates and
        ``n_sub`` its number of possible (valid) sub-formulas.
        """
        alive = np.flatnonzero(self.alive)
        totals = np.array([self.isos[i].total for i in range(len(self.isos))])
        below = self.cands.below[np.ix_(alive, alive)]
        sub_sum = totals[alive] + below.astype(float) @ totals[alive]
        pct = 100.0 * sub_sum / self.compound.sum_area
        found = np.array([1 if self.cands.peaks[i] else 0 for i in alive])
        n_desc = below.sum(axis=1)
        lik = (100.0 - np.abs(100.0 - pct)) * (found + n_desc) / self.n_sub[alive]
        self.likelihood[alive] = lik
        self.signal_percent[alive] = pct
        # Ties: higher index first, as the legacy sort of (likelihood, node id)
        order = np.lexsort((-alive, -lik))
        self.ranking = [int(alive[o]) for o in order]
        return self.ranking

    def signal_stats(self) -> tuple[float, float, int, int]:
        """Explained signal and number of explained masses.

        :return: summed fitted intensity of the validated candidates, its
            percentage of the measured signal, the number of masses explained by
            validated candidates and the number of masses with a candidate.
        """
        sum_signal = float(sum(self.isos[i].total for i in self.validated))
        to_opt: set[int] = set()
        for i in np.flatnonzero(self.alive):
            to_opt |= self.matched_peaks(int(i))
        opt: set[int] = set()
        for i in self.validated:
            opt |= self.matched_peaks(i)
        return sum_signal, 100.0 * sum_signal / self.compound.sum_area, len(opt), len(to_opt)

    # --- step 7: fit ------------------------------------------------------
    def optimise(self) -> None:
        """Fit the isotopologue profiles of the validated candidates (legacy ``graph_optimise_isotopologues``)."""
        comp, config = self.compound, self.config
        groups: list[tuple[list[int], int]] = []
        for s in range(comp.n_spectra):
            nodes = [
                i for i in self.validated
                if any(comp.spectrum_id[p] == s for p in self.cands.peaks[i])
            ]
            nodes.sort(key=lambda i: (self._abundant_mass(i), i))
            i = 0
            while i < len(nodes):
                iso = self.isos[nodes[i]]
                w_stop = round(iso.mass_max) + comp.mass_axis_offset
                j = i
                while j + 1 < len(nodes) and self.isos[nodes[j + 1]].mass_min < w_stop:
                    j += 1
                    # legacy: the window end is extended without rounding nor offset
                    w_stop = max(w_stop, self.isos[nodes[j]].mass_max)
                groups.append((nodes[i: j + 1], s))
                i = j + 1

        already = [tuple(g) in self.fitted_groups for g, _ in groups]
        for gi, (group, s) in enumerate(groups):
            if already[gi]:
                continue
            group = self._fit_group(group, s)
            groups[gi] = (group, s)

        removed: list[int] = []
        for i in list(self.validated):
            if self.isos[i].total * config.below_lod_factor < comp.lod:
                logger.debug("candidate below LOD removed: %s", i)
                removed.append(i)
        self.remove(removed)
        self.n_below_lod += len(removed)
        removed += self.remove_singletons()

        removed_set = set(removed)
        for gi, (group, s) in enumerate(groups):
            has_removed = any(i in removed_set for i in group)
            key = tuple(group)
            if has_removed and already[gi]:
                self.fitted_groups.remove(key)
            elif not has_removed and not already[gi]:
                self.fitted_groups.append(key)

    def _abundant_mass(self, i: int) -> float:
        iso = self.isos[i]
        return float(iso.mass[iso.is_abundant][0])

    def _fit_group(self, group: list[int], s: int) -> list[int]:
        comp, config = self.compound, self.config
        isos = [self.isos[i] for i in group]
        w_start = round(min(iso.mass_min for iso in isos))
        w_stop = round(max(iso.mass_max for iso in isos))
        x = mass_axis(w_start, w_stop, config.ppm_mass_res, comp.mass_axis_offset)
        in_window = np.flatnonzero(
            (comp.mass >= x[0]) & (comp.mass <= x[-1]) & (comp.spectrum_id == s)
        )
        if len(in_window) == 0:
            return group
        # Not more candidates than points: drop the ones with the smallest k
        if len(x) < len(group):
            order = np.argsort([-self.isos[i].k[s] for i in group], kind="stable")
            group = [group[o] for o in sorted(order[: len(x)])]
            isos = [self.isos[i] for i in group]

        m = comp.mass[in_window]
        area = comp.area[in_window]
        sigma = comp.sigma(m)

        def measured(offset: float) -> np.ndarray:
            return pseudo_voigt(x, area, m + offset, sigma, comp.alpha).sum(axis=0)

        profiles = np.array(
            [
                pseudo_voigt(x, iso.rel, iso.mass, comp.sigma(iso.mass), comp.alpha).sum(axis=0)
                for iso in isos
            ]
        )
        k_min = min(area.min(), comp.lod_peak[in_window].min(), comp.lod) / 1000.0
        fixed = comp.unit_mass or (len(in_window) == 1 and len(group) > 1)
        offset_bound = None if fixed else 2.0 * float(comp.mass_U[in_window].min())
        k_init = np.array([iso.k[s] for iso in isos])
        k, _ = fit_profiles(x, profiles, measured, k_min, offset_bound, k_init=k_init)
        for iso, kj in zip(isos, k):
            new_k = iso.k.copy()
            new_k[s] = kj
            iso.set_k(new_k)
        return group


def _step5to8(state: State) -> int:
    """Iterative optimisation, at least ``min_no_masses`` masses. Returns the number of iterations."""
    config = state.config
    state.rank()
    optimised_masses: set[int] = set()
    n_iter = 0
    n_added = 0
    pct = 0.0
    # Start value: any value above max_ratio_not_opt_masses
    ratio_not_opt = float(sum(1 for p in state.cands.peaks if p))
    not_opt_masses = True
    while (
        len(state.validated) < int(state.alive.sum())
        and not_opt_masses
        and n_added < config.max_added_nodes
        and (pct <= config.max_opt_signal or ratio_not_opt > config.max_ratio_not_opt_masses)
    ):
        ranking = state.ranking
        if pct > config.max_opt_signal and not_opt_masses:
            # Enough signal: prefer candidates explaining masses not yet optimised
            first = [i for i in ranking if not (state.matched_peaks(i) & optimised_masses)]
            if first:
                ranking = first + [i for i in ranking if i not in set(first)]
                state.ranking = ranking
        node = next((i for i in ranking if state.alive[i] and i not in state.validated), None)
        if node is None:
            break
        n_iter += 1
        state.validated.append(node)
        added = 1
        for i in state.validated:
            optimised_masses |= state.matched_peaks(i)
        for d in state.descendants(node):
            if int(d) not in state.validated:
                state.validated.append(int(d))
                added += 1
        n_added += added
        state.optimise()
        state.rank()
        _, pct, n_opt, n_to_opt = state.signal_stats()
        ratio_not_opt = (n_to_opt - n_opt) / n_to_opt if n_to_opt > 0 else 0.0
        if n_opt >= n_to_opt:
            not_opt_masses = False
        logger.debug("iteration %d: %.1f%% signal, %d/%d masses", n_iter, pct, n_opt, n_to_opt)
    return n_iter


def _step5to8_few_masses(state: State) -> None:
    """Few measured masses: each maximal fragment is optimised separately."""
    config = state.config
    for node_max in [int(i) for i in state.maximal()]:
        if not state.alive[node_max]:
            continue
        state.validated = [node_max] + [int(d) for d in state.descendants(node_max)]
        state.optimise()
    maximal = set(int(i) for i in state.maximal())
    state.rank()
    state.validated = []
    pct = 0.0
    for node in state.ranking:
        if pct > config.max_opt_signal:
            break
        if node in maximal and node not in state.validated:
            state.validated.append(node)
            for d in state.descendants(node):
                if int(d) not in state.validated:
                    state.validated.append(int(d))
            _, pct, _, _ = state.signal_stats()


def _step9(state: State) -> None:
    """Remove the candidates that were never validated, and singletons."""
    _, old_pct, _, _ = state.signal_stats()
    not_validated = [int(i) for i in np.flatnonzero(state.alive) if int(i) not in state.validated]
    state.remove(not_validated)
    state.remove_singletons()
    _, pct, _, _ = state.signal_stats()
    if pct - old_pct < -5.0 or pct < state.config.max_opt_signal:
        state.optimise()
        state.rank()


def run(compound: Compound, config: IdentificationConfig) -> State:
    """Steps 1 to 10 for one compound. Returns the final state."""
    from alpinac.molecular_ion import step10

    if compound.unit_mass:
        raise NotImplementedError("Identification of unit mass data is not supported.")

    # Step 1: knapsack, one element space per adduct group
    group_adducts: list[np.ndarray] = []
    spec_group = []
    for adduct in compound.spectrum_adduct:
        for g, a in enumerate(group_adducts):
            if np.array_equal(a, adduct):
                spec_group.append(g)
                break
        else:
            spec_group.append(len(group_adducts))
            group_adducts.append(adduct)
    group = np.array(spec_group)[compound.spectrum_id]
    solutions = knapsack(compound.mass_min, compound.mass_max, group, group_adducts, config)
    if sum(len(s) for s in solutions) == 0:
        raise NoSolution(
            "No formula matches the measured masses. Consider adding other "
            "elements or increasing the mass uncertainty."
        )

    # Step 2: candidates, singletons removed
    cands, n_per_peak = CandidateSet.from_knapsack(solutions)
    cands, removed = remove_singletons(cands, n_per_peak)
    logger.debug("%d candidates, %d singletons removed", len(cands), len(removed))

    # Steps 3-4
    state = State(compound, config, cands)
    state.n_singletons = len(removed)

    # Steps 5-8
    if compound.n_peaks < config.min_no_masses:
        _step5to8_few_masses(state)
    else:
        _step5to8(state)

    # Step 9
    _step9(state)

    # Step 10
    step10(state)
    return state


def identify(
    peaks: pd.DataFrame | Compound,
    config: IdentificationConfig | None = None,
    compound: int | str = 0,
    **config_changes,
) -> IdentificationResult:
    """Identify one compound from its peaks.

    :param peaks: the peak table of the compound (see :mod:`alpinac.io`)
        or a :class:`~alpinac.peaks.Compound`.
    :param config: parameters; keyword arguments override single fields,
        e.g. ``identify(peaks, target_elements="CHClF")``.
    :param compound: label of the compound in the result.
    """
    from alpinac.molecular_ion import build_result

    config = (config or IdentificationConfig()).replace(**config_changes)
    t0 = time.perf_counter()
    comp = peaks if isinstance(peaks, Compound) else Compound.from_peaks(peaks, config)
    try:
        state = run(comp, config)
        result = build_result(state, compound)
    except NoSolution as e:
        result = IdentificationResult(compound=compound, status="no_solution", message=str(e),
                                      rt=comp.rt, n_peaks=comp.n_peaks)
    result.runtime = time.perf_counter() - t0
    return result
