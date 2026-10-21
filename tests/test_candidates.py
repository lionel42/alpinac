import numpy as np

from conftest import legacy_oracle
from alpinac.candidates import CandidateSet, remove_singletons
from alpinac.elements import N_ELEMENTS
from alpinac.formula import format_formula, parse_formula


def legacy_solutions(ref) -> list[np.ndarray]:
    """The legacy knapsack solutions of each peak, as count arrays."""
    return [
        np.array([parse_formula(f) for f in sols], dtype=np.int64).reshape(
            -1, N_ELEMENTS
        )
        for sols in ref["knapsack"]
    ]


def make(formulas: list[str], peaks: list[list[int]]) -> CandidateSet:
    return CandidateSet(np.array([parse_formula(f) for f in formulas]), peaks)


def test_order_relations():
    c = make(["CCl3", "CCl2", "CCl", "Cl", "CF"], [[3], [2], [1], [0], [4]])
    assert set(c.descendants(0)) == {1, 2, 3}
    assert set(c.ancestors(3)) == {0, 1, 2}
    assert set(c.maximal()) == {0, 4}
    assert set(c.singletons()) == {4}
    assert sorted(c.hasse_edges()) == [(0, 1), (1, 2), (2, 3)]


def test_singleton_kept_if_only_candidate_of_its_peak():
    c = make(["CCl3", "CCl2", "SF"], [[2], [1], [0]])
    new, removed = remove_singletons(c)
    assert len(removed) == 0 and len(new) == 3
    c = make(["CCl3", "CCl2", "SF", "CCl"], [[2], [1], [0], [0]])
    new, removed = remove_singletons(c)
    assert [format_formula(f) for f in new.formulas] == ["CCl3", "CCl2", "CCl"]


def test_twins_merged():
    sols = [
        np.array([parse_formula("CCl")]),
        np.array([parse_formula("CCl"), parse_formula("CCl2")]),
    ]
    c, n_per_peak = CandidateSet.from_knapsack(sols)
    assert len(c) == 2
    assert sorted(c.peaks[c.find(parse_formula("CCl"))]) == [0, 1]
    assert list(n_per_peak) == [1, 2]


def test_step2_matches_legacy(reference_file):
    for ref in legacy_oracle(reference_file):
        cands, n_per_peak = CandidateSet.from_knapsack(legacy_solutions(ref))
        cands, _ = remove_singletons(cands, n_per_peak)
        names = [format_formula(f) for f in cands.formulas]
        new = sorted([n, sorted(p)] for n, p in zip(names, cands.peaks))
        old = sorted(
            [format_formula(parse_formula(f)), p] for f, p in ref["nodes_step2"]
        )
        assert new == old
        assert sorted(names[i] for i in cands.maximal()) == sorted(
            format_formula(parse_formula(f)) for f in ref["maximal_step2"]
        )
        # descendants of the legacy graph = sub-formulas among the candidates
        legacy_desc = {
            format_formula(parse_formula(f)): {
                format_formula(parse_formula(d)) for d in n["descendants"]
            }
            for f, n in ref["nodes"].items()
        }
        for i, name in enumerate(names):
            assert {names[j] for j in cands.descendants(i)} == legacy_desc[name]
