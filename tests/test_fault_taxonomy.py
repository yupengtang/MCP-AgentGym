"""The fault taxonomy and the named held-out sets.

The published trade-off rests on two withheld fault families, which is enough to measure
an effect and not enough to say what it is about: "post-training narrows the diagnosis
prior" and "post-training is bad at these two faults in particular" predict the same
numbers. The taxonomy exists so the withheld set can span mechanisms, and so disjoint
folds can be rotated to separate the two readings.

The first assertion below is the important one. Every number in the README and the
experiment log was measured against the two-family split, so if the default ever changes,
those numbers silently start describing a different experiment while every table still
renders.
"""

from __future__ import annotations

import pytest

from agentgym.incidents.faults import ALL_FAULTS
from agentgym.incidents.generator import (
    DEFAULT_HELDOUT_SET,
    FAULT_GROUPS,
    HELDOUT_SETS,
    heldout_faults,
)

ALL_NAMES = frozenset(f.name for f in ALL_FAULTS)


def test_published_split_is_unchanged():
    """Pins the split every reported number was measured against."""
    assert heldout_faults("published") == frozenset({"NetworkPolicyDeny", "NodeNotReady"})
    assert DEFAULT_HELDOUT_SET == "published"
    assert heldout_faults() == heldout_faults("published"), (
        "the default held-out set must stay the published one; changing it reinterprets "
        "every table in the README and the experiment log without touching a single figure"
    )


def test_every_family_is_in_exactly_one_group():
    """A taxonomy with a gap or an overlap cannot support a 'spans N groups' claim."""
    counts = {name: 0 for name in ALL_NAMES}
    for group, members in FAULT_GROUPS.items():
        unknown = members - ALL_NAMES
        assert not unknown, f"group {group} names families that do not exist: {sorted(unknown)}"
        for m in members:
            counts[m] += 1
    ungrouped = sorted(n for n, c in counts.items() if c == 0)
    duplicated = sorted(n for n, c in counts.items() if c > 1)
    assert not ungrouped, f"families in no group: {ungrouped}"
    assert not duplicated, f"families in more than one group: {duplicated}"


@pytest.mark.parametrize("name", sorted(HELDOUT_SETS))
def test_heldout_set_is_a_real_nonempty_proper_subset(name):
    """Held out must leave something to train on, and must name real families."""
    held = heldout_faults(name)
    unknown = held - ALL_NAMES
    assert not unknown, f"{name} withholds families that do not exist: {sorted(unknown)}"
    assert held, f"{name} withholds nothing"
    train = ALL_NAMES - held
    assert len(train) >= 6, (
        f"{name} leaves only {len(train)} training families; the corpus stops being a "
        f"catalogue the model can specialise to, which is the thing under test"
    )


@pytest.mark.parametrize("name", ["broad", "fold_a", "fold_b", "fold_c"])
def test_the_multi_group_sets_actually_span_groups(name):
    """The point of these sets is mechanism breadth, not just a longer list."""
    held = heldout_faults(name)
    spanned = {g for g, members in FAULT_GROUPS.items() if members & held}
    assert len(spanned) >= 3, (
        f"{name} withholds {len(held)} families but only {len(spanned)} mechanism "
        f"group(s) ({sorted(spanned)}); a set that does not span groups cannot "
        f"distinguish 'unfamiliar mechanism' from 'these particular faults'"
    )


def test_folds_are_disjoint():
    """Rotating folds only separates procedure from family if they do not overlap."""
    a, b, c = (heldout_faults(f) for f in ("fold_a", "fold_b", "fold_c"))
    assert not (a & b), f"fold_a and fold_b overlap: {sorted(a & b)}"
    assert not (a & c), f"fold_a and fold_c overlap: {sorted(a & c)}"
    assert not (b & c), f"fold_b and fold_c overlap: {sorted(b & c)}"


def test_every_family_is_held_out_by_some_fold_or_deliberately_never():
    """Families in no fold are a deliberate choice, and are recorded as one."""
    covered = set().union(*(heldout_faults(f) for f in ("fold_a", "fold_b", "fold_c")))
    always_train = sorted(ALL_NAMES - covered)
    assert always_train == ["OOMKilled", "ReplicasScaledToZero"], (
        f"the always-trained families changed to {always_train}; they are excluded from "
        f"every fold on purpose, so each fold keeps a trivial family in its corpus"
    )


def test_unknown_set_raises_rather_than_defaulting():
    """A typo that fell back to `published` would label a fold's result wrongly."""
    with pytest.raises(ValueError, match="unknown held-out set"):
        heldout_faults("fold_d")


def test_env_var_selects_the_set(monkeypatch):
    monkeypatch.setenv("AGENTGYM_HELDOUT_SET", "fold_b")
    assert heldout_faults() == HELDOUT_SETS["fold_b"]
    monkeypatch.delenv("AGENTGYM_HELDOUT_SET")
    assert heldout_faults() == HELDOUT_SETS["published"]
