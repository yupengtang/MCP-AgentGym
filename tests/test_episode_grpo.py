"""Tests for episode-level GRPO.

The advantage computation and the loss are the two places where this can be wrong without
raising: a broken advantage still trains, a broken normalisation still converges, and both
produce a model rather than an error. Each property below is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import pytest

from agentgym.train.episode_grpo import group_advantages, grpo_loss

torch = pytest.importorskip("torch")


@dataclass
class FakeTraj:
    incident_id: str
    reward: float
    fault: str = "OOMKilled"


def test_advantages_are_group_relative_and_standardised():
    """Within a group: mean 0, unit scale. That is the whole point of GRPO's baseline."""
    group = [FakeTraj("i1", r) for r in (0.0, 1.0, 2.0, 3.0)]
    scored, stats = group_advantages(group)

    advs = [a for _, a in scored]
    assert len(advs) == 4
    assert sum(advs) == pytest.approx(0.0, abs=1e-5)
    # Population std of the standardised values is 1.
    mean = sum(advs) / len(advs)
    std = (sum((a - mean) ** 2 for a in advs) / len(advs)) ** 0.5
    assert std == pytest.approx(1.0, abs=1e-4)
    assert stats.groups_kept == 1 and stats.episodes_kept == 4


def test_advantage_ordering_follows_reward_ordering():
    group = [FakeTraj("i1", r) for r in (3.0, -1.0, 0.5)]
    scored, _ = group_advantages(group)
    by_reward = sorted(scored, key=lambda p: p[0].reward)
    advs = [a for _, a in by_reward]
    assert advs == sorted(advs), "higher reward must not receive lower advantage"
    assert advs[0] < 0 < advs[-1]


def test_groups_are_scoped_per_incident():
    """An episode must be compared against attempts at the *same* incident.

    Pooling across incidents would make the advantage measure incident difficulty rather
    than behaviour -- the same error the DPO pair construction avoids.
    """
    trajs = [FakeTraj("easy", 3.0), FakeTraj("easy", 2.0),
             FakeTraj("hard", 0.2), FakeTraj("hard", 0.0)]
    scored, stats = group_advantages(trajs)
    assert stats.groups_kept == 2
    # The worst attempt at the easy incident still has reward above the best at the hard
    # one, yet must receive a negative advantage.
    easy_low = next(a for t, a in scored if t.incident_id == "easy" and t.reward == 2.0)
    hard_high = next(a for t, a in scored if t.incident_id == "hard" and t.reward == 0.2)
    assert easy_low < 0 < hard_high


def test_degenerate_group_is_dropped_and_counted():
    """Zero reward spread carries no signal; it must not enter the batch as zeros."""
    trajs = [FakeTraj("flat", 1.0), FakeTraj("flat", 1.0), FakeTraj("flat", 1.0)]
    scored, stats = group_advantages(trajs)
    assert scored == []
    assert stats.dropped_no_reward_spread == 1
    assert stats.groups_kept == 0 and stats.episodes_kept == 0


def test_single_attempt_group_is_dropped_and_counted():
    scored, stats = group_advantages([FakeTraj("lonely", 2.0)])
    assert scored == []
    assert stats.dropped_too_small == 1


def test_group_members_stay_adjacent(monkeypatch):
    """Episodes from one incident must come out contiguous.

    The training loop accumulates gradients without shuffling, which is what lets each
    accumulation window span whole groups so the zero-mean advantage cancels inside it --
    the variance reduction GRPO exists to buy. If `group_advantages` or `build_examples`
    ever interleaves groups, that cancellation quietly stops happening: the run still
    converges, just noisily, and no test or metric reports the change.
    """
    from agentgym.data import curate
    from agentgym.train.episode_grpo import build_examples

    # Rendering the real chat history needs a full Trajectory; ordering is what is under
    # test here, so the renderer is stubbed out.
    monkeypatch.setattr(curate, "_messages_for", lambda t: [{"role": "user", "content": ""}])

    trajs = []
    for inc in ("a", "b", "c"):
        for r in (0.0, 1.0, 2.0, 3.0):
            trajs.append(FakeTraj(inc, r))
    scored, stats = group_advantages(trajs)
    assert stats.groups_kept == 3

    order = [t.incident_id for t, _ in scored]
    # Each incident occupies exactly one contiguous run.
    runs = [k for k, _ in groupby(order)]
    assert len(runs) == len(set(runs)) == 3, f"groups interleaved: {order}"

    examples = build_examples(scored)
    ex_order = [e["incident_id"] for e in examples]
    ex_runs = [k for k, _ in groupby(ex_order)]
    assert len(ex_runs) == len(set(ex_runs)) == 3

    # And each contiguous run's advantages still sum to zero, which is the property the
    # accumulation window relies on.
    for inc in ("a", "b", "c"):
        advs = [e["advantage"] for e in examples if e["incident_id"] == inc]
        assert sum(advs) == pytest.approx(0.0, abs=1e-5)


def test_group_shuffle_permutes_groups_but_never_splits_one(monkeypatch):
    """Shuffling must reorder whole groups, never interleave their members.

    Interleaving would scatter a group across accumulation windows, so the zero-mean
    advantage no longer cancels inside any single update -- the variance reduction GRPO
    exists to buy, given up silently while the run still converges.
    """
    from agentgym.data import curate
    from agentgym.train.episode_grpo import build_examples, shuffle_by_group

    monkeypatch.setattr(curate, "_messages_for", lambda t: [{"role": "user", "content": ""}])

    trajs = [FakeTraj(inc, r) for inc in "abcdefgh" for r in (0.0, 1.0, 2.0, 3.0)]
    scored, _ = group_advantages(trajs)
    examples = build_examples(scored)

    shuffled = shuffle_by_group(examples, seed=7)

    # Same episodes, no loss or duplication.
    assert len(shuffled) == len(examples)
    assert sorted(id(e) for e in shuffled) == sorted(id(e) for e in examples)

    # Still exactly one contiguous run per incident.
    runs = [k for k, _ in groupby(e["incident_id"] for e in shuffled)]
    assert len(runs) == len(set(runs)) == 8, f"a group was split: {runs}"

    # Members keep their own advantages, and each group still sums to zero.
    for inc in "abcdefgh":
        advs = [e["advantage"] for e in shuffled if e["incident_id"] == inc]
        assert len(advs) == 4
        assert sum(advs) == pytest.approx(0.0, abs=1e-5)

    # It actually reorders, and does so deterministically for a given seed.
    assert runs != [k for k, _ in groupby(e["incident_id"] for e in examples)]
    assert [e["incident_id"] for e in shuffle_by_group(examples, seed=7)] == \
           [e["incident_id"] for e in shuffled]
    assert [e["incident_id"] for e in shuffle_by_group(examples, seed=8)] != \
           [e["incident_id"] for e in shuffled]


def test_stats_round_trip_to_dict():
    _, stats = group_advantages([FakeTraj("i", 1.0), FakeTraj("i", 2.0)])
    d = stats.as_dict()
    assert d["groups_kept"] == 1
    assert isinstance(d["reward_range"], list)


# --- loss ---------------------------------------------------------------------

class _StubModel:
    """Returns fixed logits; `ref_context()` swaps in a different set as the reference."""

    def __init__(self, logits, ref_logits=None):
        self._logits = logits
        self._ref = ref_logits if ref_logits is not None else logits
        self._adapter_on = True
        self.saw_policy_pass_with_adapter = False

    def __call__(self, input_ids=None, attention_mask=None):
        if self._adapter_on:
            self.saw_policy_pass_with_adapter = True
        out = self._logits if self._adapter_on else self._ref

        class _O:
            logits = out
        return _O()

    def ref_context(self):
        model = self

        class _Ctx:
            def __enter__(self):
                model._adapter_on = False

            def __exit__(self, *exc):
                model._adapter_on = True
        return _Ctx()


def _batch(adv, seq_len=6, vocab=8, batch=2, mask_from=2):
    ids = torch.randint(0, vocab, (batch, seq_len))
    mask = torch.zeros((batch, seq_len), dtype=torch.long)
    mask[:, mask_from:] = 1
    return {
        "input_ids": ids,
        "attention_mask": torch.ones((batch, seq_len), dtype=torch.long),
        "assistant_mask": mask,
        "advantages": torch.tensor(adv, dtype=torch.float32),
    }


def test_policy_forward_runs_with_adapter_enabled():
    """Regression: computing the policy pass inside `disable_adapter()` trains the
    adapter on gradients taken without it. The bug produces a trained model, not an
    error, so it is pinned here."""
    vocab, seq, bs = 8, 6, 2
    logits = torch.randn(bs, seq, vocab)
    model = _StubModel(logits)
    grpo_loss(model, _batch([1.0, -1.0]), beta=0.1, ref_context=model.ref_context)
    assert model.saw_policy_pass_with_adapter
    assert model._adapter_on, "adapter must be re-enabled after the reference pass"


def test_positive_advantage_rewards_higher_likelihood():
    """A more confident policy on a positively-advantaged episode must score lower loss."""
    vocab, seq, bs = 8, 6, 1
    b = _batch([1.0], seq_len=seq, vocab=vocab, batch=bs)
    flat = torch.zeros(bs, seq, vocab)
    peaked = torch.zeros(bs, seq, vocab)
    for t in range(seq - 1):
        peaked[0, t, b["input_ids"][0, t + 1]] = 6.0

    loss_flat, _ = grpo_loss(_StubModel(flat), b, beta=0.0)
    loss_peaked, _ = grpo_loss(_StubModel(peaked), b, beta=0.0)
    assert float(loss_peaked) < float(loss_flat)


def test_negative_advantage_flips_the_direction():
    vocab, seq, bs = 8, 6, 1
    b = _batch([-1.0], seq_len=seq, vocab=vocab, batch=bs)
    flat = torch.zeros(bs, seq, vocab)
    peaked = torch.zeros(bs, seq, vocab)
    for t in range(seq - 1):
        peaked[0, t, b["input_ids"][0, t + 1]] = 6.0

    loss_flat, _ = grpo_loss(_StubModel(flat), b, beta=0.0)
    loss_peaked, _ = grpo_loss(_StubModel(peaked), b, beta=0.0)
    assert float(loss_peaked) > float(loss_flat), \
        "a bad episode must be made less likely, not more"


def test_loss_is_length_normalised():
    """Doubling the number of assistant tokens at identical per-token likelihood must not
    change the loss -- otherwise the objective prefers long episodes whenever the
    advantage is positive, and the agent learns to pad."""
    vocab = 8
    torch.manual_seed(0)

    def loss_for(seq_len, mask_from):
        ids = torch.zeros((1, seq_len), dtype=torch.long)
        logits = torch.zeros(1, seq_len, vocab)  # uniform -> constant per-token logprob
        mask = torch.zeros((1, seq_len), dtype=torch.long)
        mask[:, mask_from:] = 1
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            "assistant_mask": mask,
            "advantages": torch.tensor([1.0]),
        }
        return float(grpo_loss(_StubModel(logits), batch, beta=0.0)[0])

    assert loss_for(10, 2) == pytest.approx(loss_for(20, 2), abs=1e-5)


def test_kl_is_zero_against_identical_reference_and_positive_otherwise():
    vocab, seq, bs = 8, 6, 2
    same = torch.randn(bs, seq, vocab)
    m = _StubModel(same, same)
    _, parts = grpo_loss(m, _batch([1.0, -1.0]), beta=0.1, ref_context=m.ref_context)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-5)

    m2 = _StubModel(same, torch.randn(bs, seq, vocab))
    _, parts2 = grpo_loss(m2, _batch([1.0, -1.0]), beta=0.1, ref_context=m2.ref_context)
    assert parts2["kl"] > 0.0, "k3 estimator must be non-negative and non-trivial"


def test_kl_does_not_swamp_the_policy_gradient_at_step_zero():
    """Regression, measured in job 5545540.

    The reference was taken as the *base* model (`disable_adapter()`) while the policy
    started from the SFT adapter, so KL(SFT || base) was already enormous before any
    update. At beta=0.04 the penalty was three orders of magnitude above the policy
    gradient -- pg = -0.86 against beta*KL = 1298.9 -- and the run optimised "return to
    base" instead of "improve the policy".

    The reference must be the policy training started from, which makes the KL exactly
    zero on the first step. This asserts the *relationship*, not just that the code runs.
    """
    vocab, seq, bs = 8, 6, 2
    policy = torch.randn(bs, seq, vocab)
    m = _StubModel(policy, policy)  # reference == starting policy
    loss, parts = grpo_loss(m, _batch([1.0, -1.0]), beta=0.04,
                            ref_context=m.ref_context)
    assert parts["kl"] == pytest.approx(0.0, abs=1e-5)
    assert float(loss) == pytest.approx(parts["pg_loss"], abs=1e-5), \
        "at step 0 the objective must be the policy gradient alone"


def test_kl_estimator_is_finite_under_extreme_log_ratios():
    """A fine-tuned policy and its reference can disagree by many nats on rare tokens.
    Unclamped, exp(log-ratio) overflows and the loss becomes inf/nan mid-run."""
    vocab, seq, bs = 8, 6, 2
    policy = torch.zeros(bs, seq, vocab)
    extreme = torch.zeros(bs, seq, vocab)
    extreme[:, :, 0] = 500.0  # reference is wildly confident on one token
    m = _StubModel(policy, extreme)
    loss, parts = grpo_loss(m, _batch([1.0, -1.0]), beta=0.04, ref_context=m.ref_context)
    assert torch.isfinite(loss).all()
    assert parts["kl"] >= 0.0


def test_masked_prefix_tokens_do_not_contribute():
    """Tokens outside the assistant's turns carry no gradient signal."""
    vocab, seq = 8, 8
    torch.manual_seed(1)
    logits = torch.randn(1, seq, vocab)
    ids = torch.randint(0, vocab, (1, seq))

    def loss_with(mask_row):
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones((1, seq), dtype=torch.long),
            "assistant_mask": mask_row,
            "advantages": torch.tensor([1.0]),
        }
        return float(grpo_loss(_StubModel(logits), batch, beta=0.0)[0])

    m = torch.zeros((1, seq), dtype=torch.long)
    m[:, 4:] = 1
    baseline = loss_with(m)

    # Changing only the masked-out region's ids must not move the loss.
    ids = ids.clone()
    ids[0, 0] = (ids[0, 0] + 1) % vocab
    assert loss_with(m) == pytest.approx(baseline, abs=1e-6)
