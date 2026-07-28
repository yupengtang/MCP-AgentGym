"""agentgym command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from agentgym import config as cfg

app = typer.Typer(add_completion=False, help="MCP-AgentGym")


@app.command("gen-incidents")
def gen_incidents(
    n: int = typer.Option(1200, help="How many incidents to generate."),
    seed: int = typer.Option(0),
    out: Path | None = typer.Option(None, help="Defaults to $AGENTGYM_DATA/incidents.jsonl"),
) -> None:
    """Generate the incident dataset with held-out splits."""
    from agentgym.incidents import generator

    path = out or (cfg.DATA_DIR / "incidents.jsonl")
    incidents = generator.generate(n, seed=seed)
    generator.write_jsonl(incidents, path)

    by_split: dict[str, int] = {}
    by_fault: dict[str, int] = {}
    for inc in incidents:
        by_split[inc.split] = by_split.get(inc.split, 0) + 1
        by_fault[inc.fault] = by_fault.get(inc.fault, 0) + 1
    typer.echo(f"wrote {len(incidents)} incidents -> {path}")
    typer.echo(f"splits: {json.dumps(by_split, indent=2)}")
    typer.echo(f"faults: {json.dumps(dict(sorted(by_fault.items())), indent=2)}")


@app.command("rollout")
def rollout(
    incidents: Path | None = typer.Option(None, help="incidents.jsonl"),
    out: Path | None = typer.Option(None, help="trajectories .jsonl"),
    policy: str = typer.Option("expert", help="expert | llm"),
    limit: int = typer.Option(0, help="0 = all"),
    split: str = typer.Option("", help="Only this split."),
    perturbation: str = typer.Option("clean"),
    samples: int = typer.Option(1, help="Attempts per incident. >1 mines DPO pairs."),
    shard_index: int = typer.Option(-1, help="-1 = read from SLURM_ARRAY_TASK_ID"),
    shard_total: int = typer.Option(-1),
    gym_offset: int = typer.Option(-1, help="-1 = derive from shard index"),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1"),
    model: str = typer.Option("Qwen/Qwen3-8B"),
    adapter: str = typer.Option("", help="LoRA adapter name for multi-LoRA serving."),
    temperature: float = typer.Option(0.0),
    max_steps: int = typer.Option(24),
    verbose: bool = typer.Option(False),
) -> None:
    """Run incidents through a policy and write trajectories."""
    from agentgym import rollout as rollout_mod
    from agentgym.incidents import generator

    inc_path = incidents or (cfg.DATA_DIR / "incidents.jsonl")
    all_incidents = generator.read_jsonl(inc_path)
    if split:
        all_incidents = [i for i in all_incidents if i.split == split]

    if shard_index < 0 or shard_total < 0:
        shard_index, shard_total = rollout_mod.worker_identity()
    # `--limit` is the total for this cell, applied *before* sharding. Applying it after
    # would silently multiply the workload by the shard count -- 8 shards x 32 is 256
    # episodes, not 32 -- which turns a quick sweep into an overnight one without any
    # error to explain why.
    if limit:
        all_incidents = all_incidents[:limit]
    mine = rollout_mod.shard(all_incidents, shard_index, shard_total)
    if samples > 1:
        # Repeat each incident so several attempts share an incident_id. build_dpo pairs
        # within an id, so both sides of a preference face an identical cluster and the
        # contrast is behaviour rather than scenario difficulty.
        mine = [inc for inc in mine for _ in range(samples)]

    offset = gym_offset if gym_offset >= 0 else (100 + shard_index * 3)
    out_path = out or (cfg.RUNS_DIR / f"{policy}-{perturbation}" /
                       f"trajectories-{shard_index:03d}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    if policy == "expert":
        factory = rollout_mod.expert_factory
        model_name = "expert-scripted"
    else:
        from agentgym.backends.openai_compat import VLLMBackend

        backend = VLLMBackend(base_url=base_url, model=model)
        if not backend.wait_until_ready(timeout_s=1200):
            raise typer.Exit(code=2)
        factory = rollout_mod.llm_factory(backend, adapter=adapter or None,
                                          temperature=temperature)
        model_name = adapter or model

    typer.echo(f"shard {shard_index}/{shard_total}: {len(mine)} incidents "
               f"-> {out_path} (policy={policy}, perturbation={perturbation})")

    results = rollout_mod.run_incidents(
        mine, factory,
        rollout_mod.RolloutConfig(
            max_steps=max_steps, max_tool_calls=max_steps,
            perturbation=perturbation, model_name=model_name,
            gym_offset=offset, verbose=verbose,
        ),
        out_path=out_path,
    )
    summary = rollout_mod.summarize(results)
    typer.echo(json.dumps(summary, indent=2))
    (out_path.parent / f"summary-{shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2))


@app.command("build-datasets")
def build_datasets(
    trajectories: Path | None = typer.Option(None, help="File or dir of trajectories."),
    out_dir: Path | None = typer.Option(None),
    eval_fraction: float = typer.Option(0.2, help="Share of in-distribution incidents held out."),
    seed: int = typer.Option(0),
) -> None:
    """Curate SFT / DPO datasets and run the contamination check."""
    import random

    from agentgym.data import curate
    from agentgym.data import trajectory as traj_mod

    src = trajectories or (cfg.RUNS_DIR / "expert-clean")
    files = sorted(src.glob("*.jsonl")) if src.is_dir() else [src]
    all_traj = [t for f in files for t in traj_mod.read_jsonl(f)]
    out = out_dir or (cfg.DATA_DIR / "sft")
    out.mkdir(parents=True, exist_ok=True)

    # Everything outside in-distribution is evaluation by construction; additionally
    # reserve a slice of in-distribution incidents so there is an ID test set too.
    id_ids = sorted({t.incident_id for t in all_traj if t.split == "in_distribution"})
    rng = random.Random(seed)
    rng.shuffle(id_ids)
    held = set(id_ids[: int(len(id_ids) * eval_fraction)])
    ood_ids = {t.incident_id for t in all_traj if t.split != "in_distribution"}
    eval_ids = held | ood_ids

    examples, report = curate.build_sft(all_traj, eval_incident_ids=eval_ids, seed=seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * 0.05))
    curate.write_jsonl(examples[n_val:], out / "sft_train.jsonl")
    curate.write_jsonl(examples[:n_val], out / "sft_val.jsonl")

    check = curate.contamination_check(examples, eval_ids)
    (out / "curation_report.json").write_text(json.dumps({
        "curation": report.as_dict(), "contamination": check,
        "held_out_in_distribution": len(held), "ood_incidents": len(ood_ids),
    }, indent=2))
    (out / "eval_incident_ids.json").write_text(json.dumps(sorted(eval_ids)))

    pairs, dpo_stats = curate.build_dpo(all_traj, eval_incident_ids=eval_ids)
    if pairs:
        curate.write_jsonl(pairs, out / "dpo_pairs.jsonl")

    typer.echo(json.dumps(report.as_dict(), indent=2))
    typer.echo(f"contamination: {json.dumps(check)}")
    typer.echo(f"dpo: {json.dumps(dpo_stats)}")
    typer.echo(f"wrote {len(examples)-n_val} train / {n_val} val -> {out}")


@app.command("stats")
def stats(path: Path) -> None:
    """Summarize a trajectory file (or a directory of them)."""
    from agentgym import rollout as rollout_mod
    from agentgym.data import trajectory as traj_mod

    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    trajectories = [t for f in files for t in traj_mod.read_jsonl(f)]
    typer.echo(f"{len(trajectories)} trajectories from {len(files)} file(s)")
    typer.echo(json.dumps(rollout_mod.summarize(trajectories), indent=2))

    by_fault: dict[str, list] = {}
    for t in trajectories:
        by_fault.setdefault(t.fault, []).append(t)
    typer.echo("\nper fault family:")
    for fault, group in sorted(by_fault.items()):
        solved = sum(1 for t in group if t.success)
        typer.echo(f"  {fault:30s} {solved:4d}/{len(group):<4d} "
                   f"({solved/len(group):6.1%})")


@app.command("publish")
def publish(
    run: Path | None = typer.Option(None, help="Directory of trajectories to publish."),
    adapter: Path | None = typer.Option(None, help="Adapter directory to publish."),
    report: bool = typer.Option(True, help="Publish the evaluation report."),
    uri: str | None = typer.Option(None, help="s3://bucket/prefix or file:///path. "
                                              "Defaults to $AGENTGYM_ARTIFACT_URI."),
    tag: str = typer.Option("latest", help="Namespace for this publication."),
) -> None:
    """Publish artifacts to the configured store (local or S3).

    Results are produced on whatever node Slurm allocated and consumed elsewhere, so the
    durable copy should not live on a scratch filesystem.
    """
    from agentgym.storage import open_store

    store = open_store(uri)
    published: list[str] = []

    if run is not None:
        for path in sorted(run.rglob("*.jsonl")):
            published.append(store.put_file(
                f"{tag}/trajectories/{path.relative_to(run).as_posix()}", path))
    if adapter is not None:
        published.extend(store.put_tree(f"{tag}/adapters/{adapter.name}", adapter))
    if report:
        for name in ("report.json", "report.md"):
            path = cfg.ARTIFACTS_DIR / "evaluation" / name
            if path.exists():
                published.append(store.put_file(f"{tag}/report/{name}", path))

    typer.echo(f"published {len(published)} objects")
    for u in published[:5]:
        typer.echo(f"  {u}")
    if len(published) > 5:
        typer.echo(f"  ... and {len(published) - 5} more")


@app.command("fidelity")
def fidelity(
    out: Path | None = typer.Option(None, help="Where to write the agreement report."),
) -> None:
    """Run the sim-to-real fidelity study (real processes, real kernel memory limits)."""
    from agentgym.fidelity import run_study

    results, summary = run_study()
    typer.echo(f"{'fault':30s} {'sim ready':10s} {'real ready':11s} {'reason':24s} agrees")
    for r in results:
        typer.echo(f"{r.fault:30s} {str(r.simulated_ready):10s} "
                   f"{str(r.real_ready):11s} {r.real_reason:24s} {r.agrees}")
    typer.echo(f"\nagreement: {summary['agreements']}/{summary['cases']} "
               f"= {summary['agreement_rate']:.0%}")
    path = out or (cfg.ARTIFACTS_DIR / "fidelity" / "report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary,
                                "cases": [r.__dict__ for r in results]}, indent=2))
    typer.echo(f"wrote {path}")
    if summary["agreement_rate"] < 1.0:
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor() -> None:
    """Check that the environment can actually run."""
    import shutil

    ok = True
    for binary in ("etcd", "kube-apiserver", "kube-controller-manager",
                   "kube-scheduler", "kubectl"):
        present = (cfg.BIN_DIR / binary).exists()
        ok &= present
        typer.echo(f"  {'OK ' if present else 'MISSING'}  {cfg.BIN_DIR / binary}")
    typer.echo(f"  runtime dir: {cfg.RUNTIME_DIR}")
    typer.echo(f"  data dir:    {cfg.DATA_DIR}")
    typer.echo(f"  openssl:     {shutil.which('openssl')}")
    if not ok:
        typer.echo("\nrun scripts/fetch_k8s_binaries.sh")
        raise typer.Exit(code=1)
    typer.echo("\nstarting a control plane to verify...")
    from agentgym.cluster.controlplane import ControlPlane

    with ControlPlane(root=cfg.RUNTIME_DIR / "doctor", offset=90) as cp:
        typer.echo(f"  healthz: {cp.kubectl('get', '--raw', '/healthz').stdout}")
    typer.echo("environment OK")


if __name__ == "__main__":
    app()
