"""Generate standalone HTML report and manifest for budget distribution component."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from random_sampling.budget_distribution import (
    AllocationConfig,
    BudgetDeductions,
    FairnessState,
    SessionDemand,
    SimulationScenario,
    build_batch_plan,
    build_eligible_frame,
    calculate_batch_budget,
    resolve_batch_window,
    simulate_policies,
    stable_sha256_hex,
)


def _hash_files(repo: Path, relative_paths: tuple[str, ...]) -> str:
    payloads: list[str] = []
    for rel in relative_paths:
        path = repo / rel
        payloads.append(f"{rel}:{path.read_text(encoding='utf-8')}")
    return stable_sha256_hex(*payloads)


def _demo_sessions() -> list[SessionDemand]:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sessions: list[SessionDemand] = []
    for index in range(16):
        tenant = "tenant-a" if index < 10 else "tenant-b"
        agent = "agent-1" if index % 2 == 0 else "agent-2"
        sessions.append(
            SessionDemand(
                tenant_id=tenant,
                agent_id=agent,
                session_id=f"s-{index:03d}",
                completed_at=now - timedelta(minutes=50 - index * 2),
                ingested_at=now - timedelta(minutes=48 - index * 2),
                estimated_input_tokens=350 + index * 10,
                expected_output_tokens=120,
            )
        )
    return sessions


def _scenario_sessions(base: datetime) -> tuple[SimulationScenario, ...]:
    dominant = tuple(
        SessionDemand(
            tenant_id="tenant-dominant" if i < 8 else "tenant-small",
            agent_id="agent-heavy" if i < 8 else ("agent-small-1" if i % 2 == 0 else "agent-small-2"),
            session_id=f"dom-{i:02d}",
            completed_at=base - timedelta(minutes=55 - i * 3),
            ingested_at=base - timedelta(minutes=53 - i * 3),
            estimated_input_tokens=420 + i * 65,
            expected_output_tokens=120,
        )
        for i in range(12)
    )
    variable_costs = tuple(
        SessionDemand(
            tenant_id="tenant-v",
            agent_id="agent-cost-a" if i % 2 == 0 else "agent-cost-b",
            session_id=f"var-{i:02d}",
            completed_at=base - timedelta(minutes=40 - i * 2),
            ingested_at=base - timedelta(minutes=39 - i * 2),
            estimated_input_tokens=(220, 300, 900, 750, 260, 380, 1100, 620)[i],
            expected_output_tokens=(80, 90, 130, 120, 70, 80, 150, 110)[i],
        )
        for i in range(8)
    )
    low_demand = tuple(
        SessionDemand(
            tenant_id="tenant-low",
            agent_id="agent-low",
            session_id=f"low-{i:02d}",
            completed_at=base - timedelta(minutes=18 - i * 3),
            ingested_at=base - timedelta(minutes=17 - i * 3),
            estimated_input_tokens=120,
            expected_output_tokens=30,
        )
        for i in range(4)
    )
    late_retry = (
        SessionDemand(
            tenant_id="tenant-late",
            agent_id="agent-a",
            session_id="lr-01",
            completed_at=base - timedelta(minutes=70),
            ingested_at=base - timedelta(minutes=10),
            estimated_input_tokens=400,
            expected_output_tokens=80,
        ),
        SessionDemand(
            tenant_id="tenant-late",
            agent_id="agent-a",
            session_id="lr-01",
            session_version="v2",
            completed_at=base - timedelta(minutes=20),
            ingested_at=base - timedelta(minutes=8),
            estimated_input_tokens=420,
            expected_output_tokens=90,
        ),
        SessionDemand(
            tenant_id="tenant-late",
            agent_id="agent-b",
            session_id="lr-02",
            completed_at=base - timedelta(minutes=19),
            ingested_at=base - timedelta(minutes=9),
            estimated_input_tokens=500,
            expected_output_tokens=110,
        ),
    )

    return (
        SimulationScenario(
            name="dominant-tenant-agent-contention",
            sessions_by_batch=(dominant[:6], dominant[6:]),
            budget_tokens_per_batch=2_400,
        ),
        SimulationScenario(
            name="variable-cost-packing-contention",
            sessions_by_batch=(variable_costs[:4], variable_costs[4:]),
            budget_tokens_per_batch=1_700,
        ),
        SimulationScenario(
            name="delayed-missed-catchup-shape",
            sessions_by_batch=(tuple(_demo_sessions()[:8]), tuple(_demo_sessions()[8:])),
            budget_tokens_per_batch=8_000,
        ),
        SimulationScenario(
            name="low-demand-expiry",
            sessions_by_batch=(low_demand,),
            budget_tokens_per_batch=12_000,
        ),
        SimulationScenario(
            name="late-and-retry-conceptual",
            sessions_by_batch=(late_retry,),
            budget_tokens_per_batch=4_000,
        ),
    )


def _build_report_html(plan_summary: dict, simulation_rows: list[dict]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['policy']}</td>"
        f"<td>{row['scenario']}</td>"
        f"<td>{row['utilization']:.3f}</td>"
        f"<td>{row['coverage']:.3f}</td>"
        f"<td>{row['selected']}</td>"
        f"<td>{row['starvation_proxy']:.3f}</td>"
        f"<td>{row['fairness_jain']:.3f}</td>"
        f"<td>{row['slack_tokens']}</td>"
        f"<td>{row['tpm_compliance_rate']:.3f}</td>"
        f"<td>{'yes' if row['replay_match'] else 'no'}</td>"
        "</tr>"
        for row in simulation_rows
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Random Sampling Budget Distribution Report</title>
<style>
:root {{
  --bg: #f7f4ea;
  --ink: #1f2d3d;
  --accent: #a44a3f;
  --soft: #e7dfc9;
}}
body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; background: var(--bg); color: var(--ink); }}
main {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}
h1 {{ letter-spacing: 0.03em; }}
section {{ background: rgba(255,255,255,0.65); border: 1px solid var(--soft); padding: 1rem 1.2rem; margin: 1rem 0; border-radius: 10px; }}
code {{ background: #efe8d2; padding: 0.1rem 0.3rem; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
th, td {{ border: 1px solid #c9bea1; padding: 0.35rem 0.45rem; text-align: left; }}
th {{ background: #efe6cf; }}
.notice {{ border-left: 4px solid var(--accent); padding-left: 0.75rem; }}
</style>
</head>
<body>
<main>
<h1>Deterministic Random Batch Token-Budget Distribution</h1>
<section>
<h2>Executive Summary</h2>
<p>Production-oriented deterministic planning converts elapsed successful watermark time to a token budget, allocates tenant/agent grants with floor plus capped-Hamilton surplus, then applies stable-hash random packing of whole sessions under a rolling TPM governor.</p>
</section>
<section>
<h2>Window and Checkpoint</h2>
<p>Canonical window: <code>[{plan_summary['window_start']}, {plan_summary['window_end']})</code>; source scan start <code>{plan_summary['scan_start']}</code>; bootstrap used <code>{plan_summary['bootstrap_used']}</code>.</p>
<p>Batch ID <code>{plan_summary['batch_id']}</code>; seed <code>{plan_summary['seed']}</code>; frame hash <code>{plan_summary['frame_hash']}</code>; config hash <code>{plan_summary['config_hash']}</code>.</p>
</section>
<section>
<h2>Allocation and Selection</h2>
<p>Effective budget <code>{plan_summary['effective_budget_tokens']}</code>; planned usage <code>{plan_summary['planned_usage_tokens']}</code>; slack <code>{plan_summary['slack_tokens']}</code>; selected sessions <code>{plan_summary['selected_count']}</code>; zero allocations <code>{plan_summary['zero_allocations']}</code>; redistribution rounds <code>{plan_summary['reallocation_rounds']}</code>.</p>
<p class=\"notice\">Selection policy is <code>token_constrained_random</code>; inclusion probabilities are not design-calibrated under variable-cost packing.</p>
</section>
<section>
<h2>Pacing and Examples</h2>
<p>TPM rate uses rolling <code>20,000 tokens/min</code>; reservations occur before dispatch; actual usage reconciles deltas post-completion and compliance is measured from paced schedules. The worked examples include 5-minute and 60-minute budgets, catch-up clamp, late-arrival lookback dedup, and unserviceable-session handling.</p>
</section>
<section>
<h2>Policy Comparison</h2>
<p>Fairness metric is Jain index over cumulative served token distribution across active agents, and utilization is selected tokens divided by total batch budget.</p>
<table>
<thead>
<tr><th>Policy</th><th>Scenario</th><th>Utilization</th><th>Coverage</th><th>Selected</th><th>Starvation</th><th>Jain</th><th>Slack</th><th>TPM</th><th>Replay</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</section>
<section>
<h2>Recommendation and Caveats</h2>
<p>Proposed production default is hierarchical because it preserves deterministic tenant/agent isolation and fairness deficit carry-over; this recommendation should be adopted only when constrained-scenario results show material differentiation and the tradeoff with token utilization is acceptable for the target workload.</p>
<p>Policy tradeoff framing: FCFS/global random maximize globally ordered packing behavior, equal/proportional baselines provide auditable grant discipline, and hierarchical adds fairness-memory behavior across batches at the cost of additional policy structure.</p>
<p>Reference JSON checkpoint/lease store is a single-process adapter for tests/examples. File writes are atomic via temporary-file replacement, but this is not a distributed lease implementation.</p>
</section>
<section>
<h2>Reproducibility</h2>
<p>All outputs are deterministic from frozen frame membership, seed, hashes, and stable sort keys. Replay identity is validated through simulation replay checks and persisted plan metadata.</p>
</section>
</main>
</body>
</html>
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "outputs_budget_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(
        previous_successful_watermark=cutoff - timedelta(minutes=60),
        cutoff=cutoff,
        lookback=timedelta(minutes=15),
        max_catchup_minutes=60,
    )
    budget = calculate_batch_budget(
        window=window,
        deductions=BudgetDeductions(
            safety_tokens=5_000,
            output_tokens=10_000,
            retry_tokens=5_000,
        ),
    )
    sessions = _demo_sessions()
    frame = build_eligible_frame(
        source_sessions=sessions,
        window=window,
        processed_session_keys=set(),
    )
    plan = build_batch_plan(
        pipeline_id="random-budget-distribution",
        batch_id="batch-20260101T120000Z",
        seed="seed-v1",
        window=window,
        budget=budget,
        frame=frame,
        fairness_state=FairnessState(),
        allocation_config=AllocationConfig(tenant_floor_tokens=500, agent_floor_tokens=250),
    )

    scenarios = _scenario_sessions(cutoff)
    simulation_seed = "report-seed"
    sim = simulate_policies(scenarios, seed=simulation_seed)

    rows = [
        {
            "policy": row.policy,
            "scenario": row.scenario,
            "utilization": row.utilization,
            "coverage": row.coverage,
            "selected": row.selected,
            "starvation_proxy": row.starvation_proxy,
            "fairness_jain": row.fairness_jain,
            "slack_tokens": row.slack_tokens,
            "tpm_compliance_rate": row.tpm_compliance_rate,
            "replay_match": row.replay_match,
        }
        for row in sim.metrics
    ]

    html = _build_report_html(
        {
            "window_start": plan.window.window_start.isoformat(),
            "window_end": plan.window.window_end.isoformat(),
            "scan_start": plan.window.source_scan_start.isoformat(),
            "bootstrap_used": plan.window.bootstrap_used,
            "batch_id": plan.batch_id,
            "seed": plan.seed,
            "frame_hash": plan.frame_hash,
            "config_hash": plan.config_hash,
            "effective_budget_tokens": plan.budget.effective_tokens,
            "planned_usage_tokens": plan.planned_usage_tokens,
            "slack_tokens": plan.slack_tokens,
            "selected_count": len(plan.selection.selected),
            "zero_allocations": plan.zero_allocations,
            "reallocation_rounds": plan.reallocation_rounds,
        },
        rows,
    )

    report_path = out_dir / "random-sampling-batch-budget-report.html"
    report_path.write_text(html, encoding="utf-8")

    input_material = [
        f"{s.tenant_id}|{s.agent_id}|{s.session_id}|{s.session_version}|{s.completed_at.isoformat()}|{s.total_cost_tokens}"
        for scenario in scenarios
        for batch in scenario.sessions_by_batch
        for s in batch
    ]
    input_hash = stable_sha256_hex(*input_material) if input_material else stable_sha256_hex("empty")
    config_hash = stable_sha256_hex(
        plan.config_hash,
        f"simulation_seed:{simulation_seed}",
        f"scenarios:{len(scenarios)}",
    )
    code_hash = _hash_files(
        repo,
        (
            "random_sampling/budget_distribution/batch.py",
            "random_sampling/budget_distribution/selection.py",
            "random_sampling/budget_distribution/pacing.py",
            "random_sampling/budget_distribution/checkpoint.py",
            "random_sampling/budget_distribution/simulation.py",
            "scripts/build_budget_distribution_report.py",
        ),
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": simulation_seed,
        "simulation_command": "py -3.11 scripts/build_budget_distribution_report.py",
        "input_hash": input_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "artifacts": [
            {
                "path": str(report_path.relative_to(repo)).replace('\\', '/'),
                "kind": "html",
                "description": "Standalone production-oriented budget distribution report",
            }
        ],
        "pdf_generated": False,
    }
    manifest_path = out_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
