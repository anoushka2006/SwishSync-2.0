"""Divergence eval for trusted_flight_points Phase 0.

Reads finalized shot JSON exported by write_finalized_shots_json() and computes
divergence metrics between trusted_flight (Phase 0 selection) and the existing
fit_points path (fit_diagnostics.fit_point_count).

Phase 2 gate: trusted_flight_points may only become the fit input when
CORE divergence (A/C/P/R/S) is zero across the benchmark suite defined in
docs/benchmark_suite.md.

Usage:
    python scripts/eval_trusted_flight.py <shots.json>
    python scripts/eval_trusted_flight.py outputs/shots_finalized.json --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_shots(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON array, got {type(data)}")


def _analyze_shot(shot: dict) -> dict:
    tf = shot.get("trusted_flight")
    diag = shot.get("fit_diagnostics")

    trusted_count = tf["trusted_count"] if tf is not None else None
    fit_count = diag.get("fit_point_count") if diag is not None else None
    candidate_count = len(shot.get("candidate_points", []))

    # Exclusion breakdown
    excl_by_reason: dict[str, int] = {}
    if tf is not None:
        for e in tf.get("excluded", []):
            reason = e.get("reason", "unknown")
            excl_by_reason[reason] = excl_by_reason.get(reason, 0) + 1

    diverges = trusted_count != fit_count if (trusted_count is not None and fit_count is not None) else None

    return {
        "start_frame": shot.get("start_frame"),
        "candidate_count": candidate_count,
        "trusted_count": trusted_count,
        "fit_count": fit_count,
        "diverges": diverges,
        "excl_gap_predicted": excl_by_reason.get("gap_predicted", 0),
        "excl_post_cluster": excl_by_reason.get("post_cluster", 0),
        "excl_floor_bounce": excl_by_reason.get("floor_bounce", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted flight divergence eval")
    parser.add_argument("shots_json", type=Path, help="Path to finalized shots JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-shot details")
    args = parser.parse_args()

    shots = _load_shots(args.shots_json)
    if not shots:
        print("No shots found.")
        return

    results = [_analyze_shot(s) for s in shots]
    total = len(results)
    with_trusted = sum(1 for r in results if r["trusted_count"] is not None)
    divergent = sum(1 for r in results if r["diverges"] is True)

    print(f"Shots: {total}  |  with trusted_flight: {with_trusted}  |  divergent: {divergent}")

    if with_trusted > 0:
        excl_gp = sum(r["excl_gap_predicted"] for r in results)
        excl_pc = sum(r["excl_post_cluster"] for r in results)
        excl_fb = sum(r["excl_floor_bounce"] for r in results)
        print(f"Exclusions — gap_predicted: {excl_gp}  post_cluster: {excl_pc}  floor_bounce: {excl_fb}")

    if divergent > 0:
        print(f"\nDIVERGENCE DETECTED — {divergent}/{with_trusted} shots differ (Phase 2 gate: must be 0)")

    if args.verbose:
        print()
        for r in results:
            flag = "DIFF" if r["diverges"] else "ok  "
            print(
                f"  [{flag}] frame={r['start_frame']:>4}  cand={r['candidate_count']:>3}"
                f"  trusted={r['trusted_count']}  fit={r['fit_count']}"
                f"  excl=gp:{r['excl_gap_predicted']} pc:{r['excl_post_cluster']} fb:{r['excl_floor_bounce']}"
            )

    sys.exit(1 if divergent > 0 else 0)


if __name__ == "__main__":
    main()
