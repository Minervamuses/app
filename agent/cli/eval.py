"""Evaluation CLI for the agent package.

Usage:
    python -m agent.cli.eval --suite behavior
    python -m agent.cli.eval --suite e2e --generate 15
    python -m agent.cli.eval --all --generate 5
    python -m agent.cli.eval --suite e2e --cases eval/e2e_cases.json
    python -m agent.cli.eval --all --generate 5 --output eval/
    python -m agent.cli.eval -h
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from agent.config import AgentConfig
from agent.evaluation.base import EvalResult
from agent.evaluation.behavior import BehaviorEvaluator
from agent.evaluation.endtoend import EndToEndEvaluator

SUITE_NAMES = ("behavior", "e2e")


def _run_suite(
    suite: str,
    config: AgentConfig,
    generate_n: int | None,
    cases_path: str | None,
    output_dir: str | None,
) -> EvalResult:
    """Run a single evaluation suite and return the result."""

    if suite == "behavior":
        evaluator = BehaviorEvaluator(config)
    elif suite == "e2e":
        evaluator = EndToEndEvaluator(config)
    else:
        raise ValueError(f"Unknown suite: {suite}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Load or generate cases
    if cases_path:
        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)
        print(f"[{suite}] Loaded {len(cases)} cases from {cases_path}")
    elif generate_n is not None:
        save_path = None
        if output_dir:
            save_path = str(Path(output_dir) / f"{suite}_cases_{timestamp}.json")
        cases = evaluator.generate(n=generate_n, output_path=save_path)
        print(f"[{suite}] Generated {len(cases)} cases" + (f" → {save_path}" if save_path else ""))
    elif suite == "behavior":
        # Behavior has built-in cases
        cases = evaluator.generate()
        print(f"[{suite}] Using {len(cases)} built-in cases")
    else:
        raise ValueError(f"Suite '{suite}' requires --generate N or --cases PATH")

    # Evaluate
    print(f"[{suite}] Evaluating...")
    result = evaluator.evaluate(cases)

    # Save results
    if output_dir:
        results_path = Path(output_dir) / f"{suite}_results_{timestamp}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": result.name,
                "timestamp": timestamp,
                "total": result.total,
                "scores": result.scores,
                "details": result.details,
            }, f, ensure_ascii=False, indent=2)
        print(f"[{suite}] Results → {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation suites: behavior and end-to-end."
    )
    parser.add_argument(
        "--suite", choices=SUITE_NAMES,
        help="Which evaluation suite to run.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both suites.",
    )
    parser.add_argument(
        "--generate", type=int, metavar="N",
        help="Generate N test cases (for e2e). Behavior always uses built-in cases.",
    )
    parser.add_argument(
        "--cases", type=str, metavar="PATH",
        help="Load test cases from a JSON file instead of generating.",
    )
    parser.add_argument(
        "--output", type=str, metavar="DIR", default="eval",
        help="Save cases and results to this directory (default: eval/). "
             "Filenames include a timestamp so old runs are preserved.",
    )
    args = parser.parse_args()

    if not args.suite and not args.all:
        parser.error("Specify --suite or --all")

    config = AgentConfig()
    suites = list(SUITE_NAMES) if args.all else [args.suite]
    results: list[EvalResult] = []

    for suite in suites:
        print(f"\n{'=' * 50}")
        print(f"  Running: {suite}")
        print(f"{'=' * 50}")

        result = _run_suite(
            suite=suite,
            config=config,
            generate_n=args.generate,
            cases_path=args.cases if not args.all else None,
            output_dir=args.output,
        )
        results.append(result)

    # Print summary
    print(f"\n{'=' * 50}")
    print("  Summary")
    print(f"{'=' * 50}\n")
    for result in results:
        print(result.summary())
        print()


if __name__ == "__main__":
    main()
