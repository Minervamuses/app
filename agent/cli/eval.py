"""Evaluation CLI for the agent package.

Usage:
    python -m agent.cli.eval --suite behavior
    python -m agent.cli.eval --claim c1 --split dev --allow-skips
    python -m agent.cli.eval --suite e2e --generate 15
    python -m agent.cli.eval --suite thinking
    python -m agent.cli.eval --all --generate 5
    python -m agent.cli.eval --suite e2e --cases eval/e2e_cases.json
    python -m agent.cli.eval --all --generate 5 --output eval/
    python -m agent.cli.eval -h
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent.config import AgentConfig
from agent.evaluation.base import EvalResult
from agent.evaluation.behavior import BehaviorEvaluator
from agent.evaluation.claims import (
    C1RoutingEvaluator,
    C2RetrievalEvaluator,
    C3ReviewerEvaluator,
    C3SessionEvaluator,
    C3ValidatorEvaluator,
)
from agent.evaluation.datasets import dataset_file_path, dataset_hash, load_claim_dataset
from agent.evaluation.endtoend import EndToEndEvaluator
from agent.evaluation.ledger import append_result
from agent.evaluation.thinking import ThinkingReviewerEvaluator
from agent.mcp import load_mcp_tools

SUITE_NAMES = ("behavior", "e2e", "thinking")
CLAIM_NAMES = ("c1", "c2", "c3", "c4")
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


def _run_suite(
    suite: str,
    config: AgentConfig,
    generate_n: int | None,
    cases_path: str | None,
    output_dir: str | None,
    extra_tools: list | None,
) -> EvalResult:
    """Run a single evaluation suite and return the result."""

    if suite == "behavior":
        evaluator = BehaviorEvaluator(config, extra_tools=extra_tools)
    elif suite == "e2e":
        evaluator = EndToEndEvaluator(config, extra_tools=extra_tools)
    elif suite == "thinking":
        evaluator = ThinkingReviewerEvaluator(config)
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
    elif suite in {"behavior", "thinking"}:
        # Behavior and thinking reviewer suites have built-in cases.
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
                "metadata": result.metadata,
            }, f, ensure_ascii=False, indent=2)
        print(f"[{suite}] Results → {results_path}")

    return result


def _output_root(output_dir: str | None) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    return path.parent if path.name == "eval" else path


def _run_claim(
    claim: str,
    config: AgentConfig,
    split: str,
    output_dir: str | None,
    extra_tools: list | None,
    allow_skips: bool,
) -> EvalResult:
    """Run a single new claim evaluator and append to the run ledger."""
    if claim == "c3":
        cases = load_claim_dataset(claim, split)
        print(f"[{claim}] Loaded {len(cases)} {split} cases")
        print(f"[{claim}] Evaluating C3a/C3b/C3c...")
        result = _run_c3_claim(config, cases)
    elif claim == "c1":
        evaluator = C1RoutingEvaluator(
            config,
            extra_tools=extra_tools,
            allow_skips=allow_skips,
        )
        cases = load_claim_dataset(claim, split)
        print(f"[{claim}] Loaded {len(cases)} {split} cases")
        print(f"[{claim}] Evaluating...")
        result = evaluator.evaluate(cases)
    elif claim == "c2":
        evaluator = C2RetrievalEvaluator(config)
        cases = load_claim_dataset(claim, split)
        print(f"[{claim}] Loaded {len(cases)} {split} cases")
        print(f"[{claim}] Evaluating...")
        result = evaluator.evaluate(cases)
    else:
        raise NotImplementedError(f"Claim '{claim}' is not implemented yet")

    root = _output_root(output_dir)
    if root is not None:
        dataset_path = dataset_file_path(claim, split)
        row = append_result(
            claim,
            result,
            root=root,
            include_details=(split != "test" or allow_skips),
            extra_metadata={
                "split": split,
                "dataset_id": f"{claim}/{split}",
                "dataset_hash": dataset_hash(dataset_path),
            },
        )
        print(f"[{claim}] Ledger row → eval/runs/{claim}.jsonl ({row['run_id']})")

    return result


def _run_c3_claim(config: AgentConfig, cases: list) -> EvalResult:
    """Run the three independent C3 sub-evaluators and prefix their metrics."""
    sub_evaluators = [
        ("validator", C3ValidatorEvaluator(config)),
        ("reviewer", C3ReviewerEvaluator(config)),
        ("session", C3SessionEvaluator(config)),
    ]
    sub_results = []
    scores = {}
    details = []
    for prefix, evaluator in sub_evaluators:
        result = evaluator.evaluate([
            case for case in cases
            if case.raw.get("task") == result_task_name(prefix)
        ])
        sub_results.append(result)
        scores.update({
            f"{prefix}.{metric}": value
            for metric, value in result.scores.items()
        })
        details.append({
            "subclaim": prefix,
            "name": result.name,
            "total": result.total,
            "scores": result.scores,
            "details": result.details,
        })
    return EvalResult(
        name="C3",
        total=sum(result.total for result in sub_results),
        scores=scores,
        details=details,
        metadata={"claim": "c3", "subclaims": [result.name for result in sub_results]},
    )


def result_task_name(prefix: str) -> str:
    return {
        "validator": "validator",
        "reviewer": "reviewer",
        "session": "session",
    }[prefix]


def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation suites: behavior and end-to-end."
    )
    parser.add_argument(
        "--suite", choices=SUITE_NAMES,
        help="Which evaluation suite to run.",
    )
    parser.add_argument(
        "--claim", choices=CLAIM_NAMES,
        help="Which new claim evaluator to run. Currently only c1 is implemented.",
    )
    parser.add_argument(
        "--split", choices=("dev", "test"), default="dev",
        help="Dataset split for --claim (default: dev).",
    )
    parser.add_argument(
        "--allow-skips", action="store_true",
        help="Allow claim cases whose required tools are unavailable. "
             "Official runs should omit this flag so skips fail fast.",
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
    parser.add_argument(
        "--no-mcp", action="store_true",
        help="Disable MCP tool loading even if configured via environment.",
    )
    args = parser.parse_args()

    if not args.suite and not args.claim and not args.all:
        parser.error("Specify --suite, --claim, or --all")

    config = AgentConfig()
    extra_tools = []
    if args.no_mcp:
        print("[eval] MCP loading disabled via --no-mcp")
    else:
        extra_tools = asyncio.run(load_mcp_tools())
        print(f"[eval] Loaded {len(extra_tools)} MCP tool(s)")

    if args.claim:
        print(f"\n{'=' * 50}")
        print(f"  Running claim: {args.claim}/{args.split}")
        print(f"{'=' * 50}")
        result = _run_claim(
            claim=args.claim,
            config=config,
            split=args.split,
            output_dir=args.output,
            extra_tools=extra_tools,
            allow_skips=args.allow_skips,
        )
        print(f"\n{'=' * 50}")
        print("  Summary")
        print(f"{'=' * 50}\n")
        print(result.summary())
        return

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
            extra_tools=extra_tools,
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
