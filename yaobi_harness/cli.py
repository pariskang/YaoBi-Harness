"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .graph import YaobiGraphRunner
from .llm.factory import build_client, describe_client
from .llm.base import LLMError
from .render import render
from .state import Budget, ClinicalRunState
from .tools import DeidentificationKeyError, ExpertCaseStore, ToolRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yaobi-harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run one clinical reasoning pass")
    run.add_argument("--complaint", required=True)
    run.add_argument("--role", choices=["patient", "physician", "researcher"], default="physician")
    run.add_argument("--xlsx", help="local authorized Excel; never committed or returned raw")
    run.add_argument("--facts", help="JSON object of known facts (age, special_population, medications, ...)")
    run.add_argument("--facts-file", help="path to a JSON file with the same content as --facts")
    run.add_argument("--allow-prescription", action="store_true")
    run.add_argument("--checkpoint-dir")
    run.add_argument("--llm-provider", choices=["azure", "poe", "minimax", "litellm", "none"],
                     help="overrides YAOBI_LLM_PROVIDER")
    run.add_argument("--llm-model", help="deployment/bot/model name for the chosen provider")
    run.add_argument("--max-tool-calls", type=int, default=24)
    run.add_argument("--max-llm-calls", type=int, default=12)
    run.add_argument("--max-loops", type=int, default=3)
    run.add_argument("--debug-state", action="store_true", help="print the full internal state instead of the role view")

    resume = sub.add_parser("resume", help="resume a checkpointed run")
    resume.add_argument("checkpoint")
    resume.add_argument("--facts", help="JSON object merged into the resumed state before continuing")
    resume.add_argument("--allow-prescription", action="store_true")
    resume.add_argument("--checkpoint-dir")
    resume.add_argument("--debug-state", action="store_true")

    inspect = sub.add_parser("inspect-xlsx", help="summarise a local authorized Excel without returning raw rows")
    inspect.add_argument("path")

    sub.add_parser("llm-check", help="show which LLM provider is configured")
    return parser


def _load_facts(inline: str | None, path: str | None) -> dict:
    facts: dict = {}
    if path:
        with open(path, encoding="utf-8") as handle:
            facts.update(json.load(handle))
    if inline:
        facts.update(json.loads(inline))
    return facts


def _make_llm(provider: str | None, model: str | None):
    overrides = {"model": model} if model else {}
    return build_client(provider, **overrides)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "llm-check":
        try:
            client = build_client()
        except LLMError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(describe_client(client), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "inspect-xlsx":
        try:
            store = ExpertCaseStore(args.path)
        except DeidentificationKeyError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        herb_count = sum(len(r.get("herbs", [])) for r in store.records)
        print(json.dumps(
            {
                "records_loaded": len(store.records),
                "parsed_herb_items": herb_count,
                "privacy": "direct identifiers removed; retrieval returns structured deidentified fields only",
                "positioning": ["single_expert_memory", "followup_trajectory", "similar_case_retrieval",
                                "stratified_dose_distribution"],
                "not_causal_trial": True,
            },
            ensure_ascii=False, indent=2,
        ))
        return 0

    if args.cmd == "resume":
        state = YaobiGraphRunner.load_checkpoint(args.checkpoint)
        # Facts supplied on resume are the whole point of resuming (the answers
        # to the run's open questions), so they must be merged *before* the run.
        if args.facts:
            state.facts.update(json.loads(args.facts))
        runner = YaobiGraphRunner(checkpoint_dir=args.checkpoint_dir)
        out = runner.run(state, allow_prescription=args.allow_prescription, resumed=True)
        print(json.dumps(render(out, debug=args.debug_state), ensure_ascii=False, indent=2))
        return 0

    try:
        llm = _make_llm(args.llm_provider, args.llm_model)
    except LLMError as exc:
        print(json.dumps({"error": f"LLM 配置错误: {exc}"}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    try:
        tools = ToolRegistry(args.xlsx) if args.xlsx else ToolRegistry()
    except DeidentificationKeyError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    state = ClinicalRunState(complaint=args.complaint, role=args.role)
    state.facts.update(_load_facts(args.facts, args.facts_file))
    state.budget = Budget(
        max_loops=args.max_loops,
        max_tool_calls=args.max_tool_calls,
        max_llm_calls=args.max_llm_calls,
    )
    runner = YaobiGraphRunner(tools, checkpoint_dir=args.checkpoint_dir, llm=llm)
    out = runner.run(state, allow_prescription=args.allow_prescription)
    print(json.dumps(render(out, debug=args.debug_state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
