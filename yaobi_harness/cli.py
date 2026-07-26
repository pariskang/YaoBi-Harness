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
    run.add_argument("--max-llm-calls", type=int, default=40)
    run.add_argument("--max-loops", type=int, default=3)
    run.add_argument("--knowledge-store", help="path to the licensed knowledge store built by `knowledge build`")
    run.add_argument("--skill-manifest", help="skill manifest to use (e.g. one merged with a generated expert skill)")
    run.add_argument("--debug-state", action="store_true", help="print the full internal state instead of the role view")

    resume = sub.add_parser("resume", help="resume a checkpointed run")
    resume.add_argument("checkpoint")
    resume.add_argument("--facts", help="JSON object merged into the resumed state before continuing")
    resume.add_argument("--allow-prescription", action="store_true")
    resume.add_argument("--checkpoint-dir")
    resume.add_argument("--debug-state", action="store_true")

    chat = sub.add_parser("chat", help="multi-turn clinical dialogue in the terminal")
    chat.add_argument("--role", choices=["patient", "physician", "researcher"], default="patient")
    chat.add_argument("--xlsx")
    chat.add_argument("--knowledge-store")
    chat.add_argument("--skill-manifest")
    chat.add_argument("--allow-prescription", action="store_true")
    chat.add_argument("--llm-provider", choices=["azure", "poe", "minimax", "litellm", "none"])
    chat.add_argument("--llm-model")
    chat.add_argument("--transcript", help="write the full transcript and state here on exit")
    chat.add_argument("--message", action="append",
                      help="send this message and exit; repeat for a scripted conversation")

    inspect = sub.add_parser("inspect-xlsx", help="summarise a local authorized Excel without returning raw rows")
    inspect.add_argument("path")

    sub.add_parser("llm-check", help="show which LLM provider is configured")

    ui = sub.add_parser("ui", help="serve the local operator console")
    ui.add_argument("--host", default="127.0.0.1", help="bind address; keep on localhost unless fronted by an authenticated proxy")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--knowledge-store")
    ui.add_argument("--xlsx")
    ui.add_argument("--checkpoint-dir")
    ui.add_argument("--skill-manifest")
    ui.add_argument("--llm-provider", choices=["azure", "poe", "minimax", "litellm", "none"])
    ui.add_argument("--llm-model")
    ui.add_argument("--open", action="store_true", help="open a browser window")
    ui.add_argument("--public", action="store_true",
                    help="expose via an ngrok tunnel; forces token auth (demo/review only)")
    ui.add_argument("--access-token", help="fixed access token; one is generated when --public is set")
    ui.add_argument("--ngrok-authtoken", help="overrides NGROK_AUTHTOKEN")
    ui.add_argument("--ngrok-region")

    skill = sub.add_parser("skill", help="inspect skills and generate one from the expert corpus")
    ssub = skill.add_subparsers(dest="skcmd", required=True)

    slist = ssub.add_parser("list", help="show every skill, its tools and whether it may run autonomously")
    slist.add_argument("--skill-manifest")

    sshow = ssub.add_parser("show", help="print one skill's full instructions")
    sshow.add_argument("skill_id")
    sshow.add_argument("--skill-manifest")

    sbuild = ssub.add_parser("build-expert", help="mine the authorized case corpus into a loadable skill")
    sbuild.add_argument("--xlsx", help="authorized Excel; omit to use --records-json")
    sbuild.add_argument("--records-json", help="already de-identified records as JSON, for testing")
    sbuild.add_argument("--out", default="skills/expert_practice.yaml")
    sbuild.add_argument("--merge", action="store_true", help="also write the skill into the main manifest")
    sbuild.add_argument("--skill-manifest", help="manifest to merge into (defaults to the packaged one)")
    sbuild.add_argument("--min-support", type=int, default=2,
                        help="withhold aggregate values seen fewer times than this")
    sbuild.add_argument("--source-label", default="授权专家病例库")

    knowledge = sub.add_parser("knowledge", help="build and inspect the licensed knowledge store")
    ksub = knowledge.add_subparsers(dest="kcmd", required=True)

    ksub.add_parser("sources", help="list every source and whether this deployment may use it")

    kbuild = ksub.add_parser("build", help="fetch everything this deployment is licensed to fetch")
    kbuild.add_argument("--store", required=True)
    kbuild.add_argument("--cache-dir")
    kbuild.add_argument("--include", nargs="*", help="limit to these source ids")
    kbuild.add_argument("--ingredients", nargs="*", help="override the default orthopaedic ingredient list")
    kbuild.add_argument("--nice-api-key", help="NICE syndication key (requires a recorded attestation)")

    kfile = ksub.add_parser("ingest-file", help="load an operator-supplied export under licence")
    kfile.add_argument("--store", required=True)
    kfile.add_argument("--source", required=True, help="source id, e.g. chp_2025 / ddinter / cma_guidelines")
    kfile.add_argument("--kind", required=True, choices=["dose_ranges", "interactions", "guidelines", "citations"])
    kfile.add_argument("--path", required=True)
    kfile.add_argument("--version", default="")

    kstats = ksub.add_parser("stats", help="show store contents and the active licence policy")
    kstats.add_argument("--store", required=True)

    ksub.add_parser("rules", help="show the built-in orthopaedic interaction rule pack")

    kcheck = ksub.add_parser("check-interactions", help="screen a medication list without running a full case")
    kcheck.add_argument("--medications", nargs="+", required=True)
    kcheck.add_argument("--conditions", nargs="*", default=[])
    kcheck.add_argument("--store")

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


def _emit(payload) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _chat_command(args) -> int:
    """Interactive dialogue, or a scripted one via repeated --message."""
    from .conversation import ConversationSession

    try:
        llm = _make_llm(args.llm_provider, args.llm_model)
        knowledge = _open_knowledge(args.knowledge_store)
        tools = ToolRegistry(args.xlsx or None, knowledge=knowledge)
    except (LLMError, DeidentificationKeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    runner = YaobiGraphRunner(tools, skill_manifest=args.skill_manifest, llm=llm)
    session = ConversationSession(role=args.role, runner=runner,
                                  allow_prescription=args.allow_prescription)

    def show(reply) -> None:
        flag = " ⚠已升级为急症" if reply.escalated else ""
        print(f"\n🤖 [{reply.release_status}/{reply.risk_mode}{flag}]")
        print("   " + reply.message.replace("\n", "\n   "))
        if reply.extracted:
            print(f"   ↳ 本轮获得: {json.dumps(reply.extracted, ensure_ascii=False)}")

    if args.message:
        for message in args.message:
            print(f"\n👤 {message}")
            show(session.send(message))
    else:
        print(f"腰痹智能体对话（角色={args.role}，模型={describe_client(llm)['provider']}）")
        print("直接输入症状开始；输入 /quit 结束，/facts 查看已知信息。\n")
        while True:
            try:
                message = input("👤 ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if message in ("/quit", "/exit", "q"):
                break
            if message == "/facts":
                print(json.dumps(session.facts, ensure_ascii=False, indent=2))
                continue
            if not message:
                continue
            try:
                reply = session.send(message)
            except ValueError as exc:
                print(f"   {exc}")
                continue
            show(reply)
            if not reply.awaiting_answer:
                print("\n（本次对话已达终态；继续输入可开启新的追问）")

    if args.transcript:
        with open(args.transcript, "w", encoding="utf-8") as handle:
            json.dump(session.to_dict(), handle, ensure_ascii=False, indent=2)
        print(f"\n对话记录已写入 {args.transcript}")
    return 0


def _default_manifest() -> str:
    from pathlib import Path as _Path

    return str(_Path(__file__).parent / "skills" / "manifest.yaml")


def _skill_command(args) -> int:
    from .expert.profile import build_profile
    from .expert.skillgen import audit_for_phi, build_skill_entry, merge_into_manifest, write_skill_file
    from .skills.loader import SkillRegistry

    manifest = args.skill_manifest or _default_manifest()

    if args.skcmd == "list":
        registry = SkillRegistry.from_file(manifest)
        return _emit({"manifest": manifest, "skills": registry.catalog()})

    if args.skcmd == "show":
        registry = SkillRegistry.from_file(manifest)
        spec = registry.specs.get(args.skill_id)
        if spec is None:
            print(json.dumps({"error": f"unknown skill {args.skill_id}"}, ensure_ascii=False), file=sys.stderr)
            return 2
        return _emit({
            "skill_id": spec.skill_id, "version": spec.version, "description": spec.description,
            "autonomous": spec.autonomous, "allowed_tools": list(spec.allowed_tools),
            "forbidden_tools": list(spec.forbidden_tools), "output_schema": spec.output_schema,
            "hard_requirements": list(spec.hard_requirements), "instructions": spec.instructions,
        })

    # build-expert
    if args.xlsx:
        try:
            store = ExpertCaseStore(args.xlsx)
        except DeidentificationKeyError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        records = store.records
    elif args.records_json:
        with open(args.records_json, encoding="utf-8") as handle:
            raw = json.load(handle)
        records = ExpertCaseStore.from_records(raw).records
    else:
        print(json.dumps({"error": "需要 --xlsx 或 --records-json"}, ensure_ascii=False), file=sys.stderr)
        return 2

    profile = build_profile(records, min_support=args.min_support)
    entry = build_skill_entry(profile, source_label=args.source_label)
    problems = audit_for_phi(entry)
    if problems:
        print(json.dumps({"error": "生成的技能未通过 PHI 自检", "problems": problems},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    out = write_skill_file(profile, args.out, source_label=args.source_label)
    merged = merge_into_manifest(entry, manifest) if args.merge else None
    return _emit({
        "written": str(out),
        "merged_into": str(merged) if merged else None,
        "skill_id": entry["skill_id"],
        "total_cases": profile.total_cases,
        "patterns": {name: p.n_cases for name, p in profile.patterns.items()},
        "followup": profile.followup,
        "phi_audit": "passed",
        "next": "用 --skill-manifest 指向合并后的 manifest，或直接使用已合并的主 manifest",
    })


def _knowledge_command(args) -> int:
    from .knowledge import ortho_interactions
    from .knowledge.connectors.files import FileIngestError
    from .knowledge.ingest import build, ingest_file, list_sources, open_store
    from .knowledge.licensing import LicenseError, LicensePolicy

    try:
        policy = LicensePolicy.from_env()
    except LicenseError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    if args.kcmd == "sources":
        return _emit({"policy": policy.to_dict(), "sources": list_sources(policy)})
    if args.kcmd == "rules":
        return _emit({
            "summary": ortho_interactions.rule_pack_summary(),
            "rules": [r.to_dict() for r in ortho_interactions.ORTHO_RULES],
        })
    if args.kcmd == "check-interactions":
        store = open_store(args.store, policy) if args.store else None
        registry = ToolRegistry(knowledge=store, deid_key="cli-interaction-check")
        result = registry.drug_interaction_check(args.medications, args.conditions)
        return _emit(result.data)

    store = open_store(args.store, policy)
    try:
        if args.kcmd == "build":
            report = build(
                store,
                ingredients=args.ingredients or _default_ingredients(),
                cache_dir=args.cache_dir,
                nice_api_key=args.nice_api_key,
                include=args.include,
            )
            return _emit(report)
        if args.kcmd == "ingest-file":
            return _emit(ingest_file(store, args.source, args.kind, args.path, args.version))
        if args.kcmd == "stats":
            return _emit(store.stats())
    except (LicenseError, FileIngestError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    finally:
        store.close()
    return 0


def _default_ingredients():
    from .knowledge.ingest import ORTHOPAEDIC_INGREDIENTS

    return ORTHOPAEDIC_INGREDIENTS


def _open_knowledge(path: str | None):
    """Open the knowledge store for a clinical run, or return None."""
    if not path:
        return None
    from .knowledge.ingest import open_store

    return open_store(path)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "chat":
        return _chat_command(args)

    if args.cmd == "skill":
        return _skill_command(args)

    if args.cmd == "knowledge":
        return _knowledge_command(args)

    if args.cmd == "ui":
        from .ui.server import serve

        serve(
            host=args.host, port=args.port,
            knowledge_store=args.knowledge_store, xlsx=args.xlsx,
            llm_provider=args.llm_provider, llm_model=args.llm_model,
            checkpoint_dir=args.checkpoint_dir, skill_manifest=args.skill_manifest,
            open_browser=args.open, public=args.public, access_token=args.access_token,
            ngrok_authtoken=args.ngrok_authtoken, ngrok_region=args.ngrok_region,
        )
        return 0

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
        knowledge = _open_knowledge(args.knowledge_store)
        tools = ToolRegistry(args.xlsx or None, knowledge=knowledge)
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
    runner = YaobiGraphRunner(tools, checkpoint_dir=args.checkpoint_dir,
                              skill_manifest=args.skill_manifest, llm=llm)
    out = runner.run(state, allow_prescription=args.allow_prescription)
    print(json.dumps(render(out, debug=args.debug_state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
