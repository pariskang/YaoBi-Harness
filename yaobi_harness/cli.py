"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

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
    run.add_argument("--panel", action="store_true",
                     help="convene the multi-speciality consult panel (several subagents; costs more tokens)")
    run.add_argument("--image", action="append", metavar="KIND:PATH",
                     help="attach a de-identified image, e.g. radiograph:/path/x.jpg; repeatable. "
                          "Attaching one asserts you have removed名/ID/日期/条码/人脸")
    run.add_argument("--no-vision", action="store_true", help="disable the vision model even if configured")
    run.add_argument("--skill-dir", action="append", help="extra SKILL.md root, highest precedence; repeatable")
    run.add_argument("--journal", metavar="PATH",
                     help="record every tool and model call here, so this decision can be replayed offline later")
    run.add_argument("--replay", metavar="PATH",
                     help="replay a recorded journal instead of calling anything; diverging calls fail the run closed")
    run.add_argument("--panel-concurrency", type=int, metavar="N",
                     help="threads for the consult panel (default 4; 1 forces sequential)")
    run.add_argument("--summary", action="store_true",
                     help="print the structured clinical note as text instead of the JSON view")

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
    chat.add_argument("--image", action="append", metavar="KIND:PATH",
                      help="attach a de-identified image before the first turn; repeatable")
    chat.add_argument("--no-vision", action="store_true")
    chat.add_argument("--skill-dir", action="append")
    chat.add_argument("--journal", metavar="PATH", help="record every call for later offline replay")
    chat.add_argument("--panel-concurrency", type=int, metavar="N",
                      help="threads for the consult panel (default 4; 1 forces sequential)")

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
    ui.add_argument("--no-vision", action="store_true", help="disable the vision model even if configured")
    ui.add_argument("--skill-dir", action="append", help="extra SKILL.md root, highest precedence; repeatable")
    ui.add_argument("--panel-concurrency", type=int, metavar="N",
                    help="default consult-panel threads for the console (the page can override per run)")

    journal = sub.add_parser("journal", help="inspect a recorded call journal")
    journal.add_argument("path")
    journal.add_argument("--entries", action="store_true", help="list every recorded call")

    interview = sub.add_parser("interview", help="inspect the history-taking axes (十问歌 + 骨科专科)")
    interview.add_argument("--tier", choices=["RED_FLAG", "CORE", "SPECIALTY", "TCM", "CONTEXT"])
    interview.add_argument("--axis", help="show one axis in full")
    interview.add_argument("--complaint", help="show which axes this complaint makes relevant, and which are required")
    interview.add_argument("--facts", help="JSON object of known facts, to see what is already closed")
    interview.add_argument("--role", choices=["patient", "physician", "researcher"], default="patient")

    vision = sub.add_parser("vision", help="read one clinical image (non-diagnostic; requires a vision model)")
    vision.add_argument("image", help="local image path")
    vision.add_argument("--kind", default="other",
                        choices=["radiograph", "mri_ct", "tongue", "posture_gait",
                                 "limb_surface", "report_document", "other"])
    vision.add_argument("--context", default="", help="clinical background to focus the read")
    vision.add_argument("--deidentified", action="store_true",
                        help="required: asserts名/ID/日期/条码/人脸 have been masked")

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


def _make_vision(enabled: bool = True):
    """Build the vision client, or ``None`` when it is off or unconfigured.

    Never raises: an unset ``POE_API_KEY`` should leave the image tools reporting
    themselves unavailable, not stop the CLI from running a text-only case.
    """
    if not enabled:
        return None
    from .vision.client import build_vision_client

    return build_vision_client()


def _emit(payload) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_images(specs: list[str] | None) -> list[dict]:
    """Parse ``KIND:PATH`` image arguments.

    Passing an image on the command line is itself the de-identification
    attestation: the operator typed the path, so they are the one asserting the
    file is masked. The tool still refuses to read anything the PHI pre-check
    flags, so the attestation is a declaration of intent, not a bypass.
    """
    from .vision.client import IMAGE_KINDS

    images: list[dict] = []
    for spec in specs or []:
        kind, _, path = str(spec).partition(":")
        if not path:
            kind, path = "other", spec
        if kind not in IMAGE_KINDS:
            raise ValueError(f"未知图片类型 {kind!r}；支持 {list(IMAGE_KINDS)}")
        images.append({"kind": kind, "ref": path, "deidentified": True})
    return images


def _open_journal(args) -> Any:
    """Build the run's journal from ``--journal`` / ``--replay``."""
    from .journal import open_journal

    replay = getattr(args, "replay", None)
    record = getattr(args, "journal", None)
    if replay and record:
        raise ValueError("--journal 与 --replay 不能同时使用：一次运行只能录制或重放")
    if replay:
        return open_journal(replay, mode="replay")
    return open_journal(record, mode="record") if record else None


def _journal_command(args) -> int:
    """Show what a recorded journal contains, and how to replay it."""
    from .journal import Journal

    try:
        journal = Journal.load(args.path, mode="replay")
    except Exception as exc:  # noqa: BLE001
        return _emit({"error": str(exc)})

    payload = {"summary": journal.summary(), "replay_hint": journal.replay_hint()}
    if args.entries:
        payload["entries"] = [
            {"seq": e.seq, "kind": e.kind, "label": e.label, "req_hash": e.req_hash[:16],
             "result_summary": (
                 (e.result or {}).get("summary") if e.kind == "tool"
                 else str((e.result or {}).get("text", ""))[:120]
             )}
            for e in journal.entries
        ]
    else:
        payload["calls"] = [f"{e.seq}. {e.kind}:{e.label}" for e in journal.entries[:40]]
    return _emit(payload)


def _interview_command(args) -> int:
    """Show the history-taking axes, and what a given complaint makes relevant."""
    from .interview.axes import AXES, AXES_BY_ID, coverage, plan_next, relevant_axes, required_open_axes

    if args.axis:
        axis = AXES_BY_ID.get(args.axis)
        if axis is None:
            return _emit({"error": f"未知问诊轴 {args.axis!r}", "available": sorted(AXES_BY_ID)})
        return _emit(axis.to_dict())

    if args.complaint:
        facts = json.loads(args.facts) if args.facts else {}
        return _emit({
            "complaint": args.complaint,
            "role": args.role,
            "relevant": [
                {"axis_id": a.axis_id, "label": a.label, "tier": a.tier,
                 "answered": a.satisfied(facts)}
                for a in relevant_axes(facts, args.complaint, role=args.role)
            ],
            "required_open": [a.label for a in required_open_axes(facts, args.complaint, role=args.role)],
            "next_round": plan_next(facts, args.complaint, role=args.role).to_dict(),
            "coverage": coverage(facts, args.complaint, role=args.role),
        })

    selected = [a for a in AXES if not args.tier or a.tier == args.tier]
    return _emit({
        "total": len(selected),
        "axes": [
            {"axis_id": a.axis_id, "label": a.label, "tier": a.tier, "tradition": a.tradition,
             "closes": list(a.closes), "rationale": a.rationale, "probes": list(a.probes)}
            for a in selected
        ],
    })


def _vision_command(args) -> int:
    """Read one image. Refuses without the de-identification attestation."""
    from .vision.client import VisionError, build_vision_client, describe_vision

    if not args.deidentified:
        return _emit({
            "error": "缺少去标识化声明",
            "required": "--deidentified",
            "why": "影像翻拍照片常带姓名/住院号/日期/条码；请先遮盖再判读",
        })
    client = build_vision_client()
    if client is None or not client.available:
        return _emit({
            "error": "未配置视觉模型",
            "how_to_fix": "export YAOBI_VISION_PROVIDER=poe POE_API_KEY=... YAOBI_VISION_MODEL=Gemini-3.1-Pro",
            "status": describe_vision(client),
        })
    try:
        read = client.read(args.image, kind=args.kind, context=args.context)
    except VisionError as exc:
        return _emit({"error": str(exc)})
    return _emit({"vision": describe_vision(client), "read": read.to_dict()})


def _chat_command(args) -> int:
    """Interactive dialogue, or a scripted one via repeated --message."""
    from .conversation import ConversationSession

    try:
        llm = _make_llm(args.llm_provider, args.llm_model)
        knowledge = _open_knowledge(args.knowledge_store)
        tools = ToolRegistry(args.xlsx or None, knowledge=knowledge,
                             vision=_make_vision(not getattr(args, "no_vision", False)))
    except (LLMError, DeidentificationKeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    # ``chat --journal`` accepted a path and then recorded nothing, because the
    # journal was only ever built on the ``run`` path. A flag that silently does
    # nothing is worse than no flag: the operator believes the dialogue is
    # replayable.
    try:
        journal = _open_journal(args)
    except Exception as exc:  # noqa: BLE001 - a bad journal path must not traceback
        print(json.dumps({"error": f"日志打开失败: {exc}"}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    runner = YaobiGraphRunner(tools, skill_manifest=args.skill_manifest, llm=llm,
                              skill_dirs=getattr(args, "skill_dir", None), journal=journal,
                              panel_concurrency=getattr(args, "panel_concurrency", None))
    session = ConversationSession(role=args.role, runner=runner,
                                  allow_prescription=args.allow_prescription)
    try:
        for image in _parse_images(getattr(args, "image", None)):
            session.attach_image(image["ref"], kind=image["kind"], deidentified=True)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    def show(reply) -> None:
        flag = " ⚠已升级为急症" if reply.escalated else ""
        print(f"\n🤖 [{reply.release_status}/{reply.risk_mode}{flag}]")
        print("   " + reply.message.replace("\n", "\n   "))
        if reply.extracted:
            print(f"   ↳ 本轮获得: {json.dumps(reply.extracted, ensure_ascii=False)}")
        interview = reply.interview or {}
        if interview.get("verdict"):
            print(
                f"   ↳ 问诊: 第{interview['rounds_used']}轮 覆盖{interview['coverage_ratio']:.0%} "
                f"判定={interview['verdict']}({interview.get('judged_by') or 'rule'}) "
                f"提问来源={interview['composer']}"
            )
            if interview.get("blocking"):
                print(f"   ↳ 必答未闭合: {'、'.join(interview['blocking'])}")
            for note in interview.get("notes", [])[:2]:
                print(f"   ↳ 提问记录: {note}")
        for request in reply.image_requests or []:
            print(f"   📷 请上传【{request.get('kind')}】: {request.get('why', '')}")
            print("      上传前请遮盖姓名、各类编号、日期、条码与人脸；"
                  "命令行用 --image kind:path 附加")
        screening = ((reply.delivered or {}).get("intake") or {}).get("screening") or {}
        if screening.get("triage_by") == "llm":
            print(f"   ↳ 分诊: {screening.get('triage_level')}（模型判定）{screening.get('triage_reason', '')[:80]}")

    # The agent opens. Waiting for the patient to recite a complaint into an
    # empty prompt is both colder and worse at collecting a history.
    opening = session.open()
    print(f"\n🤖 {opening.message}")

    def show_note() -> None:
        note = session.clinical_note()
        if not note:
            return
        print("\n" + "=" * 68)
        print(session.note_text())
        print("=" * 68)

    if args.message:
        for message in args.message:
            print(f"\n👤 {message}")
            show(session.send(message))
        show_note()
    else:
        print(f"\n（角色={args.role}，模型={describe_client(llm)['provider']}）")
        print("输入 /quit 结束，/facts 查看已知信息，/note 查看病历摘要。\n")
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
            if message in ("/note", "/summary"):
                if session.clinical_note():
                    print(session.note_text())
                else:
                    print("本次问诊还没有得出结论，暂无病历摘要。")
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

    if args.cmd == "journal":
        return _journal_command(args)

    if args.cmd == "interview":
        return _interview_command(args)

    if args.cmd == "vision":
        return _vision_command(args)

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
            skill_dirs=getattr(args, "skill_dir", None), vision=not getattr(args, "no_vision", False),
            panel_concurrency=getattr(args, "panel_concurrency", None),
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
        tools = ToolRegistry(args.xlsx or None, knowledge=knowledge,
                             vision=_make_vision(not getattr(args, "no_vision", False)))
    except DeidentificationKeyError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    try:
        journal = _open_journal(args)
    except Exception as exc:  # noqa: BLE001 - a bad journal path must not traceback
        print(json.dumps({"error": f"日志打开失败: {exc}"}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    state = ClinicalRunState(complaint=args.complaint, role=args.role)
    state.facts.update(_load_facts(args.facts, args.facts_file))
    state.enable_panel = bool(getattr(args, "panel", False))
    try:
        state.images = _parse_images(getattr(args, "image", None))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    state.budget = Budget(
        max_loops=args.max_loops,
        max_tool_calls=args.max_tool_calls,
        max_llm_calls=args.max_llm_calls,
    )
    runner = YaobiGraphRunner(tools, checkpoint_dir=args.checkpoint_dir,
                              skill_manifest=args.skill_manifest, llm=llm,
                              skill_dirs=getattr(args, "skill_dir", None), journal=journal,
                              panel_concurrency=getattr(args, "panel_concurrency", None))
    out = runner.run(state, allow_prescription=args.allow_prescription)
    if getattr(args, "summary", False):
        note = out.outputs.get("clinical_note")
        if note:
            print(note["text"])
        else:
            print(f"本次运行未得出结论（{out.release_status}），暂不生成病历摘要。", file=sys.stderr)
            return 2
    else:
        print(json.dumps(render(out, debug=args.debug_state), ensure_ascii=False, indent=2))
    # A diverged replay has not reproduced the recording, so it must not exit 0:
    # a script that treats exit status as "the replay confirmed the decision"
    # would otherwise be told yes.
    if journal is not None and journal.mode == "replay" and journal.diverged:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
