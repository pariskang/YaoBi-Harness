from __future__ import annotations
import argparse, json
from .state import ClinicalRunState
from .tools import ToolRegistry, ExpertCaseStore
from .graph import YaobiGraphRunner


def main(argv=None) -> int:
    p=argparse.ArgumentParser(prog="yaobi-harness")
    sub=p.add_subparsers(dest="cmd", required=True)
    r=sub.add_parser("run"); r.add_argument("--complaint", required=True); r.add_argument("--role", choices=["patient","physician","researcher"], default="physician"); r.add_argument("--xlsx", help="local authorized Excel; never committed or returned raw"); r.add_argument("--allow-prescription", action="store_true"); r.add_argument("--checkpoint-dir")
    i=sub.add_parser("inspect-xlsx"); i.add_argument("path")
    args=p.parse_args(argv)
    if args.cmd=="inspect-xlsx":
        store=ExpertCaseStore(args.path)
        herb_count=sum(len(r.get("herbs",[])) for r in store.records)
        print(json.dumps({"records_loaded":len(store.records),"parsed_herb_items":herb_count,"privacy":"direct identifiers removed; retrieval returns structured deidentified fields only","positioning":["single_expert_memory","followup_trajectory","similar_case_retrieval","stratified_dose_distribution"],"not_causal_trial":True},ensure_ascii=False,indent=2)); return 0
    state=ClinicalRunState(complaint=args.complaint, role=args.role)
    out=YaobiGraphRunner(ToolRegistry(args.xlsx) if args.xlsx else ToolRegistry(), checkpoint_dir=args.checkpoint_dir).run(state, allow_prescription=args.allow_prescription)
    print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
