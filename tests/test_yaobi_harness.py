from pathlib import Path

from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.state import ClinicalRunState, Budget
from yaobi_harness.tools import ToolRegistry, ExpertCaseStore, parse_herbs
from yaobi_harness.skills.loader import SkillRegistry

RAW_RECORD = {
    "就诊序号": "12725633", "医师工号": "007", "医师姓名": "沈钦荣", "科室代码": "1166",
    "姓名": "张三", "性别": "男", "年龄": "63岁", "病案号": "50512983", "地址": "某区某街道",
    "主诉": "右腰部疼痛10天", "现病史": "久坐后疼痛明显，无双下肢麻木疼痛。舌略暗。", "中医诊断": "腰痹/证型：气血痹阻证", "西医诊断": "腰痛",
    "中药": "1/独活*1克/10克/用法：无/贴数:7\n,2/盐杜仲*1克/12克/用法：无/贴数:7\n,3/细辛*1克/3克/用法：无/贴数:7",
}


def test_deidentified_case_search_never_returns_phi():
    out = ToolRegistry(records=[RAW_RECORD]).case_store.search("腰痛 久坐", 1)[0]
    blob = str(out)
    assert "张三" not in blob and "50512983" not in blob and "某区某街道" not in blob and "007" not in blob
    assert out["research_patient_id"].startswith("YP")


def test_realistic_herb_dose_parser_matches_star_slash_format():
    herbs = parse_herbs(RAW_RECORD["中药"])
    assert {h["herb_name"]: h["dose_g"] for h in herbs} == {"独活": 10.0, "盐杜仲": 12.0, "细辛": 3.0}


def test_negated_red_flags_do_not_trigger_urgent_but_chest_pain_does():
    routine = YaobiGraphRunner().run(ClinicalRunState("腰痛，无发热、无外伤、无大小便失禁、无会阴麻木", role="patient"))
    urgent = YaobiGraphRunner().run(ClinicalRunState("突发胸痛、大汗、呼吸困难", role="patient"))
    assert routine.risk_mode == "routine"
    assert urgent.risk_mode == "urgent" and urgent.release_status == "urgent_action_plan"


def test_urgent_mode_withholds_prescription_and_gives_action_plan():
    st = ClinicalRunState("突发腰痛伴尿潴留和会阴麻木", role="patient")
    out = YaobiGraphRunner().run(st, allow_prescription=True)
    assert out.risk_mode == "urgent"
    assert out.release_status == "urgent_action_plan"
    assert "prescription_draft" not in out.outputs
    assert "immediate_action" in out.outputs["urgent_action_plan"]


def test_budget_zero_fails_closed_before_tool_execution():
    st = ClinicalRunState("腰痛3月", role="physician")
    st.budget = Budget(max_tool_calls=0)
    out = YaobiGraphRunner().run(st)
    assert out.release_status == "failed_closed"
    assert out.budget.used_tool_calls == 0


def test_critical_tool_failure_fails_closed():
    out = YaobiGraphRunner(ToolRegistry(failing_tools={"clinical_guideline_search"})).run(ClinicalRunState("腰痛3月，久坐加重", role="physician"))
    assert out.release_status == "failed_closed"
    assert any("关键工具失败" in x for x in out.safety_issues)


def test_patient_cannot_call_formula_tool_via_broker_path():
    st = ClinicalRunState("腰痛3月，久坐加重", role="patient")
    out = YaobiGraphRunner().run(st, allow_prescription=True)
    assert "formula" not in out.outputs and "prescription_draft" not in out.outputs


def test_risk_herb_blocks_draft_even_with_dose_data():
    st = ClinicalRunState("腰痛3月，怕冷，久坐加重", role="physician")
    st.facts["special_population"] = {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"}
    out = YaobiGraphRunner(ToolRegistry(records=[RAW_RECORD])).run(st, allow_prescription=True)
    assert out.release_status == "treatment_advice_only"
    assert any("风险药" in x or "可靠剂量依据" in x or "授权药典范围" in x for x in out.safety_issues)


def test_skill_manifest_enforces_forbidden_tools():
    reg = SkillRegistry.from_file(Path("yaobi_harness/skills/manifest.yaml"))
    ok, problems = reg.enforce("yaobi.urgent_triage", "patient", ["red_flag_evidence_search", "herb_dose_distribution"])
    assert not ok and any("forbidden" in p for p in problems)


def test_contextual_red_flags_ignore_family_hypothetical_and_past_recovered():
    for text in ["父亲患癌，本人只是久坐腰酸", "如果以后胸痛怎么办，目前无不适", "去年曾跌倒且已经痊愈", "慢性骨质疏松，多年无新发症状", "单纯夜间腰痛"]:
        out = YaobiGraphRunner().run(ClinicalRunState(text, role="patient"))
        assert out.risk_mode == "routine", text
    urgent = YaobiGraphRunner().run(ClinicalRunState("突然不能排尿、下身迟钝、双腿越来越无力", role="patient"))
    assert urgent.risk_mode == "urgent"


def test_single_case_999g_cannot_create_dose_draft_without_authorized_ranges_and_min_n():
    bad = dict(RAW_RECORD)
    bad["中药"] = "\n".join([f",{i}/{h}*1克/999克/用法：无/贴数:7" for i, h in enumerate(["独活","桑寄生","杜仲","牛膝","当归","川芎","白芍","熟地黄","党参","茯苓","甘草"], 1)])
    st = ClinicalRunState("腰痛3月，久坐加重", role="physician")
    st.facts.update({"special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"}, "medications_confirmed": True, "allergies_confirmed": True})
    out = YaobiGraphRunner(ToolRegistry(records=[bad])).run(st, allow_prescription=True)
    assert out.release_status == "treatment_advice_only"
    assert "prescription_draft" not in out.outputs


def test_physician_review_rejects_incomplete_risk_herb():
    result = ToolRegistry().physician_review_submit({"prescription_hash": "abc", "herbs": [{"herb_name": "附片"}]}, {"附片": True}, physician_id="D1", signature="sig")
    assert not result.ok
    assert any("risk_herb" in p or "missing_dose" in p for p in result.data["problems"])


def test_skill_manifest_is_used_by_runner_when_deleted_or_denied(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("skills:\n  - skill_id: yaobi.intake\n    version: 2.0.0\n    allowed_tools: []\n", encoding="utf-8")
    out = YaobiGraphRunner(skill_manifest=manifest).run(ClinicalRunState("腰痛", role="patient"))
    assert out.release_status == "failed_closed"
    assert any("skill_policy_denied" in x for x in out.safety_issues)
