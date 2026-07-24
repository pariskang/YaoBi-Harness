from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.state import ClinicalRunState


def test_urgent_mode_withholds_prescription_and_gives_action_plan():
    st = ClinicalRunState("突发腰痛伴尿潴留和会阴麻木", role="patient")
    out = YaobiGraphRunner().run(st, allow_prescription=True)
    assert out.risk_mode == "urgent"
    assert out.release_status == "urgent_action_plan"
    assert "prescription_draft" not in out.outputs
    assert "immediate_action" in out.outputs["urgent_action_plan"]


def test_patient_routine_gets_no_dose_prescription():
    st = ClinicalRunState("腰痛3月，久坐加重，右下肢麻木", role="patient")
    out = YaobiGraphRunner().run(st, allow_prescription=True)
    assert out.release_status in {"needs_more_information", "treatment_advice_only"}
    assert "prescription_draft" not in out.outputs
    assert "biomedical" in out.outputs and "tcm_pattern" in out.outputs


def test_physician_dose_fails_closed_without_dose_evidence():
    st = ClinicalRunState("腰痛3月，久坐加重，右下肢麻木", role="physician")
    st.facts["special_population"] = {"pregnancy": False, "age": 55, "renal": "unknown", "liver": "unknown"}
    out = YaobiGraphRunner().run(st, allow_prescription=True)
    assert out.release_status == "treatment_advice_only"
    assert any("缺少剂量依据" in x for x in out.safety_issues)
