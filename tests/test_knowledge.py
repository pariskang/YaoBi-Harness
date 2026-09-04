"""Tests for the external-knowledge layer.

Two things are being pinned here:

* **Licence enforcement is real.** A non-commercial dataset cannot enter a
  commercial deployment, a read-only source cannot have its body text stored,
  and a credentialed source stays closed without an attestation. These are the
  rules that keep the repository shipping code rather than content.
* **Real data changes clinical behaviour.** A licensed dose range overrides the
  local table, a licensed guideline stops being stub-grade evidence, and an
  orthopaedic interaction blocks a run that would otherwise have gone out as
  ordinary advice.

Network access is never required: connectors are exercised against a local HTTP
server that mimics the upstream contract.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.knowledge import ortho_interactions as oi
from yaobi_harness.knowledge.connectors.files import (
    FileIngestError, ingest_ddinter, ingest_dose_ranges, ingest_guidelines, ingest_interactions,
)
from yaobi_harness.knowledge.connectors.web import DailyMedConnector, OpenFdaConnector, RxNormConnector
from yaobi_harness.knowledge.ingest import list_sources
from yaobi_harness.knowledge.licensing import (
    Attestation, DeploymentMode, LicenseError, LicensePolicy, Reuse,
)
from yaobi_harness.knowledge.sources import SOURCE_CATALOG, get_source
from yaobi_harness.knowledge.store import KnowledgeStore
from yaobi_harness.render import citation_bundle, render
from yaobi_harness.state import ClinicalRunState
from yaobi_harness.tools import ToolRegistry

TEST_KEY = "unit-test-fixed-key"
RESEARCH = DeploymentMode.RESEARCH
COMMERCIAL = DeploymentMode.COMMERCIAL

FORMULA_HERBS = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎", "白芍", "熟地黄", "党参", "茯苓", "甘草"]
BLOOD_STASIS_EXTRA = ["桃仁", "红花", "延胡索"]


def case_with_doses(index: int, dose: float) -> dict:
    herbs = FORMULA_HERBS + BLOOD_STASIS_EXTRA
    body = "".join(f",{i}/{h}*1克/{dose}克/用法：无/贴数:7\n" for i, h in enumerate(herbs, 1))
    return {"病案号": f"C{index}", "年龄": "63岁", "主诉": "腰痛",
            "中医诊断": "腰痹/证型：气滞血瘀证", "中药": body}


def physician_state() -> ClinicalRunState:
    state = ClinicalRunState("腰痛3月，刺痛固定，久坐加重", role="physician")
    state.facts.update({
        "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
        "medications_confirmed": True,
        "allergies_confirmed": True,
    })
    return state


def attested(source_id: str, mode: DeploymentMode = RESEARCH) -> LicensePolicy:
    return LicensePolicy(mode, {source_id: Attestation("Test Hospital", f"{source_id}-LIC-1", "2030-01-01")})


# ------------------------------------------------------------------ licensing

class LicensePolicyTests(unittest.TestCase):
    def test_catalog_declares_a_licence_for_every_source(self):
        for source_id, source in SOURCE_CATALOG.items():
            with self.subTest(source=source_id):
                self.assertTrue(source.license.license_name)
                self.assertIsInstance(source.license.reuse, Reuse)

    def test_noncommercial_sources_are_blocked_in_commercial_mode(self):
        policy = LicensePolicy(COMMERCIAL)
        for source_id in ("ddinter", "who_guidelines", "who_intl_pharmacopoeia"):
            with self.subTest(source=source_id):
                enabled, reason = policy.evaluate(get_source(source_id))
                self.assertFalse(enabled)
                self.assertIn("noncommercial", reason)

    def test_public_domain_sources_are_allowed_in_commercial_mode(self):
        policy = LicensePolicy(COMMERCIAL)
        for source_id in ("openfda", "dailymed", "rxnorm", "yaobi_ortho_rules"):
            with self.subTest(source=source_id):
                self.assertTrue(policy.evaluate(get_source(source_id))[0], source_id)

    def test_credentialed_sources_need_an_attestation(self):
        for source_id in ("nice", "chp_2025", "drugbank", "bnf_stockley", "usp_nf", "ph_eur", "nmpa_labels"):
            with self.subTest(source=source_id):
                self.assertFalse(LicensePolicy(RESEARCH).evaluate(get_source(source_id))[0])
                self.assertTrue(attested(source_id).evaluate(get_source(source_id))[0])

    def test_policy_from_env_reads_mode_and_attestations(self):
        saved = {k: os.environ.get(k) for k in ("YAOBI_DEPLOYMENT_MODE", "YAOBI_LICENSE_ATTESTATIONS")}
        try:
            os.environ["YAOBI_DEPLOYMENT_MODE"] = "commercial"
            os.environ["YAOBI_LICENSE_ATTESTATIONS"] = json.dumps(
                {"chp_2025": {"licensee": "H", "license_reference": "R1"}}
            )
            policy = LicensePolicy.from_env()
            self.assertIs(policy.mode, COMMERCIAL)
            self.assertIn("chp_2025", policy.attestations)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value

    def test_bad_env_configuration_raises(self):
        saved = os.environ.get("YAOBI_DEPLOYMENT_MODE")
        try:
            os.environ["YAOBI_DEPLOYMENT_MODE"] = "whatever"
            with self.assertRaises(LicenseError):
                LicensePolicy.from_env()
        finally:
            os.environ.pop("YAOBI_DEPLOYMENT_MODE", None)
            if saved is not None:
                os.environ["YAOBI_DEPLOYMENT_MODE"] = saved

    def test_source_listing_reports_enabled_state(self):
        rows = {r["source_id"]: r for r in list_sources(LicensePolicy(RESEARCH))}
        self.assertTrue(rows["openfda"]["enabled"])
        self.assertFalse(rows["nice"]["enabled"])


class StoreLicenceEnforcementTests(unittest.TestCase):
    def test_commercial_store_refuses_noncommercial_writes(self):
        with KnowledgeStore(":memory:", LicensePolicy(COMMERCIAL)) as store:
            with self.assertRaises(LicenseError):
                store.add_interaction("ddinter", "warfarin", "ibuprofen", "major")

    def test_credentialed_write_requires_attestation(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            with self.assertRaises(LicenseError):
                store.add_dose_range("chp_2025", "独活", 3.0, 10.0)
        with KnowledgeStore(":memory:", attested("chp_2025")) as store:
            store.add_dose_range("chp_2025", "独活", 3.0, 10.0)
            self.assertEqual(store.dose_range("独活")["max_value"], 10.0)

    def test_link_only_source_stores_citation_but_never_body(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            store.add_guideline(
                "cma_guidelines", "CMA-1", "中国腰痛诊疗指南",
                topic="腰痛", url="https://example.org/cma-1",
                body="不应入库的受版权全文", recommendations=["先排除红旗信号"],
            )
            hit = store.search_guidelines("腰痛")[0]
        self.assertFalse(hit["has_full_text"])
        self.assertEqual(hit["recommendations"], ["先排除红旗信号"])
        self.assertEqual(hit["provenance"]["source_id"], "cma_guidelines")

    def test_link_only_source_requires_a_citation_url(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            with self.assertRaises(LicenseError):
                store.add_guideline("aaos_cpg", "AAOS-1", "Distal radius fracture CPG")

    def test_invalid_dose_range_is_rejected(self):
        with KnowledgeStore(":memory:", attested("chp_2025")) as store:
            with self.assertRaises(ValueError):
                store.add_dose_range("chp_2025", "独活", 10.0, 3.0)

    def test_licensed_pharmacopoeia_wins_over_a_lower_priority_source(self):
        policy = attested("chp_2025")
        with KnowledgeStore(":memory:", policy) as store:
            store.add_dose_range("openfda", "独活", 1.0, 99.0, basis="label text")
            store.add_dose_range("chp_2025", "独活", 3.0, 10.0, basis="《中国药典》2025 一部")
            best = store.dose_range("独活")
        self.assertEqual(best["provenance"]["source_id"], "chp_2025")
        self.assertEqual((best["min_value"], best["max_value"]), (3.0, 10.0))


# --------------------------------------------------------------- file ingest

class FileIngestTests(unittest.TestCase):
    def _write(self, tmp: str, name: str, payload) -> Path:
        path = Path(tmp) / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_dose_range_ingest_populates_the_store(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", attested("chp_2025")) as store:
            path = self._write(tmp, "ranges.json", {"dose_ranges": [
                {"substance": "独活", "min": 3, "max": 10, "unit": "g", "basis": "《中国药典》2025 一部"},
                {"substance": "细辛", "min": 1, "max": 3, "unit": "g"},
                {"substance": "缺字段"},
            ]})
            report = ingest_dose_ranges(store, "chp_2025", path, version="2025")
            self.assertEqual(report["dose_ranges_stored"], 2)
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(store.dose_range("细辛")["max_value"], 3.0)

    def test_csv_ingest_is_supported(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", attested("chp_2025")) as store:
            path = Path(tmp) / "ranges.csv"
            path.write_text("substance,min,max,unit\n杜仲,6,10,g\n", encoding="utf-8")
            self.assertEqual(ingest_dose_ranges(store, "chp_2025", path)["dose_ranges_stored"], 1)

    def test_guideline_ingest_and_search(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            path = self._write(tmp, "g.json", {"guidelines": [{
                "id": "VA-LBP-2022", "title": "VA/DoD Low Back Pain CPG", "topic": "low back pain",
                "url": "https://www.healthquality.va.gov/", "published": "2022-02-01",
                "evidence_grade": "Strong for", "summary": "conservative care first",
                "recommendations": ["Screen for red flags before imaging"],
            }]})
            ingest_guidelines(store, "vadod_cpg", path, version="2022")
            hits = store.search_guidelines("low back pain")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["provenance"]["source_id"], "vadod_cpg")
        self.assertEqual(hits[0]["evidence_grade"], "Strong for")

    def test_interaction_ingest_is_symmetric(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", attested("drugbank")) as store:
            path = self._write(tmp, "ddi.json", {"interactions": [
                {"subject": "Warfarin", "object": "Ibuprofen", "severity": "major",
                 "management": "avoid; monitor INR"},
            ]})
            ingest_interactions(store, "drugbank", path)
            found = store.interactions_for(["ibuprofen", "warfarin"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["severity"], "major")

    def test_ddinter_ingest_is_refused_in_commercial_mode(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", LicensePolicy(COMMERCIAL)) as store:
            path = self._write(tmp, "ddinter.json", [{"Drug_A": "a", "Drug_B": "b", "Level": "Major"}])
            with self.assertRaises(LicenseError):
                ingest_ddinter(store, path)

    def test_ddinter_ingest_works_in_research_mode(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            path = self._write(tmp, "ddinter.json", [{"Drug_A": "colchicine", "Drug_B": "clarithromycin", "Level": "Major"}])
            report = ingest_ddinter(store, path)
            self.assertEqual(report["interactions_stored"], 1)
            self.assertIn("非商业", report["notice"])

    def test_missing_file_raises_a_clear_error(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            with self.assertRaises(FileIngestError):
                ingest_guidelines(store, "vadod_cpg", "/nonexistent/path.json")


# ---------------------------------------------------------------- rule pack

class OrthopaedicRuleTests(unittest.TestCase):
    def test_nsaid_plus_anticoagulant_is_major(self):
        hits = oi.evaluate(["布洛芬", "华法林"])
        self.assertEqual(hits[0]["rule_id"], "ORTHO-001")
        self.assertEqual(hits[0]["severity"], "major")

    def test_triple_whammy_needs_all_three_classes(self):
        self.assertFalse(any(h["rule_id"] == "ORTHO-002" for h in oi.evaluate(["ibuprofen", "enalapril"])))
        self.assertTrue(any(h["rule_id"] == "ORTHO-002" for h in oi.evaluate(["ibuprofen", "enalapril", "furosemide"])))

    def test_opioid_plus_sedative_is_contraindicated(self):
        hits = oi.evaluate(["羟考酮", "阿普唑仑"])
        self.assertEqual(hits[0]["severity"], "contraindicated")
        self.assertTrue(oi.blocking(hits))

    def test_tramadol_serotonin_rule_fires_for_ssri_and_maoi(self):
        self.assertTrue(any(h["rule_id"] == "ORTHO-006" for h in oi.evaluate(["曲马多", "舍曲林"])))
        self.assertTrue(any(h["rule_id"] == "ORTHO-006" for h in oi.evaluate(["tramadol", "linezolid"])))

    def test_condition_gated_rules_stay_quiet_without_the_condition(self):
        self.assertFalse(any(h["rule_id"] == "ORTHO-010" for h in oi.evaluate(["阿仑膦酸钠"])))
        self.assertTrue(any(h["rule_id"] == "ORTHO-010" for h in oi.evaluate(["阿仑膦酸钠"], ["renal_impairment"])))
        self.assertTrue(any(h["rule_id"] == "ORTHO-013" for h in oi.evaluate(["romosozumab"], ["recent_mi_or_stroke"])))

    def test_bisphosphonate_and_calcium_absorption_rule(self):
        self.assertTrue(any(h["rule_id"] == "ORTHO-009" for h in oi.evaluate(["阿仑膦酸钠", "碳酸钙"])))

    def test_colchicine_with_cyp3a4_inhibitor_is_contraindicated(self):
        hits = oi.evaluate(["秋水仙碱", "克拉霉素"])
        self.assertEqual(hits[0]["rule_id"], "ORTHO-017")
        self.assertEqual(hits[0]["severity"], "contraindicated")

    def test_neuraxial_anaesthesia_rule_fires_for_planned_surgery(self):
        self.assertTrue(any(h["rule_id"] == "ORTHO-015"
                            for h in oi.evaluate(["利伐沙班"], ["planned_neuraxial_anesthesia"])))

    def test_unrelated_medications_produce_no_findings(self):
        self.assertEqual(oi.evaluate(["茯苓", "vitamin c"]), [])

    def test_findings_are_sorted_most_severe_first(self):
        hits = oi.evaluate(["羟考酮", "阿普唑仑", "布洛芬", "华法林"])
        severities = [h["severity"] for h in hits]
        self.assertEqual(severities, sorted(severities, key=lambda s: oi.SEVERITY_ORDER[s]))

    def test_every_rule_carries_mechanism_and_management(self):
        for rule in oi.ORTHO_RULES:
            with self.subTest(rule=rule.rule_id):
                self.assertTrue(rule.mechanism)
                self.assertTrue(rule.management)
                self.assertIn(rule.severity, oi.SEVERITY_ORDER)

    def test_bilingual_class_matching(self):
        self.assertIn("nsaid", oi.classify("布洛芬缓释胶囊 0.3g"))
        self.assertIn("nsaid", oi.classify("Ibuprofen 400 mg tablet"))
        self.assertIn("anticoagulant", oi.classify("华法林钠片"))


# ----------------------------------------------------- clinical integration

class KnowledgeDrivenBehaviourTests(unittest.TestCase):
    def test_licensed_guideline_replaces_stub_evidence(self):
        with TemporaryDirectory() as tmp:
            store = KnowledgeStore(Path(tmp) / "k.db", LicensePolicy(RESEARCH))
            store.add_guideline(
                "vadod_cpg", "VA-LBP-2022", "VA/DoD Low Back Pain CPG",
                topic="low back pain differential", url="https://www.healthquality.va.gov/",
                published="2022-02-01", version="3.0", evidence_grade="Strong for",
                summary="low back pain differential and red flag screening",
                recommendations=["Screen for red flags before imaging"],
            )
            out = YaobiGraphRunner(ToolRegistry(knowledge=store, deid_key=TEST_KEY)).run(
                ClinicalRunState("腰痛3月，久坐加重", role="physician")
            )
            store.close()
        levels = {e.source: e.level for e in out.evidence.values()}
        self.assertEqual(levels["clinical_guideline_search"], "guideline_or_standard")
        self.assertNotEqual(levels["clinical_guideline_search"], "stub_not_for_clinical_use")

    def test_no_store_configured_keeps_guidelines_as_stub(self):
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="physician"))
        levels = {e.source: e.level for e in out.evidence.values()}
        self.assertEqual(levels["clinical_guideline_search"], "stub_not_for_clinical_use")
        self.assertFalse(out.outputs["run_meta"]["knowledge"]["configured"])

    def test_licensed_dose_range_gates_the_prescription_draft(self):
        """The pharmacopoeia range decides the draft, not the local table."""
        records = [case_with_doses(i, 30.0) for i in range(6)]
        with TemporaryDirectory() as tmp:
            store = KnowledgeStore(Path(tmp) / "k.db", attested("chp_2025"))
            for herb in FORMULA_HERBS + BLOOD_STASIS_EXTRA:
                store.add_dose_range("chp_2025", herb, 3.0, 9.0, basis="《中国药典》2025 一部", version="2025")
            out = YaobiGraphRunner(
                ToolRegistry(records=records, knowledge=store, deid_key=TEST_KEY)
            ).run(physician_state(), allow_prescription=True)
            store.close()
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertTrue(any("超出授权" in i or "异常值" in i for i in out.safety_issues), out.safety_issues)

    def test_licensed_dose_range_allows_an_in_range_draft_with_provenance(self):
        records = [case_with_doses(i, 9.0) for i in range(6)]
        with TemporaryDirectory() as tmp:
            store = KnowledgeStore(Path(tmp) / "k.db", attested("chp_2025"))
            for herb in FORMULA_HERBS + BLOOD_STASIS_EXTRA:
                store.add_dose_range("chp_2025", herb, 3.0, 15.0, basis="《中国药典》2025 一部", version="2025")
            out = YaobiGraphRunner(
                ToolRegistry(records=records, knowledge=store, deid_key=TEST_KEY)
            ).run(physician_state(), allow_prescription=True)
            store.close()
        self.assertEqual(out.release_status, "draft_for_physician", out.safety_issues)
        citations = citation_bundle(out)
        self.assertTrue(any(c.get("source_id") == "chp_2025" for c in citations), citations)

    def test_medication_interaction_escalates_a_routine_run(self):
        state = ClinicalRunState("腰痛3月，久坐加重", role="patient")
        state.facts["medications"] = ["布洛芬 0.3g bid", "华法林 3mg qd"]
        out = YaobiGraphRunner().run(state)
        self.assertEqual(out.release_status, "needs_examination")
        self.assertTrue(any("用药安全" in i for i in out.safety_issues))
        view = render(out, "patient")
        self.assertTrue(view["medication_warnings"])
        self.assertIn("what_to_do", view["medication_warnings"][0])

    def test_missing_medication_list_is_recorded_as_an_information_gap(self):
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
        self.assertIn("当前用药清单", out.missing_information)
        self.assertFalse(out.outputs["medication_safety"]["reviewed"])

    def test_licensed_ddi_database_findings_reach_the_run(self):
        with TemporaryDirectory() as tmp:
            store = KnowledgeStore(Path(tmp) / "k.db", attested("drugbank"))
            store.add_interaction("drugbank", "cyclobenzaprine", "duloxetine", "major",
                                  management="monitor for serotonin syndrome", version="2025.1")
            state = ClinicalRunState("腰痛3月，久坐加重", role="physician")
            state.facts["medications"] = ["cyclobenzaprine", "duloxetine"]
            out = YaobiGraphRunner(ToolRegistry(knowledge=store, deid_key=TEST_KEY)).run(state)
            store.close()
        findings = out.outputs["medication_safety"]["findings"]
        self.assertTrue(any(f.get("severity") == "major" for f in findings), findings)

    def test_patient_view_never_leaks_the_licence_policy_or_ledger(self):
        state = ClinicalRunState("腰痛3月，久坐加重", role="patient")
        state.facts["medications"] = ["布洛芬", "华法林"]
        view = render(YaobiGraphRunner().run(state), "patient")
        self.assertNotIn("citations", view)
        self.assertNotIn("evidence_ledger", view)

    def test_medication_safety_agent_cannot_reach_prescription_tools(self):
        from yaobi_harness.skills.loader import SkillRegistry

        manifest = Path(__file__).resolve().parents[1] / "yaobi_harness" / "skills" / "manifest.yaml"
        ok, problems = SkillRegistry.from_file(manifest).enforce(
            "yaobi.medication_safety", "physician", ["herb_dose_distribution"]
        )
        self.assertFalse(ok)
        self.assertTrue(any("forbidden" in p for p in problems))


# ------------------------------------------------------------ web connectors

OPENFDA_LABEL = {
    "results": [{
        "set_id": "set-123", "version": "20", "effective_time": "20250617",
        "openfda": {"brand_name": ["Coumadin"], "generic_name": ["WARFARIN SODIUM"], "rxcui": ["855288"]},
        "drug_interactions": ["Concomitant use of drugs that increase bleeding risk."],
        "contraindications": ["Pregnancy; haemorrhagic tendencies."],
        "dosage_and_administration": ["Individualize dosing based on INR."],
    }]
}
DAILYMED_SPL = {"data": [{"setid": "set-123", "title": "COUMADIN- warfarin sodium tablet", "spl_version": "20",
                          "published_date": "Jun 17, 2025"}]}
RXNORM_ID = {"idGroup": {"rxnormId": ["11289"]}}
RXNORM_PROP = {"propConceptGroup": {"propConcept": [{"propName": "RxNormName", "propValue": "warfarin"}]}}
RXCLASS = {"rxclassDrugInfoList": {"rxclassDrugInfo": [
    {"rxclassMinConceptItem": {"classId": "B01AA03", "className": "Warfarin"}}
]}}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if "/drug/label.json" in self.path:
            payload = OPENFDA_LABEL
        elif "/spls.json" in self.path:
            payload = DAILYMED_SPL
        elif "/rxcui.json" in self.path:
            payload = RXNORM_ID
        elif "/property.json" in self.path:
            payload = RXNORM_PROP
        elif "/rxclass/" in self.path:
            payload = RXCLASS
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class WebConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_openfda_ingest_stores_sections_with_label_version(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            connector = OpenFdaConnector(store.policy, endpoint=f"{self.base}/drug/label.json")
            report = connector.ingest(store, ["warfarin sodium"])
            self.assertEqual(report["sections_stored"], 3)
            sections = store.label_sections("warfarin sodium")
        self.assertEqual({s["section"] for s in sections},
                         {"drug_interactions", "contraindications", "dosage_and_administration"})
        self.assertEqual(sections[0]["label_version"], "20")
        self.assertEqual(sections[0]["effective_time"], "20250617")

    def test_dailymed_ingest_records_the_setid(self):
        with KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            connector = DailyMedConnector(store.policy, endpoint=self.base)
            self.assertEqual(connector.ingest(store, ["warfarin"])["labels_stored"], 1)
            sections = store.label_sections("warfarin", ["label_reference"])
        self.assertIn("set-123", sections[0]["provenance"]["url"])

    def test_rxnorm_normalizes_a_name_to_rxcui_and_atc(self):
        connector = RxNormConnector(LicensePolicy(RESEARCH), endpoint=self.base)
        resolved = connector.normalize("Coumadin")
        self.assertEqual(resolved["rxcui"], "11289")
        self.assertEqual(resolved["normalized_name"], "warfarin")
        self.assertEqual(resolved["atc"][0]["atc_code"], "B01AA03")

    def test_connector_caches_responses(self):
        with TemporaryDirectory() as tmp, KnowledgeStore(":memory:", LicensePolicy(RESEARCH)) as store:
            connector = OpenFdaConnector(store.policy, endpoint=f"{self.base}/drug/label.json", cache_dir=tmp)
            connector.fetch_label("warfarin sodium")
            cached = list(Path(tmp).rglob("*.json"))
            self.assertTrue(cached)

    def test_credentialed_connector_refuses_to_construct_without_attestation(self):
        from yaobi_harness.knowledge.connectors.web import NiceConnector

        with self.assertRaises(LicenseError):
            NiceConnector(LicensePolicy(RESEARCH))
        NiceConnector(attested("nice"), endpoint=self.base, api_key="k")

    def test_drug_normalize_falls_back_to_offline_class_matching(self):
        result = ToolRegistry(deid_key=TEST_KEY).drug_normalize("布洛芬缓释胶囊")
        self.assertEqual(result.data["mode"], "offline_class_match")
        self.assertIn("nsaid", result.data["local_classes"])


if __name__ == "__main__":
    unittest.main()
