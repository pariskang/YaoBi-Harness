import unittest

from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.state import ClinicalRunState


class UnittestDiscoverySmokeTest(unittest.TestCase):
    def test_unittest_discovery_runs_project_smoke(self):
        out = YaobiGraphRunner().run(ClinicalRunState("突发胸闷、喘不上气、脸色苍白", role="patient"))
        self.assertEqual(out.risk_mode, "urgent")
        self.assertEqual(out.release_status, "urgent_action_plan")
