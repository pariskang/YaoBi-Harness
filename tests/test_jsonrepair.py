"""Tests for the model-output JSON repairer.

The value of this module is measured in one number: how much real model output the
harness can use. Every case below is a malformation observed in practice, and each
one used to cost the whole model-driven path — the run fell back to rules, and the
operator saw ``planner_mode: rule`` with no explanation.

The other half of the suite matters just as much. A repairer that silently invents
a reading is worse than one that fails, because a parse failure falls back to the
deterministic path while a wrong parse puts fabricated content into a clinical
record. So the corruption tests — a brace inside a string, an apostrophe in a
sentence, a URL that looks like a comment — are the ones that pin the design.
"""

from __future__ import annotations

import json
import unittest

from yaobi_harness.llm.base import extract_json, extract_json_with_repairs
from yaobi_harness.llm.jsonrepair import loads, loads_with_repairs, repair


class WellFormedInputTests(unittest.TestCase):
    def test_valid_json_is_returned_untouched_and_unrepaired(self):
        value, repairs = loads_with_repairs('{"a": 1, "b": [2, 3]}')
        self.assertEqual(value, {"a": 1, "b": [2, 3]})
        self.assertEqual(repairs, [], "a clean payload must not be reported as repaired")

    def test_chinese_content_needs_no_repair(self):
        payload = {"分诊": "routine", "理由": "1个月病程，无红旗；乏力多为睡眠不足"}
        value, repairs = loads_with_repairs(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(value, payload)
        self.assertEqual(repairs, [])


class FramingTests(unittest.TestCase):
    """Fences and prose: the model answered correctly and wrapped it."""

    def test_fenced_with_language_tag(self):
        self.assertEqual(loads('```json\n{"a": 1}\n```'), {"a": 1})

    def test_fenced_without_language_tag(self):
        self.assertEqual(loads('```\n{"a": 1}\n```'), {"a": 1})

    def test_unclosed_fence(self):
        """What a response truncated mid-fence looks like."""
        self.assertEqual(loads('```json\n{"a": 1}'), {"a": 1})

    def test_prose_before_and_after(self):
        self.assertEqual(loads('好的，结果如下：\n{"a": 1}\n以上仅供参考。'), {"a": 1})

    def test_prose_with_a_fence_inside_it(self):
        self.assertEqual(loads('这是结果：\n```json\n{"a": 1}\n```\n完毕'), {"a": 1})


class SyntaxSlipTests(unittest.TestCase):
    """Slips with exactly one possible reading, so repairing them guesses nothing."""

    def test_trailing_comma_in_an_object(self):
        self.assertEqual(loads('{"a": 1,}'), {"a": 1})

    def test_trailing_comma_in_an_array(self):
        self.assertEqual(loads("[1, 2, 3,]"), [1, 2, 3])

    def test_doubled_comma(self):
        self.assertEqual(loads('{"a": 1,, "b": 2}'), {"a": 1, "b": 2})

    def test_single_quoted_strings(self):
        self.assertEqual(loads("{'a': 1, 'b': 'x'}"), {"a": 1, "b": "x"})

    def test_unquoted_keys(self):
        self.assertEqual(loads("{a: 1, b_c: 2}"), {"a": 1, "b_c": 2})

    def test_python_literals(self):
        self.assertEqual(loads('{"a": True, "b": False, "c": None}'),
                         {"a": True, "b": False, "c": None})

    def test_line_comment(self):
        self.assertEqual(loads('{"a": 1, // 说明\n "b": 2}'), {"a": 1, "b": 2})

    def test_block_comment(self):
        self.assertEqual(loads('{"a": 1, /* 说明 */ "b": 2}'), {"a": 1, "b": 2})

    def test_typographic_quotes_from_a_chinese_ime(self):
        self.assertEqual(loads('{“a”: “x”}'), {"a": "x"})

    def test_full_width_comma_and_colon(self):
        self.assertEqual(loads('{"a"：1，"b"：2}'), {"a": 1, "b": 2})

    def test_raw_newline_inside_a_string(self):
        self.assertEqual(loads('{"a": "第一行\n第二行"}'), {"a": "第一行\n第二行"})

    def test_a_bare_word_value_is_read_as_a_string(self):
        self.assertEqual(loads('{"sex": male}'), {"sex": "male"})

    def test_everything_at_once(self):
        text = "```\n{'tasks': [{'id': 'S1', agent: None,},], /* 注释 */}\n```"
        self.assertEqual(loads(text), {"tasks": [{"id": "S1", "agent": None}]})


class TruncationTests(unittest.TestCase):
    """The highest-value repair: a response cut off by ``max_tokens``.

    A truncated answer usually contains everything that matters, so recovering it
    is the difference between using the model's work and discarding it.
    """

    def test_unterminated_string(self):
        self.assertEqual(loads('{"a": "hello'), {"a": "hello"})

    def test_unclosed_array_and_object(self):
        self.assertEqual(loads('{"a": 1, "b": [2, 3'), {"a": 1, "b": [2, 3]})

    def test_deeply_nested_truncation(self):
        self.assertEqual(loads('{"a": {"b": {"c": [1, {"d": "tex'),
                               {"a": {"b": {"c": [1, {"d": "tex"}]}}})

    def test_truncated_right_after_a_comma(self):
        self.assertEqual(loads('{"a": 1, "b": 2,'), {"a": 1, "b": 2})

    def test_a_truncated_plan_still_yields_its_tasks(self):
        """The case that motivated this: a planner cut off mid-task."""
        text = '{"reasoning": "先筛红旗", "tasks": [{"task_id": "P1", "agent": "IntakeAgent"}, {"task_id": "P2", "agent": "Biomed'
        value = loads(text)
        self.assertEqual(len(value["tasks"]), 2)
        self.assertEqual(value["tasks"][0]["agent"], "IntakeAgent")

    def test_truncation_is_reported_as_such(self):
        _, repairs = loads_with_repairs('{"a": "hello')
        self.assertIn("unterminated", repairs)
        self.assertIn("unclosed", repairs)

    def test_a_trailing_fragment_is_dropped_to_recover_the_complete_items(self):
        """``{"c"`` — a key with no value — cannot be closed, but the two complete
        objects before it are plainly there. Dropping from the end never invents."""
        value, repairs = loads_with_repairs('[{"a": 1}, {"b": 2}, {"c"')
        self.assertEqual(value, [{"a": 1}, {"b": 2}])
        self.assertIn("dropped_incomplete_tail", repairs)

    def test_a_plan_truncated_at_a_bare_key_keeps_its_finished_tasks(self):
        value = loads('{"tasks": [{"task_id": "P1", "agent": "IntakeAgent"}, {"task_id"')
        self.assertEqual(value, {"tasks": [{"task_id": "P1", "agent": "IntakeAgent"}]})

    def test_truncation_right_after_a_colon(self):
        self.assertEqual(loads('{"a": 1, "b": 2, "c": '), {"a": 1, "b": 2})

    def test_the_span_end_is_found_by_depth_not_by_the_last_bracket(self):
        """``rfind('}')`` lands on the brace closing the *first* element, which
        silently deleted everything after it."""
        value = loads('{"tasks": [{"id": "a}b"}, {"id": "c')
        self.assertEqual(value, {"tasks": [{"id": "a}b"}, {"id": "c"}]})


class NoCorruptionTests(unittest.TestCase):
    """A wrong parse is worse than no parse. These pin that boundary."""

    def test_a_brace_inside_a_string_is_content(self):
        self.assertEqual(loads('{"a": "{not structure}"}'), {"a": "{not structure}"})

    def test_a_bracket_inside_a_string_is_content(self):
        self.assertEqual(loads('{"a": "[1,2]"}'), {"a": "[1,2]"})

    def test_an_apostrophe_inside_a_string_is_content(self):
        self.assertEqual(loads('{"note": "it\'s fine"}'), {"note": "it's fine"})

    def test_a_double_slash_inside_a_string_is_not_a_comment(self):
        self.assertEqual(loads('{"u": "https://example.com/a"}'),
                         {"u": "https://example.com/a"})

    def test_a_comma_inside_a_string_is_content(self):
        self.assertEqual(loads('{"a": "1,2,3"}'), {"a": "1,2,3"})

    def test_prose_after_a_complete_object_is_discarded_not_parsed(self):
        self.assertEqual(loads('{"a": 1}\n以上仅供参考。'), {"a": 1})

    def test_two_concatenated_objects_yield_the_first(self):
        """A reading, not a guess: merging them would invent a document."""
        self.assertEqual(loads('{"a": 1}{"b": 2}'), {"a": 1})

    def test_brackets_inside_strings_do_not_change_depth(self):
        self.assertEqual(loads('{"a": "[[[", "b": 2}'), {"a": "[[[", "b": 2})

    def test_typographic_quotes_inside_a_normal_string_survive(self):
        """Only a string *opened* by a smart quote may be closed by its partner."""
        self.assertEqual(loads('{"a": "他说“好”，然后走了"}'), {"a": "他说“好”，然后走了"})

    def test_an_escaped_quote_is_preserved(self):
        self.assertEqual(loads('{"a": "say \\"hi\\""}'), {"a": 'say "hi"'})

    def test_a_backslash_at_the_end_of_a_string(self):
        self.assertEqual(loads('{"path": "C:\\\\tmp"}'), {"path": "C:\\tmp"})


class RefusalTests(unittest.TestCase):
    """Ambiguity must fail. Inventing a reading would fabricate clinical content."""

    def test_prose_with_no_json_at_all(self):
        self.assertIsNone(loads("考虑腰椎间盘突出，建议做 MRI。"))

    def test_empty_and_whitespace(self):
        self.assertIsNone(loads(""))
        self.assertIsNone(loads("   \n\t "))

    def test_a_key_with_no_colon(self):
        self.assertIsNone(loads('{"a" 1}'))

    def test_no_span_means_no_repaired_text(self):
        text, _ = repair("完全没有 JSON")
        self.assertIsNone(text)

    def test_the_repairer_never_executes_anything(self):
        """``ast.literal_eval`` is the only eval-adjacent path, and it evaluates
        nothing. A payload shaped like code must not run."""
        for hostile in (
            '{"a": __import__("os").system("echo pwned")}',
            "__import__('os').system('echo pwned')",
            '{"a": eval("1+1")}',
        ):
            with self.subTest(hostile=hostile):
                extract_json(hostile, None)  # must not raise, must not execute


class ExtractJsonIntegrationTests(unittest.TestCase):
    def test_extract_json_uses_the_repairer(self):
        self.assertEqual(extract_json('```json\n{"a": 1,}\n```'), {"a": 1})

    def test_repairs_are_reported_to_the_caller(self):
        value, repairs = extract_json_with_repairs('```json\n{"a": 1,}\n```')
        self.assertEqual(value, {"a": 1})
        self.assertIn("fence", repairs)
        self.assertIn("trailing_comma", repairs)

    def test_a_clean_payload_reports_no_repairs(self):
        _, repairs = extract_json_with_repairs('{"a": 1}')
        self.assertEqual(repairs, [])

    def test_the_default_is_returned_when_nothing_parses(self):
        sentinel = object()
        self.assertIs(extract_json("没有 JSON", sentinel), sentinel)
        self.assertIs(extract_json("", sentinel), sentinel)

    def test_a_python_tuple_literal_still_parses(self):
        """The repairer does not rewrite tuples, so ``literal_eval`` still earns
        its place as the last resort."""
        value, repairs = extract_json_with_repairs("{'a': (1, 2)}")
        self.assertEqual(value, {"a": (1, 2)})
        self.assertIn("python_literal", repairs)


class ScaleTests(unittest.TestCase):
    def test_a_large_payload_survives_a_repair(self):
        payload = {"tasks": [{"task_id": f"P{i}", "agent": "IntakeAgent"} for i in range(400)]}
        text = json.dumps(payload, ensure_ascii=False)[:-1] + ",}"  # trailing comma
        self.assertEqual(len(loads(text)["tasks"]), 400)

    def test_deep_nesting_does_not_recurse(self):
        """The scanner is iterative, so depth is bounded by memory, not the stack."""
        text = "{" + '"a": {' * 300
        value = loads(text)
        self.assertIsInstance(value, dict)


if __name__ == "__main__":
    unittest.main()
