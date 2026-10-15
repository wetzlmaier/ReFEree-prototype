"""
Unit tests for ReFEreeEvaluator, sentence segmentation, and scoring formula.
"""

import unittest
from referee.criteria import CriterionType, CRITERIA
from referee.evaluator import ReFEreeEvaluator
from referee.llm import MockLLMClient


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.mock_client = MockLLMClient(default_score=1)
        self.evaluator = ReFEreeEvaluator(llm_client=self.mock_client)

    def test_sentence_tokenization(self):
        summary = (
            "The iter_slice function slices bytes into chunks. "
            "It yields chunks along with a boolean flag. "
            "Finally, it returns."
        )
        segments = self.evaluator.tokenize_summary(summary)
        self.assertEqual(len(segments), 3)
        self.assertIn("The iter_slice function slices bytes into chunks.", segments[0])
        self.assertIn("It yields chunks along with a boolean flag.", segments[1])
        self.assertIn("Finally, it returns.", segments[2])

    def test_scoring_perfect_consistency(self):
        code = "def foo(): pass"
        summary = "Function foo does nothing. It simply passes."
        # All 1s by default
        result = self.evaluator.evaluate(input_code=code, summary=summary)
        self.assertEqual(result.overall_score, 1.0)
        self.assertEqual(result.weighted_score, 1.0)
        self.assertEqual(len(result.segments), 2)
        for seg in result.segments:
            self.assertTrue(seg.is_consistent())
            self.assertEqual(seg.segment_score, 1.0)

    def test_scoring_with_inconsistency(self):
        code = "def foo(): pass"
        summary = "Function foo does nothing. Function bar calculates pi."

        # Simulate violation on segment 2: C1 (bar is wrong name) and C3 (wrong functionality)
        self.mock_client.set_rule("function bar", "name inconsistency", 0)
        self.mock_client.set_rule("function bar", "functionality inconsistency", 0)

        result = self.evaluator.evaluate(input_code=code, summary=summary)
        # 2 segments, 4 criteria = 8 total evaluations.
        # Segment 1: all 4 pass -> 4/4
        # Segment 2: C1=0, C2=1, C3=0, C4=1 -> 2/4 = 0.5
        # Total passes = 4 + 2 = 6.
        # Overall Score = 6 / (2 * 4) = 0.75
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].segment_score, 1.0)
        self.assertEqual(result.segments[1].segment_score, 0.5)
        self.assertIn(CriterionType.C1_NAME, result.segments[1].violated_criteria)
        self.assertIn(CriterionType.C3_FUNCTIONALITY, result.segments[1].violated_criteria)
        self.assertAlmostEqual(result.overall_score, 0.75, places=4)

    def test_prompt_formatting(self):
        c1 = CRITERIA[CriterionType.C1_NAME]
        sys_prompt = self.evaluator.format_system_prompt(c1)
        self.assertIn("Name Inconsistency", sys_prompt)
        self.assertIn("Evaluation Criteria:", sys_prompt)
        self.assertIn("Evaluation Steps:", sys_prompt)

        user_prompt = self.evaluator.format_user_prompt(
            input_code="def f(): pass",
            related_information="# helper #\ndef helper(): pass",
            segment="Function f does nothing.",
        )
        self.assertIn("## CODE:", user_prompt)
        self.assertIn("(Related Information) # helper #", user_prompt)
        self.assertIn("(Input Code) def f(): pass", user_prompt)
        self.assertIn("## SUMMARY TEXT: Function f does nothing.", user_prompt)
        self.assertIn("## SCORE (score only):", user_prompt)


if __name__ == "__main__":
    unittest.main()

