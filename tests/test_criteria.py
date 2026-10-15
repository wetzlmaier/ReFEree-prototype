"""
Unit tests for ReFEree criteria (C1-C4).
"""

import unittest
from referee.criteria import CriterionType, CRITERIA, CRITERIA_ORDER


class TestCriteria(unittest.TestCase):
    def test_all_four_criteria_present(self):
        self.assertEqual(len(CRITERIA_ORDER), 4)
        self.assertIn(CriterionType.C1_NAME, CRITERIA)
        self.assertIn(CriterionType.C2_TYPE, CRITERIA)
        self.assertIn(CriterionType.C3_FUNCTIONALITY, CRITERIA)
        self.assertIn(CriterionType.C4_CONTEXT_IRRELEVANT, CRITERIA)

    def test_criteria_names_and_definitions(self):
        c1 = CRITERIA[CriterionType.C1_NAME]
        self.assertEqual(c1.name, "Name Inconsistency")
        self.assertIn("does not match the actual identifier", c1.explanation)

        c2 = CRITERIA[CriterionType.C2_TYPE]
        self.assertEqual(c2.name, "Type Inconsistency")
        self.assertIn("return type or variable type", c2.explanation)

        c3 = CRITERIA[CriterionType.C3_FUNCTIONALITY]
        self.assertEqual(c3.name, "Functionality Inconsistency")
        self.assertIn("functionality or purpose described in the summary", c3.explanation)

        c4 = CRITERIA[CriterionType.C4_CONTEXT_IRRELEVANT]
        self.assertEqual(c4.name, "Context Irrelevant")
        self.assertIn("unnecessary or irrelevant to the input code", c4.explanation)

    def test_weights(self):
        # Default equal weights (1:1:1:1)
        for c in CRITERIA.values():
            self.assertEqual(c.default_weight, 1.0)

        # Tuned weights from Table 14: 0.6 : 1.2 : 1.2 : 1.0
        self.assertEqual(CRITERIA[CriterionType.C1_NAME].tuned_weight, 0.6)
        self.assertEqual(CRITERIA[CriterionType.C2_TYPE].tuned_weight, 1.2)
        self.assertEqual(CRITERIA[CriterionType.C3_FUNCTIONALITY].tuned_weight, 1.2)
        self.assertEqual(CRITERIA[CriterionType.C4_CONTEXT_IRRELEVANT].tuned_weight, 1.0)


if __name__ == "__main__":
    unittest.main()

