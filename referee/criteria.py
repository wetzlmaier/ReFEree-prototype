"""
Definitions of the four factual inconsistency criteria (C1-C4) from Table 1 and Table 14
of the ReFEree publication.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CriterionType(str, Enum):
    """Enumeration of the 4 factual inconsistency criteria in ReFEree."""
    C1_NAME = "C1"
    C2_TYPE = "C2"
    C3_FUNCTIONALITY = "C3"
    C4_CONTEXT_IRRELEVANT = "C4"


@dataclass(frozen=True)
class Criterion:
    """Represents an individual factual inconsistency criterion."""
    id: CriterionType
    name: str
    explanation: str
    default_weight: float = 1.0
    tuned_weight: float = 1.0


# Exact definitions from Table 1 of Bae et al. (ACL 2026)
CRITERIA: Dict[CriterionType, Criterion] = {
    CriterionType.C1_NAME: Criterion(
        id=CriterionType.C1_NAME,
        name="Name Inconsistency",
        explanation=(
            "The name of a function, class, or variable in the summary does not match "
            "the actual identifier in the input code or mistakenly refers to a different "
            "entity with the same name in the project."
        ),
        default_weight=1.0,
        tuned_weight=0.6,
    ),
    CriterionType.C2_TYPE: Criterion(
        id=CriterionType.C2_TYPE,
        name="Type Inconsistency",
        explanation=(
            "The described function’s return type or variable type in the summary is "
            "inconsistent with the actual type in the input code or in directly dependent functions."
        ),
        default_weight=1.0,
        tuned_weight=1.2,
    ),
    CriterionType.C3_FUNCTIONALITY: Criterion(
        id=CriterionType.C3_FUNCTIONALITY,
        name="Functionality Inconsistency",
        explanation=(
            "The functionality or purpose described in the summary does not accurately "
            "reflect what the input code or its dependent functions actually implement. "
            "This often arises when dependency relationships are ignored or misinterpreted, "
            "leading to an incorrect description of behavior."
        ),
        default_weight=1.0,
        tuned_weight=1.2,
    ),
    CriterionType.C4_CONTEXT_IRRELEVANT: Criterion(
        id=CriterionType.C4_CONTEXT_IRRELEVANT,
        name="Context Irrelevant",
        explanation=(
            "The summary contains content that is unnecessary or irrelevant to the input "
            "code or relevant information, such as descriptions of unrelated entities or "
            "overly generalized dependency context that does not contribute to understanding "
            "the function."
        ),
        default_weight=1.0,
        tuned_weight=1.0,
    ),
}

CRITERIA_ORDER: List[CriterionType] = [
    CriterionType.C1_NAME,
    CriterionType.C2_TYPE,
    CriterionType.C3_FUNCTIONALITY,
    CriterionType.C4_CONTEXT_IRRELEVANT,
]

