"""
ReFEree Core Evaluator Implementation.

Decomposes code summaries into sentence segments, evaluates each segment
against the four factual inconsistency criteria (C1-C4) using LLM prompting
with 1-hop code context, and aggregates the scores according to Section 3.3
and Appendix A.3 of the ReFEree publication.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

from referee.criteria import CRITERIA, CRITERIA_ORDER, Criterion, CriterionType
from referee.search import CodeRelatedInfoSearcher, RelatedInformationItem
from referee.llm import BaseLLMClient, MockLLMClient


def _empty_violated() -> List[CriterionType]:
    return []


@dataclass
class SegmentResult:
    """Evaluation result for an individual sentence segment."""
    segment_index: int
    text: str
    criterion_scores: Dict[CriterionType, int]  # 1 (consistent) or 0 (inconsistent)
    violated_criteria: List[CriterionType] = field(default_factory=_empty_violated)
    segment_score: float = 0.0  # Mean of criteria scores for this segment (0.0 to 1.0)

    def is_consistent(self) -> bool:
        """Returns True if all 4 criteria evaluated to 1 (no inconsistency)."""
        return len(self.violated_criteria) == 0


@dataclass
class EvaluationResult:
    """Full ReFEree evaluation result for a code summary."""
    input_code: str
    summary: str
    related_information: str
    segments: List[SegmentResult]
    overall_score: float  # Unweighted score according to Section 3.3 equation (0.0 to 1.0)
    weighted_score: float  # Optional weighted score according to Appendix C.5
    criterion_summary: Dict[str, Dict[str, Any]]

    def summary_report(self) -> str:
        """Generates a human-readable explanation matching the paper's qualitative format."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("ReFEree Factual Consistency Evaluation Report")
        lines.append("=" * 60)
        lines.append(f"Overall Consistency Score: {self.overall_score:.4f} (0.0 - 1.0)")
        lines.append(f"Weighted Consistency Score: {self.weighted_score:.4f} (Table 14 weights)")
        lines.append(f"Total Segments: {len(self.segments)}")
        lines.append("-" * 60)
        lines.append("Segment-Level Breakdown:")
        for seg in self.segments:
            violation_str = (
                f" [VIOLATIONS: {', '.join(c.value for c in seg.violated_criteria)}]"
                if seg.violated_criteria
                else " [PASS]"
            )
            lines.append(f"\n[Segment {seg.segment_index}] Score: {seg.segment_score:.2f}{violation_str}")
            lines.append(f"  Text: {seg.text}")
            crit_details = [
                f"{c.value}={seg.criterion_scores.get(c, 0)}"
                for c in CRITERIA_ORDER
            ]
            lines.append(f"  Criteria: {', '.join(crit_details)}")

        lines.append("\n" + "-" * 60)
        lines.append("Criteria Breakdown:")
        for crit_key, stats in self.criterion_summary.items():
            lines.append(
                f"  {stats['name']} ({crit_key}): "
                f"Consistent in {stats['consistent_segments']}/{stats['total_segments']} segments "
                f"({stats['consistency_rate']:.1%})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You will be given one summary text written for a source code. Your task is to evaluate the summary from {criterion} aspect. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{criterion} (1 or 0) – {explanation}

Evaluation Steps:
1. Read the CODE carefully and understand its main intent.
2. Read the code summary text and check if it accurately describes the code.
3. Evaluate whether {criterion} exists, where “1” means “{criterion} does not exist” and “0” means “{criterion} exists” based on the Evaluation Criteria."""

USER_PROMPT_TEMPLATE = """## CODE:
(Related Information) {related_information}
(Input Code) {input_code}
## SUMMARY TEXT: {segment}
## SCORE (score only):"""


class ReFEreeEvaluator:
    """
    Main ReFEree evaluator implementing the reference-free, fine-grained evaluation pipeline.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        searcher: Optional[CodeRelatedInfoSearcher] = None,
    ):
        self.llm_client = llm_client or MockLLMClient()
        self.searcher = searcher or CodeRelatedInfoSearcher()

    def tokenize_summary(self, summary: str) -> List[str]:
        """
        Splits summary into sentence-level segments using NLTK sentence tokenizer,
        with fallback to robust regex segmentation.
        """
        cleaned = summary.strip()
        if not cleaned:
            return []

        try:
            import nltk
            import nltk.tokenize
            sent_tokenize_fn: Any = getattr(nltk.tokenize, "sent_tokenize")
            try:
                raw_sentences: Any = sent_tokenize_fn(cleaned)
                if isinstance(raw_sentences, list):
                    sentences = cast(List[str], raw_sentences)
                    if sentences:
                        return [s.strip() for s in sentences if s.strip()]
            except LookupError:
                download_fn: Any = getattr(nltk, "download")
                _ = download_fn("punkt", quiet=True)
                _ = download_fn("punkt_tab", quiet=True)
                raw_sentences2: Any = sent_tokenize_fn(cleaned)
                if isinstance(raw_sentences2, list):
                    sentences2 = cast(List[str], raw_sentences2)
                    if sentences2:
                        return [s.strip() for s in sentences2 if s.strip()]
        except Exception:
            pass

        # Robust regex fallback
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9`\"'“‘])"
        parts = re.split(pattern, cleaned)
        return [p.strip() for p in parts if p.strip()]

    def format_system_prompt(self, criterion: Criterion) -> str:
        """Formats the system prompt for a criterion according to Appendix A.3."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            criterion=criterion.name,
            explanation=criterion.explanation,
        )

    def format_user_prompt(
        self,
        input_code: str,
        related_information: str,
        segment: str,
    ) -> str:
        """Formats the user prompt according to Appendix A.3."""
        rel_info_text = related_information.strip() if related_information else "None"
        return USER_PROMPT_TEMPLATE.format(
            related_information=rel_info_text,
            input_code=input_code.strip(),
            segment=segment.strip(),
        )

    def evaluate(
        self,
        input_code: str,
        summary: str,
        related_information: Optional[Union[str, List[RelatedInformationItem]]] = None,
        project_root: Optional[str] = None,
        target_entity_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Executes full ReFEree evaluation:
        1. Code-related information retrieval (1-hop DFS on context graph).
        2. Sentence-level summary segmentation.
        3. Segment-level evaluation across 4 criteria (C1-C4) via LLM.
        4. Score aggregation.
        """
        # Step 1: Code-related information
        if related_information is None:
            items = self.searcher.search_for_code(
                input_code=input_code,
                target_entity_name=target_entity_name,
                project_root=project_root,
            )
            rel_info_str = self.searcher.format_related_information(items)
        elif isinstance(related_information, list):
            rel_info_str = self.searcher.format_related_information(related_information)
        else:
            rel_info_str = str(related_information).strip()

        # Step 2: Sentence segmentation
        segments = self.tokenize_summary(summary)
        if not segments:
            # Handle empty summary
            return EvaluationResult(
                input_code=input_code,
                summary=summary,
                related_information=rel_info_str,
                segments=[],
                overall_score=0.0,
                weighted_score=0.0,
                criterion_summary={},
            )

        # Step 3: Segment-level criterion evaluation
        segment_results: List[SegmentResult] = []
        criterion_pass_counts: Dict[CriterionType, int] = {c: 0 for c in CRITERIA_ORDER}

        for idx, segment_text in enumerate(segments, start=1):
            crit_scores: Dict[CriterionType, int] = {}
            violated: List[CriterionType] = []

            for crit_type in CRITERIA_ORDER:
                criterion = CRITERIA[crit_type]
                sys_prompt = self.format_system_prompt(criterion)
                user_prompt = self.format_user_prompt(
                    input_code=input_code,
                    related_information=rel_info_str,
                    segment=segment_text,
                )

                score = self.llm_client.evaluate_criterion(sys_prompt, user_prompt)
                score = 1 if score == 1 else 0
                crit_scores[crit_type] = score

                if score == 1:
                    criterion_pass_counts[crit_type] += 1
                else:
                    violated.append(crit_type)

            seg_score = sum(crit_scores.values()) / len(CRITERIA_ORDER)
            segment_results.append(
                SegmentResult(
                    segment_index=idx,
                    text=segment_text,
                    criterion_scores=crit_scores,
                    violated_criteria=violated,
                    segment_score=seg_score,
                )
            )

        # Step 4: Aggregation
        # Paper Formula (Section 3.3):
        # SCORE = 1 / (|D| * |Criteria|) * sum_{S in D} sum_{C in Criteria} f(S, C)
        num_segments = len(segments)
        num_criteria = len(CRITERIA_ORDER)
        total_pass = sum(sum(seg.criterion_scores.values()) for seg in segment_results)
        overall_score = total_pass / (num_segments * num_criteria)

        # Weighted configuration (Appendix C.5, Table 14: 0.6 : 1.2 : 1.2 : 1.0)
        total_weighted_pass = sum(
            sum(CRITERIA[c].tuned_weight * seg.criterion_scores[c] for c in CRITERIA_ORDER)
            for seg in segment_results
        )
        sum_weights = sum(CRITERIA[c].tuned_weight for c in CRITERIA_ORDER)
        weighted_score = total_weighted_pass / (num_segments * sum_weights)

        # Build criterion summary
        crit_summary: Dict[str, Dict[str, Any]] = {}
        for c in CRITERIA_ORDER:
            passes = criterion_pass_counts[c]
            crit_summary[c.value] = {
                "name": CRITERIA[c].name,
                "consistent_segments": passes,
                "total_segments": num_segments,
                "consistency_rate": passes / num_segments,
            }

        return EvaluationResult(
            input_code=input_code,
            summary=summary,
            related_information=rel_info_str,
            segments=segment_results,
            overall_score=overall_score,
            weighted_score=weighted_score,
            criterion_summary=crit_summary,
        )

