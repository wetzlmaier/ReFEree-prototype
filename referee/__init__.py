"""
ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization.

Based on the research paper:
    Bae et al. (ACL 2026), "ReFEree: Reference-Free and Fine-Grained Method for
    Evaluating Factual Consistency in Real-World Code Summarization".
"""

from referee.criteria import CriterionType, Criterion, CRITERIA, CRITERIA_ORDER
from referee.graph import ProjectContextGraph, CodeEntity, EdgeRelation
from referee.search import CodeRelatedInfoSearcher, RelatedInformationItem
from referee.external_docs import ExternalDocRegistry, default_external_doc_registry
from referee.llm import BaseLLMClient, OpenAILLMClient, MockLLMClient, CallableLLMClient, load_env_file
from referee.evaluator import ReFEreeEvaluator, SegmentResult, EvaluationResult

__all__ = [
    "CriterionType",
    "Criterion",
    "CRITERIA",
    "CRITERIA_ORDER",
    "ProjectContextGraph",
    "CodeEntity",
    "EdgeRelation",
    "CodeRelatedInfoSearcher",
    "RelatedInformationItem",
    "ExternalDocRegistry",
    "default_external_doc_registry",
    "BaseLLMClient",
    "OpenAILLMClient",
    "MockLLMClient",
    "CallableLLMClient",
    "load_env_file",
    "ReFEreeEvaluator",
    "SegmentResult",
    "EvaluationResult",
]

