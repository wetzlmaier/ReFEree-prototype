"""
Command Line Interface for ReFEree.

Provides commands to evaluate code summaries, inspect 1-hop dependencies,
and run paper demonstration examples.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Optional

from referee.evaluator import ReFEreeEvaluator
from referee.graph import ProjectContextGraph
from referee.search import CodeRelatedInfoSearcher
from referee.llm import OpenAILLMClient, MockLLMClient, load_env_file


def read_content_or_file(input_val: str) -> str:
    """If input_val is an existing file, read and return its content; otherwise return input_val."""
    if os.path.isfile(input_val):
        with open(input_val, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return input_val


def build_evaluator(
    mock: bool = False,
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    repo_dir: Optional[str] = None,
) -> ReFEreeEvaluator:
    """Builds a configured ReFEreeEvaluator instance."""
    # Context graph and searcher
    graph = ProjectContextGraph(root_dir=repo_dir)
    if repo_dir and os.path.isdir(repo_dir):
        graph.add_repository(repo_dir)
    searcher = CodeRelatedInfoSearcher(graph=graph)

    # LLM Client
    if mock:
        llm_client = MockLLMClient()
    else:
        load_env_file()
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            print(
                "[Warning] No OPENAI_API_KEY detected in environment or arguments. "
                "Defaulting to MockLLMClient for offline execution. "
                "Use --api-key <key> or set OPENAI_API_KEY to use OpenAI models.",
                file=sys.stderr,
            )
            llm_client = MockLLMClient()
        else:
            llm_client = OpenAILLMClient(model=model, api_key=resolved_key)

    return ReFEreeEvaluator(llm_client=llm_client, searcher=searcher)


def cmd_evaluate(args: argparse.Namespace) -> None:
    code_text = read_content_or_file(args.code)
    summary_text = read_content_or_file(args.summary)
    related_info = read_content_or_file(args.related_info) if args.related_info else None

    evaluator = build_evaluator(
        mock=args.mock,
        model=args.model,
        api_key=args.api_key,
        repo_dir=args.repo,
    )

    result = evaluator.evaluate(
        input_code=code_text,
        summary=summary_text,
        related_information=related_info,
        project_root=args.repo,
    )

    if args.json:
        output_dict = {
            "overall_score": result.overall_score,
            "weighted_score": result.weighted_score,
            "related_information": result.related_information,
            "segments": [
                {
                    "index": s.segment_index,
                    "text": s.text,
                    "segment_score": s.segment_score,
                    "criterion_scores": {c.value: score for c, score in s.criterion_scores.items()},
                    "violated_criteria": [c.value for c in s.violated_criteria],
                }
                for s in result.segments
            ],
            "criterion_summary": result.criterion_summary,
        }
        print(json.dumps(output_dict, indent=2))
    else:
        print(result.summary_report())


def cmd_search(args: argparse.Namespace) -> None:
    code_text = read_content_or_file(args.code)
    graph = ProjectContextGraph(root_dir=args.repo)
    if args.repo and os.path.isdir(args.repo):
        graph.add_repository(args.repo)
    searcher = CodeRelatedInfoSearcher(graph=graph)

    items = searcher.search_for_code(code_text, project_root=args.repo)
    print(f"Discovered {len(items)} 1-hop related information items:")
    for i, item in enumerate(items, 1):
        print(f"\n--- Item {i} ({item.name}) [{item.source_type}] ---")
        print(item.format_paper_style())


def cmd_demo(args: argparse.Namespace) -> None:
    """Runs demonstration with canonical examples from the paper."""
    print("=" * 60)
    print("ReFEree Demonstration: Paper Canonical Examples")
    print("=" * 60)

    # Example 1: Table 4 / Table 20 from paper (convert_to_idn)
    code_1 = '''def convert_to_idn(url):
    parts = list(urllib.parse.urlsplit(url))
    try:
        parts[1].encode('ascii')
    except UnicodeEncodeError:
        host = parts[1].rsplit(':', 1)
        newhost = []
        port = ''
        if len(host) == 2:
            port = host.pop()
        for h in host[0].split('.'):
            newhost.append(h.encode('idna').decode('utf-8'))
        parts[1] = '.'.join(newhost)
        if port:
            parts[1] += ':' + port
        return urllib.parse.urlunsplit(parts)
    else:
        return url'''

    summary_1 = (
        "The 'convert_to_idn' function is designed to convert the hostname portion of a URL into "
        "its Internationalized Domain Name (IDN) ASCII-compatible encoding using the Punycode standard. "
        "It first splits the URL into its components using the 'urlparse' function, then attempts to "
        "encode the hostname in ASCII. "
        "If the hostname contains Unicode characters, it separates the hostname from the port if present, "
        "encodes each label of the hostname into IDNA format, and finally recombines the parts, preserving the port number. "
        "The function returns a list of URL components with the converted hostname. "
        "This utility is particularly useful in web crawlers and multilingual domain management systems where URLs "
        "need to be normalized for consistent processing and storage."
    )

    print("\nRunning Example: 'convert_to_idn' (Table 4 & Table 20 in paper)")
    print(f"Code:\n{code_1}\n")

    # Set up mock with paper ground-truth violations from Table 20:
    # Segment 2: violates C1, C2, C3, C4 (urlparse instead of urlsplit)
    # Segment 4: violates C1, C2, C3, C4 (returns list of URL components instead of str)
    # Segment 5: violates C4 (overly generalized irrelevant domain crawler text)
    mock_client = MockLLMClient()
    for crit in ["name", "type", "functionality", "context"]:
        mock_client.set_rule("urlparse", crit, 0)
        mock_client.set_rule("returns a list of url components", crit, 0)
    mock_client.set_rule("web crawlers", "context", 0)

    evaluator = ReFEreeEvaluator(llm_client=mock_client)
    res = evaluator.evaluate(input_code=code_1, summary=summary_1)
    print(res.summary_report())
    print("\n[Paper Reference Ground Truth]:")
    print("  Paper ReFEree Score: 0.55")
    print(f"  Prototype Score:     {res.overall_score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="referee",
        description="ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Code Summarization",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a code summary")
    eval_parser.add_argument("--code", "-c", required=True, help="Input code string or file path")
    eval_parser.add_argument("--summary", "-s", required=True, help="Summary text string or file path")
    eval_parser.add_argument("--related-info", "-r", help="Optional pre-existing related info string or file")
    eval_parser.add_argument("--repo", help="Path to project repository root")
    eval_parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name (default: gpt-4.1-mini)")
    eval_parser.add_argument("--api-key", help="OpenAI API key")
    eval_parser.add_argument("--mock", action="store_true", help="Run with mock LLM client (offline)")
    eval_parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    # search command
    search_parser = subparsers.add_parser("search", help="Search 1-hop dependencies for code")
    search_parser.add_argument("--code", "-c", required=True, help="Input code string or file path")
    search_parser.add_argument("--repo", help="Path to project repository root")

    # demo command
    subparsers.add_parser("demo", help="Run paper demonstration examples")

    args = parser.parse_args()
    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
