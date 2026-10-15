"""
Project-Level Context Graph Example for ReFEree.

Demonstrates how ReFEree constructs a Project Context Graph across multiple files,
extracts 1-hop cross-file dependencies, and uses them to evaluate factual consistency
as illustrated in Figure 1 and Figure 2 of the paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from referee import (
    ProjectContextGraph,
    CodeRelatedInfoSearcher,
    ReFEreeEvaluator,
    MockLLMClient,
)

# File 1: utils.py
utils_code = '''
def key_prefix(prefix: str, ids: list) -> str:
    """Combines prefix and ids into a normalized prefix string."""
    prefix_str = str(prefix)
    return str(prefix_str)
'''

# File 2: main.py
main_code = '''
from utils import key_prefix

def s3_key_prefixes(prefix, ids, org_ids=None):
    """Generates a list of S3 key prefixes."""
    result = [key_prefix(prefix, ids)]
    return result
'''

# Summary claiming key_prefix generates a list (Type error)
summary = (
    "The s3_key_prefixes function generates a list of S3 key prefixes. "
    "The key_prefix function generates a comprehensive list of key prefixes for each identifier. "
    "Finally, s3_key_prefixes returns the accumulated list."
)


def main():
    print("=" * 60)
    print("Multi-File Project Context Evaluation (Figure 1 & Figure 2 in paper)")
    print("=" * 60)

    # 1. Build Project Context Graph
    graph = ProjectContextGraph()
    graph.add_python_source(utils_code, file_path="utils.py")
    graph.add_python_source(main_code, file_path="main.py")

    # 2. 1-hop Dependency Search
    searcher = CodeRelatedInfoSearcher(graph=graph)
    related_items = searcher.search_for_code(
        main_code, file_path="main.py", target_entity_name="s3_key_prefixes"
    )

    print(f"\nDiscovered {len(related_items)} 1-hop dependency items:")
    for item in related_items:
        print(f"[{item.source_type}] {item.name}:")
        print(item.content.strip())

    # 3. Formatted related information prepended to code
    related_info_str = searcher.format_related_information(related_items)
    print(f"\nPrepended Related Information:\n{related_info_str}")

    # 4. Evaluate with Mock Client reflecting the Type Inconsistency in sentence 2
    mock_client = MockLLMClient(default_score=1)
    # Sentence 2 states key_prefix returns a comprehensive list, but actual return type is str (C2 Type Inconsistency)
    mock_client.set_rule("comprehensive list of key prefixes", "type", 0)
    mock_client.set_rule("comprehensive list of key prefixes", "functionality", 0)

    evaluator = ReFEreeEvaluator(llm_client=mock_client, searcher=searcher)
    result = evaluator.evaluate(
        input_code=main_code,
        summary=summary,
        related_information=related_info_str,
    )

    print("\n" + result.summary_report())


if __name__ == "__main__":
    main()
