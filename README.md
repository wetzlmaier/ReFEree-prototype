# ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization

A complete Python prototype implementing the **ReFEree** algorithm as proposed by Bae et al. (ACL 2026):
> **ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization**  
> *Suyoung Bae, CheolWon Na, Jaehoon Lee, Yumin Lee, YunSeok Choi, Jee-Hyong Lee*  
> Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026).

---

## Overview

Traditional evaluation methods for code summarization rely either on lexical overlap with reference summaries (which suffer from one-to-many variations) or on coarse LLM-as-a-judge approaches (which assess summaries as a whole with single-scalar scores and ignore external code context).

**ReFEree** solves this through a two-stage, reference-free, fine-grained framework:

```
+--------------------+        +-----------------------------+        +--------------------------+
|  Input Code &      |  --->  | Stage 1: Code-Related       |  --->  | Prepended Related Info:  |
|  Repository / Docs |        | Information Searching       |        | # {name} # {content}     |
+--------------------+        | (1-hop DFS on Context Graph)|        +--------------------------+
                              +-----------------------------+                     |
                                                                                  v
+--------------------+        +-----------------------------+        +--------------------------+
| Generated Summary  |  --->  | Sentence Segmentation       |  --->  | Stage 2: Segment-Level   |
|                    |        | (NLTK Tokenizer)            |        | Criterion Evaluation     |
+--------------------+        +-----------------------------+        | (C1, C2, C3, C4 via LLM) |
                                                                     +--------------------------+
                                                                                  |
                                                                                  v
                                                                     +--------------------------+
                                                                     | Fine-Grained Score &     |
                                                                     | Explainability Report    |
                                                                     +--------------------------+
```

---

## Core Algorithm Components

### 1. Four Factual Inconsistency Criteria (Section 3.1, Table 1)
| Criterion | Name | Definition |
|---|---|---|
| **[C1]** | **Name Inconsistency** | The name of a function, class, or variable in the summary does not match the actual identifier in the input code or mistakenly refers to a different entity with the same name in the project. |
| **[C2]** | **Type Inconsistency** | The described function’s return type or variable type in the summary is inconsistent with the actual type in the input code or in directly dependent functions. |
| **[C3]** | **Functionality Inconsistency** | The functionality or purpose described in the summary does not accurately reflect what the input code or its dependent functions actually implement. This often arises when dependency relationships are ignored or misinterpreted, leading to an incorrect description of behavior. |
| **[C4]** | **Context Irrelevant** | The summary contains content that is unnecessary or irrelevant to the input code or relevant information, such as descriptions of unrelated entities or overly generalized dependency context that does not contribute to understanding the function. |

### 2. Stage 1: Code-Related Information Searching (Section 3.2, Appendix A.2)
1. **Project Context Graph Construction:**
   Constructs a directed acyclic graph (DAG) $G = \{(h, r, t) \mid h, t \in E, r \in R\}$ using AST analysis with 5 relation types:
   - `assign`: Variable creation/mutation in assignment statements.
   - `as`: Bindings in `with ... as` or `except ... as` statements.
   - `refers`: Reference to existing variable, attribute, or function call.
   - `typeof`: Data type of variable or function return type.
   - `inherits`: Subclass inheritance relation.
2. **1-Hop Dependency Search:**
   Traverses direct (depth-1) dependencies from the target entity:
   - **Internal-file dependencies:** Directly extracted node code bodies or class docstrings.
   - **Cross-file dependencies:** Resolves import statements to module files within the repository.
   - **External dependencies:** Queries external API documentation (predefined or dynamic inspection) for standard library/third-party APIs.
3. **Information Formatting:**
   Prepends the retrieved context to the code in the format:
   ```
   # {name} #
   {content}
   ```

### 3. Stage 2: Segment-Level Evaluation & Score Aggregation (Section 3.3, Appendix A.3)
1. **Summary Segmentation:**
   Decomposes the summary into sentence-level segments $D = \{S_1, S_2, \dots, S_n\}$ using sentence tokenization.
2. **Binary Criterion Evaluation Prompt:**
   For each segment $S$ and criterion $C$, the LLM evaluates presence (0) or absence (1) of inconsistency:
   $$f(S, C) \in \{0, 1\}$$
3. **Score Aggregation:**
   Unweighted score (Section 3.3):
   $$\text{SCORE} = \frac{1}{|D| \times |\text{Criteria}|} \sum_{S \in D} \sum_{C \in \text{Criteria}} f(S, C)$$
   Optional weighted score (Appendix C.5, Table 14: $C_1=0.6, C_2=1.2, C_3=1.2, C_4=1.0$):
   $$\text{SCORE}_{\text{weighted}} = \frac{1}{|D| \sum w_C} \sum_{S \in D} \sum_{C \in \text{Criteria}} w_C \cdot f(S, C)$$

---

## Directory Structure

```
ReFEree-prototype/
├── pyproject.toml             # Package configuration & dependencies
├── README.md                  # Comprehensive documentation
├── referee/                   # Core Python package
│   ├── __init__.py            # Public API exports
│   ├── criteria.py            # Criteria definitions (C1-C4) & weights
│   ├── graph.py               # AST parsing & Project Context Graph (5 relation types)
│   ├── search.py              # 1-hop DFS dependency search & prepending format
│   ├── external_docs.py       # Predefined & dynamic API documentation registry
│   ├── llm.py                 # OpenAI, Callable, and Mock LLM client adapters
│   ├── evaluator.py           # Evaluator, prompt formatting, scoring formulas
│   ├── cli.py                 # Command-line interface
│   └── __main__.py            # CLI entrypoint (python -m referee)
├── tests/                     # Unit and integration test suite
│   ├── test_criteria.py       # C1-C4 definition and weight tests
│   ├── test_graph.py          # AST entity extraction & 5 edge relation tests
│   ├── test_search.py         # 1-hop DFS, in-file, cross-file & external doc tests
│   ├── test_evaluator.py      # Tokenization, prompt formatting & score formula tests
│   └── test_paper_examples.py # Canonical paper examples (Table 4, 20, 21)
└── examples/                  # Executable examples
    ├── quickstart.py          # Basic evaluation snippet
    └── project_example.py     # Multi-file repository evaluation (Figure 1 & 2)
```

---

## Installation

Install in editable mode in your Python 3.9+ environment:

```bash
pip install -e .
```

Dependencies:
- `nltk>=3.8` (for sentence tokenization)
- `openai>=1.0.0` (for OpenAI LLM evaluations)

---

## Usage

### 1. Python API

#### Basic Evaluation

```python
from referee import ReFEreeEvaluator, OpenAILLMClient

# Initialize with OpenAI model (default: gpt-4.1-mini)
client = OpenAILLMClient(model="gpt-4.1-mini", api_key="your-api-key")
evaluator = ReFEreeEvaluator(llm_client=client)

code = '''
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid discount percentage")
    return round(price * (1 - discount_percent / 100.0), 2)
'''

summary = (
    "The calculate_discount function calculates the final discounted price. "
    "It validates the discount percentage and returns the discounted price as an integer. "
    "This method is used in relational database triggers."
)

result = evaluator.evaluate(input_code=code, summary=summary)

print(f"Overall Score: {result.overall_score:.4f}")
print(result.summary_report())
```

#### Multi-File Project Context Evaluation

```python
from referee import ProjectContextGraph, CodeRelatedInfoSearcher, ReFEreeEvaluator, OpenAILLMClient

# Build project graph from repository directory
graph = ProjectContextGraph(root_dir="./my_project")
graph.add_repository("./my_project")

searcher = CodeRelatedInfoSearcher(graph=graph)
evaluator = ReFEreeEvaluator(llm_client=OpenAILLMClient(), searcher=searcher)

result = evaluator.evaluate(
    input_code=target_code,
    summary=summary_text,
    project_root="./my_project"
)
```

#### Offline Testing with `MockLLMClient`

You can test without an OpenAI API key using `MockLLMClient`:

```python
from referee import ReFEreeEvaluator, MockLLMClient

mock_client = MockLLMClient(default_score=1)
# Register specific rule: (substring, criterion_name_or_id) -> 0
mock_client.set_rule("returns the discounted price as an integer", "type", 0)

evaluator = ReFEreeEvaluator(llm_client=mock_client)
result = evaluator.evaluate(input_code=code, summary=summary)
print(result.overall_score)
```

---

### 2. Command Line Interface (CLI)

#### Run Paper Demonstration
Runs the canonical example (`convert_to_idn`) from Table 4 & Table 20 in the paper:
```bash
referee demo
```

#### Evaluate a Code Summary
```bash
referee evaluate --code path/to/code.py --summary path/to/summary.txt
```

Options:
- `--code`, `-c`: Input code string or file path.
- `--summary`, `-s`: Summary text string or file path.
- `--repo`: Path to repository directory for cross-file dependency resolution.
- `--model`: LLM model name (default: `gpt-4.1-mini`).
- `--api-key`: OpenAI API key (or use `OPENAI_API_KEY` env var).
- `--mock`: Run with offline mock LLM.
- `--json`: Output machine-readable JSON.

#### Search 1-Hop Dependencies
Inspect 1-hop dependencies identified for a code snippet:
```bash
referee search --code path/to/code.py --repo ./my_project
```

---

## Running Tests

All 15 unit and integration tests validate the implementation against the paper's formulas and ground-truth values:

```bash
python -m unittest discover -s tests -v
```

