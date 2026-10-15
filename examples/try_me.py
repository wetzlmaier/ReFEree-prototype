"""
Interactive ReFEree Try-Out Example.

Run this script directly:
    python examples/try_me.py

You can modify the `code` and `summary` strings below to test any code!
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from referee import ReFEreeEvaluator, MockLLMClient, OpenAILLMClient

# --------------------------------------------------------------------------
# 1. Provide the Python Code to Evaluate
# --------------------------------------------------------------------------
code = '''
def calculate_duration_days(start_date, end_date):
    """Calculates number of whole days between two date objects."""
    import datetime
    delta = end_date - start_date
    return delta.days
'''

# --------------------------------------------------------------------------
# 2. Provide the Summary to Evaluate
# --------------------------------------------------------------------------
# Notice the nuances:
# - Sentence 1: Accurately describes what the code does (Consistent)
# - Sentence 2: Claims it uses 'date_diff' and returns a float (Name Inconsistency C1, Type Inconsistency C2)
# - Sentence 3: Mentions financial trading and NASDAQ holidays (Context Irrelevant C4)
summary = (
    "The calculate_duration_days function computes the difference in days between two dates. "
    "It uses the helper function date_diff to compute the duration and returns a floating-point value. "
    "This calculation is critical for Wall Street automated high-frequency trading algorithms adjusting for NASDAQ holidays."
)

# --------------------------------------------------------------------------
# 3. Choose Evaluator Mode (Offline Simulation or Live OpenAI)
# --------------------------------------------------------------------------
api_key = os.environ.get("OPENAI_API_KEY")

if api_key:
    print("[Mode] Using live OpenAI GPT model (OPENAI_API_KEY detected)...")
    evaluator = ReFEreeEvaluator(llm_client=OpenAILLMClient(model="gpt-4.1-mini", api_key=api_key))
else:
    print("[Mode] Running in offline demo mode (no API key required)...")
    # Simulate segment evaluation:
    mock_client = MockLLMClient(default_score=1)
    # Sentence 2 violates C1 (Name: 'date_diff') and C2 (Type: 'floating-point')
    mock_client.set_rule("date_diff", "name", 0)
    mock_client.set_rule("date_diff", "type", 0)
    # Sentence 3 violates C4 (Context: NASDAQ/Wall Street is ungrounded)
    mock_client.set_rule("nasdaq", "context", 0)
    evaluator = ReFEreeEvaluator(llm_client=mock_client)

# --------------------------------------------------------------------------
# 4. Run Evaluation
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Evaluating Source Code & Summary with ReFEree")
print("=" * 60)

result = evaluator.evaluate(input_code=code, summary=summary)

print(result.summary_report())

