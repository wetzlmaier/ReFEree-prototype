import re
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# We'll define the cases and prompts internally.
CASE = {
    'Name inconsistency': 'the name of a function, class, or variable mentioned in the SUMMARY TEXT does not match the actual identifier used in the PYTHON CODE',
    'Type inconsistency': 'the described function\'s return type or variable type in the SUMMARY TEXT is inconsistent with the actual type used or inferred in the PYTHON CODE',
    'Functionality inconsistency': 'the described functionality or purpose of the code in the SUMMARY TEXT does not accurately reflect what the PYTHON CODE actually implements. This typically occurs when the input code is dependent on a complex project environment or lacks enough information, leading to a misunderstanding of the actual functionality',
    'Irrelevant context': 'the SUMMARY TEXT contains content that is unnecessary or unrelated to the PYTHON CODE, such as overly generalized descriptions of specific code entities or information that can be excluded without affecting the understanding of the code'
}

PROMPT_TEMPLATE = '''You will be given one summary text written for a source code. Your task is to evaluate the summary from {case} aspect.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{case} (1 or 0) - {explanation}.

Evaluation Steps:
1. Read the PYTHON CODE carefully and understand its main intent.
2. Read the code summary text and check if it accurately describe the code.
3. Evaluate whether {case} exists, where `1` means “{case} does not exist” and `0` means “{case} exists” based on the Evaluation Criteria.

### PYTHON CODE:
{x}
### SUMMARY TEXT: {text}
### SCORE (score only):'''

class ReFEree:
    def __init__(self, evaluator_func):
        """
        evaluator_func: A callable that takes a string prompt and returns an integer (0 or 1).
                        0 means factual inconsistency detected.
                        1 means no factual inconsistency detected.
        """
        self.evaluator_func = evaluator_func

    def evaluate(self, code: str, summary: str, related_information: str = None) -> float:
        """
        Evaluates the factual consistency of the provided summary against the code.
        Returns a score between 0.0 and 1.0.
        """
        segments = nltk.sent_tokenize(summary)
        if not segments:
            return 0.0

        total_evaluations = len(segments) * len(CASE)
        total_score = 0

        for segment in segments:
            for case, explanation in CASE.items():
                prompt = PROMPT_TEMPLATE.format(
                    case=case,
                    explanation=explanation,
                    x=code,
                    text=segment
                )

                if related_information:
                    prompt = prompt.replace(
                        "### PYTHON CODE:",
                        f"### RELATED INFORMATION:\n{related_information}\n### PYTHON CODE:"
                    )

                score = self.evaluator_func(prompt)
                total_score += score

        return total_score / total_evaluations
