from referee import ReFEree

# Dummy model evaluator that simply returns 1
def mock_evaluator(prompt):
    # In a real scenario, this would call an LLM API and parse the response to extract 0 or 1.
    return 1

evaluator = ReFEree(mock_evaluator)

code = "def add(a, b):\n    return a + b"
summary = "This function adds two numbers. It is a very useful function."

score = evaluator.evaluate(code, summary)
print(f"Calculated Score: {score}")

def mock_evaluator_half(prompt):
    if "Name inconsistency" in prompt or "Type inconsistency" in prompt:
        return 1
    return 0

evaluator_half = ReFEree(mock_evaluator_half)
score_half = evaluator_half.evaluate(code, summary)
print(f"Calculated Score with half evaluations passing: {score_half}")
