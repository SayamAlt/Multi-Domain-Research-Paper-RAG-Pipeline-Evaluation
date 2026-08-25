import json, asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.generator import generate_answer

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/generator_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the golden dataset
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)
    
# Build test cases for each generator answer on the golden dataset
test_cases = []

for gd in golden_dataset:
    context = gd["ideal_context"]
    answer = asyncio.run(generate_answer(gd["query"], context))
    
    test_cases.append(
        LLMTestCase(
            input=gd["query"],
            actual_output=answer,
            retrieval_context=context 
            # No expected output here since faithfulness metric never reads expected_output; it may leak expected output
        )
    )
    
# Define a list of key metrics for generator evaluation
metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

# Run evaluation of all test cases from golden dataset to display final report
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "judge_model": JUDGE_MODEL,
        "threshold": THRESHOLD,
        "generator_model": "gpt-4o-mini",
        "temperature": 0,
        "context_source": "ideal_context (golden dataset)"
    }
)