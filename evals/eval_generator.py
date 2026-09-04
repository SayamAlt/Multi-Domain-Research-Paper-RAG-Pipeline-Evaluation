import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.generator import generate_answer
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/generator_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

def run():
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for gd in goldens:
        context = gd["ideal_context"]
        answer = asyncio.run(generate_answer(gd["query"], context))
        test_cases.append(
            LLMTestCase(
                input=gd["query"],
                actual_output=answer,
                retrieval_context=context
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
    ]

    result = evaluate(
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
    return summarize_by_metric(result)

def run_local():
    return run()

if __name__ == "__main__":
    print_summary("generator", run_local())