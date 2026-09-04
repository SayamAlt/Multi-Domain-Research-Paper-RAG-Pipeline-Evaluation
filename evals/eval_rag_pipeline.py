import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from src.rag_pipeline import RAGPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/rag_triad_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

def run(pipeline):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for gd in goldens:
        result = asyncio.run(pipeline.invoke(gd["query"]))
        test_cases.append(
            LLMTestCase(
                input=gd["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"]
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualRelevancyMetric(threshold=0.6, model=JUDGE_MODEL, include_reason=True)
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL,
            "threshold": THRESHOLD,
            "generator_model": "gpt-4o-mini",
            "temperature": 0,
            "embedding_model": "text-embedding-3-large",
            "vector_store": "FAISS",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "fetch_k": 20,
            "top_k": 5,
            "reranker": "cross-encoder/ms-marco-electra-base"
        }
    )
    return summarize_by_metric(result)

def run_local():
    return run(asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5, score_threshold=0.0)))

if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())