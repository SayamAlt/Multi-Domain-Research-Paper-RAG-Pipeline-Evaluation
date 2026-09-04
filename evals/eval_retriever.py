import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for gd in goldens:
        retrieved_docs = asyncio.run(retriever.invoke(gd["query"]))
        retrieval_context = [doc.page_content for doc in retrieved_docs]
        test_cases.append(
            LLMTestCase(
                input=gd["query"],
                expected_output=gd["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="Generator is not evaluated in this run"
            )
        )

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL,
            "threshold": THRESHOLD,
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
    return run(asyncio.run(RerankingRetriever.create(fetch_k=20, top_k=5)))

if __name__ == "__main__":
    print_summary("retriever", run_local())