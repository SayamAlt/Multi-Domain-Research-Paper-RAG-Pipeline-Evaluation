import json, asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from src.reranker import RerankingRetriever

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/retriever_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)
    
# Run the retriever on each question to fill retrieval_context to build one test case for each record
retriever = asyncio.run(RerankingRetriever.create(fetch_k=20, top_k=5))

test_cases = []

for gd in golden_dataset:
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
    
# Define a list of retrieval-specific metrics
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

# Evaluate all test cases to generate a final report of metrics
evaluate(
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