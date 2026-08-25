import json, asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.rag_pipeline import RAGPipeline

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/rag_triad_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the golden dataset
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)

# Create pipeline instance once (loads vector store + reranker model)
pipeline = asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5, score_threshold=0.0))

# Build test cases — pipeline does its own retrieval per query
test_cases = []

for gd in golden_dataset:
    result = asyncio.run(pipeline.invoke(gd["query"]))

    test_cases.append(
        LLMTestCase(
            input=gd["query"],
            actual_output=result["answer"],
            retrieval_context=result["context"]
        )
    )
    
# RAG triad metrics: faithfulness, answer relevancy, contextual relevancy
metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

evaluate(
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