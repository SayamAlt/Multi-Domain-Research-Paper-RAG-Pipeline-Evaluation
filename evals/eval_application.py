import json, asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from src.rag_pipeline import RAGPipeline

load_dotenv()

GOLDEN_DATASET_PATH = "goldens/application_level_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

# Load the golden dataset
with open(GOLDEN_DATASET_PATH) as f:
    golden_dataset = json.load(f)
    
# Run the RAG pipeline on each record of golden dataset to build a test case from LIVE output
rag_pipeline = asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5))
test_cases = []

for gd in golden_dataset:
    result = asyncio.run(rag_pipeline.invoke(gd["query"]))

    test_cases.append(
        LLMTestCase(
            input=gd["query"],
            actual_output=result["answer"],
            expected_output=gd["expected_answer"]
        )
    )
    
# Define a list of metrics to perform application-level evaluation of the RAG pipeline
# Correctness - reference-based, judges TRUTH (not coverage or length of output)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually incorrect. Judge truth, not completeness.",
        "A factually correct answer must score atleast 0.9 even if it is short, less-detailed, or covers fewer points than the expected output.",
        "Completely ignore brevity, missing elaboration, or omitted points as omission is not a judgement criteria here.",
        "Additional correct information must NEVER lower the score."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
        Rubric(score_range=(5,8), expected_outcome="Mostly correct claims but one or two minor inaccuracies."),
        Rubric(score_range=(0,4), expected_outcome="Contains a clear factual error or a claim that contradicts the expected output.")
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=THRESHOLD,
    top_logprobs=20,
    model=JUDGE_MODEL,
    strict_mode=False
)

# Completeness - reference-based, judges coverage (not correctness)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points mentioned in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or ONLY partially covers.",
        "Judge coverage ONLY. Do NOT lower the score because a covered point is stated incorrectly - factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="Addresses essentially all key points from the expected output."),
        Rubric(score_range=(5,8), expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0,4), expected_outcome="Misses several key points; only partially covers the expected output.")
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    top_logprobs=20,
    strict_mode=False
)

# Style - reference-free, judges TONE only (No expected output)
style = GEval(
    name="Style",
    evaluation_steps=[
        "Judge only the writing style, tone, and delivery of the actual output — not correctness or completeness.",
        "Reward answers that are direct and get to the point immediately, with no preamble, filler phrases, or pleasantries.",
        "Reward use of precise technical language: specific model names, metric values, architecture names, and quantitative details where the context supports them.",
        "Reward structured formatting (bullet points, numbered steps, or clear sections) when the answer covers multiple distinct sub-topics; penalize walls of prose for multi-part answers.",
        "Penalize hedging language such as 'it might', 'perhaps', 'you could consider', 'generally speaking', or vague qualifiers that weaken technical claims.",
        "Penalize conversational padding: introductory restatements of the question, closing summaries that repeat what was just said, and filler transitions like 'In conclusion' or 'To summarize'.",
        "Do NOT reward or penalize based on correctness, completeness, or length — only on style and tone."
    ],
    rubric=[
        Rubric(score_range=(9,10), expected_outcome="Fully aligned with expected style: direct opening, precise technical vocabulary with specific details, well-structured for complexity, zero hedging or padding."),
        Rubric(score_range=(7,8), expected_outcome="Mostly direct and technical but contains minor hedging, one filler phrase, or slightly loose structure on a multi-part answer."),
        Rubric(score_range=(4,6), expected_outcome="Noticeable style friction: vague qualifiers, conversational opener or closing summary, or prose dump on a multi-part answer that deserved structure."),
        Rubric(score_range=(0,3), expected_outcome="Heavily padded or hedged throughout; reads like a polite chatbot rather than a precise technical system — lots of 'great question', 'certainly', or soft non-committal language.")
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    threshold=THRESHOLD,
    top_logprobs=20,
    model=JUDGE_MODEL,
    strict_mode=False
)

# Evaluate the RAG pipeline on all aforementioned metrics to perform application-level evaluation
evaluate(
    test_cases=test_cases,
    metrics=[correctness, completeness, style],
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