import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.metrics.g_eval import Rubric
from src.rag_pipeline import RAGPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/application_level_golden_dataset.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

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
        Rubric(score_range=(9, 10), expected_outcome="Fully aligned with expected style: direct opening, precise technical vocabulary with specific details, well-structured for complexity, zero hedging or padding."),
        Rubric(score_range=(7, 8), expected_outcome="Mostly direct and technical but contains minor hedging, one filler phrase, or slightly loose structure on a multi-part answer."),
        Rubric(score_range=(4, 6), expected_outcome="Noticeable style friction: vague qualifiers, conversational opener or closing summary, or prose dump on a multi-part answer that deserved structure."),
        Rubric(score_range=(0, 3), expected_outcome="Heavily padded or hedged throughout; reads like a polite chatbot rather than a precise technical system — lots of 'great question', 'certainly', or soft non-committal language.")
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
        style,
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
    return run(asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5)))

if __name__ == "__main__":
    print_summary("application", run_local())
