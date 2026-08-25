from collections import Counter
from sentence_transformers import CrossEncoder
from src.retriever import load_vector_store
import asyncio

# Load the cross encoder model to rerank retrieval results
CROSS_ENCODER = "cross-encoder/ms-marco-electra-base"

class RerankingRetriever:
    def __init__(self, fetch_k=15, top_k=5, score_threshold=0.0):
        self.reranker = CrossEncoder(CROSS_ENCODER, max_length=512, device="cpu")
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.score_threshold = score_threshold

    @classmethod
    async def create(cls, fetch_k=15, top_k=5, score_threshold=0.0):
        instance = cls(fetch_k=fetch_k, top_k=top_k, score_threshold=score_threshold)
        instance.vector_store = await load_vector_store()
        return instance

    async def invoke(self, query: str):
        loop = asyncio.get_event_loop()
        candidates_with_scores = await loop.run_in_executor(
            None, lambda: self.vector_store.similarity_search_with_relevance_scores(query, k=self.fetch_k)
        )

        if not candidates_with_scores:
            return []

        # Majority vote on top third of FAISS results → dominant source paper for this query
        top_slice = max(1, self.fetch_k // 3)
        top_candidates = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)[:top_slice]
        source_counts = Counter(doc.metadata.get("source", "") for doc, _ in top_candidates)
        dominant_source = source_counts.most_common(1)[0][0]

        # Keep only chunks from dominant source before reranking
        candidates = [doc for doc, _ in candidates_with_scores if doc.metadata.get("source", "") == dominant_source]

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = await loop.run_in_executor(
            None, lambda: self.reranker.predict(pairs, batch_size=len(pairs))
        )
        ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results if score > self.score_threshold][:self.top_k]