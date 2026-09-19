from src.reranker import RerankingRetriever
from src.generator import generate_answer
from langsmith import traceable
import asyncio

class RAGPipeline:
    @classmethod
    async def create(cls, fetch_k=15, top_k=5, score_threshold=0.0):
        instance = cls.__new__(cls)
        # 1 retriever instance - loads the vector store along with the reranker model once
        instance.retriever = await RerankingRetriever.create(fetch_k=fetch_k, top_k=top_k, score_threshold=score_threshold)
        return instance
        
    @traceable(run_type="chain", name="RAG Pipeline")
    async def invoke(self, query: str) -> dict:
        # Over-retrieve fetch_k docs and then rerank down to top_k docs
        docs = await self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]
        # Generate grounded answer from the retrieved context
        answer = await generate_answer(query, context)
        return {
            "query": query,
            "context": context,
            "answer": answer
        }
    
if __name__ == "__main__":
    async def main():
        rag_pipeline = await RAGPipeline.create()
        result = await rag_pipeline.invoke("What preprocessing steps were applied to Sentinel-2 imagery before training the U-Net model, and which spectral bands were selected for land cover classification?")
        print("Query:", result["query"])
        print("Answer:", result["answer"])
        print("\nRetrieved context chunks:")
        
        for idx, chunk in enumerate(result["context"]):
            print(f"[{idx}] {chunk[:120]}...")
            
    asyncio.run(main())