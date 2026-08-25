import os, glob, asyncio
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
DB_DIR = "faiss_vector_store"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

async def load_data():
    docs = []

    for path in glob.glob(f"{DATA_DIR}/*.pdf"):
        pdf_loader = PyMuPDFLoader(file_path=path)
        parsed_docs = pdf_loader.lazy_load()
        docs.extend(parsed_docs)

    return docs

async def load_vector_store():
    if os.path.exists(DB_DIR):
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = await load_data()
    splitted_docs = text_splitter.split_documents(documents=docs)
    vector_store = FAISS.from_documents(documents=splitted_docs, embedding=embeddings)
    vector_store.save_local(DB_DIR)
    return vector_store

async def build_retriever():
    return (await load_vector_store()).as_retriever(search_kwargs={"k": 5})

if __name__ == "__main__":
    async def main():
        retriever = await build_retriever()
        results = retriever.invoke("Suggest the most optimized deep learning approaches to detect lung cancer")
        print("Retrieved results:")

        for res in results:
            print(res, "\n")

    asyncio.run(main())