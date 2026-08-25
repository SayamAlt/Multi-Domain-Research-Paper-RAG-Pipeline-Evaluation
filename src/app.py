import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import time
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Research Paper QA",
    page_icon="📄",
    layout="centered",
)

@st.cache_resource(show_spinner="Loading retriever and reranker...")
def get_pipeline(fetch_k: int, top_k: int) -> RAGPipeline:
    return asyncio.run(RAGPipeline.create(fetch_k=fetch_k, top_k=top_k))

with st.sidebar:
    st.header("Retrieval settings")
    fetch_k = st.slider(
        "fetch_k - candidates from vector store",
        min_value=5,
        max_value=30,
        value=15,
        help="How many chunks the retriever fetches before reranking.",
    )
    top_k = st.slider(
        "top_k - chunks kept after reranking",
        min_value=1,
        max_value=10,
        value=5,
        help="How many chunks the reranker keeps and passes to the generator.",
    )
    if top_k > fetch_k:
        st.warning("top_k is larger than fetch_k. You will get at most fetch_k chunks.")

    st.divider()
    show_context = st.toggle("Show retrieved context", value=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are grounded only in the loaded research papers. "
        "Questions outside that scope will be declined."
    )

st.title("Research Paper QA")
st.caption(
    "Ask questions about the loaded research papers on lung cancer detection, "
    "NSCLC microbiome classification, and Bhutan land cover."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_context(context: list[str], latency: float | None = None) -> None:
    label = f"Retrieved chunks: {len(context)}"
    if latency is not None:
        label += f" ({latency:.1f}s)"
    with st.expander(label):
        for i, chunk in enumerate(context, start=1):
            st.markdown(f"**Chunk {i}**")
            st.text(chunk)
            if i != len(context):
                st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and show_context
            and message.get("context")
        ):
            render_context(message["context"], message.get("latency"))

query = st.chat_input("Ask a question about the research papers...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            rag = get_pipeline(fetch_k, top_k)
            with st.spinner("Retrieving and generating..."):
                started = time.perf_counter()
                result = asyncio.run(rag.invoke(query))
                latency = time.perf_counter() - started
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.session_state.messages.pop()
        else:
            answer = result["answer"]
            context = result["context"]
            st.markdown(answer)
            if show_context:
                render_context(context, latency)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                    "latency": latency,
                }
            )