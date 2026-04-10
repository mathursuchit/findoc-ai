import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
CHROMA_DIR = "chroma_store"

# some starter questions that work well for earnings reports / 10-Ks
EXAMPLE_QUESTIONS = [
    "What are the main revenue streams?",
    "What risk factors does the company highlight?",
    "What was the net income for the reported period?",
    "Who are the key executives mentioned?",
    "What is the company's outlook for next year?",
]

st.set_page_config(page_title="FinDoc AI", page_icon="📊", layout="wide")
st.title("📊 FinDoc AI — Financial Document Analyst")
st.caption("Upload SEC filings, earnings reports, or annual reports and chat with them.")

# init session state keys if they don't exist yet
for key, default in [
    ("messages", []),
    ("chat_history", []),
    ("chain", None),
    ("retriever", None),
    ("indexed_files", []),
    ("llm", None),
    ("embeddings", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# only load these once per session - they're slow to initialize
if st.session_state.llm is None:
    st.session_state.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )
if st.session_state.embeddings is None:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_chain(retriever, llm):
    # without this step, follow-up questions like "what did he say about that?"
    # fail because the retriever has no context about what "that" refers to
    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Rewrite the above as a standalone question. Return only the rewritten question."),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    def get_question(d):
        if d.get("chat_history"):
            return condense_chain.invoke(d)
        return d["input"]

    answer_prompt = ChatPromptTemplate.from_template(
        "You are a financial analyst assistant. "
        "Answer using only the context below. "
        "If the answer isn't there, say so — don't make things up.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        # include filename + page so the LLM knows where each chunk came from
        return "\n\n".join(
            f"[{doc.metadata.get('source_file', 'unknown')} | "
            f"Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
            for doc in docs
        )

    return (
        RunnableLambda(get_question)
        | {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )


with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload financial PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            if st.button("Index Documents", type="primary"):
                with st.spinner("Indexing..."):
                    all_chunks = []
                    for file in new_files:
                        tmp_path = f"tmp_{file.name}"
                        with open(tmp_path, "wb") as f:
                            f.write(file.read())

                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()

                        for page in pages:
                            page.metadata["source_file"] = file.name

                        # 1000/200 worked well in testing - smaller chunks hurt
                        # recall, larger chunks hurt precision
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200
                        )
                        chunks = splitter.split_documents(pages)
                        all_chunks.extend(chunks)
                        st.session_state.indexed_files.append(file.name)
                        os.remove(tmp_path)

                    if os.path.exists(CHROMA_DIR):
                        vectorstore = Chroma(
                            persist_directory=CHROMA_DIR,
                            embedding_function=st.session_state.embeddings
                        )
                        vectorstore.add_documents(all_chunks)
                    else:
                        vectorstore = Chroma.from_documents(
                            all_chunks,
                            st.session_state.embeddings,
                            persist_directory=CHROMA_DIR
                        )

                    # MMR helps when querying across multiple docs - plain similarity
                    # search tends to return all chunks from the same document
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
                    )
                    st.session_state.retriever = retriever
                    st.session_state.chain = build_chain(retriever, st.session_state.llm)

                st.success(f"Indexed {len(all_chunks)} chunks.")

    if st.session_state.indexed_files:
        st.divider()
        st.markdown("**Indexed documents:**")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- {fname}")

    st.divider()
    st.markdown("**Try asking:**")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- *{q}*")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()


if not st.session_state.chain:
    st.info("Upload financial documents in the sidebar and click **Index Documents** to start.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['file']}** — Page {src['page']}\n\n> {src['text']}")

    question = st.chat_input("Ask about the documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            container = st.empty()
            answer = ""

            with st.spinner("Retrieving context..."):
                source_docs = st.session_state.retriever.invoke(question)

            for token in st.session_state.chain.stream({
                "input": question,
                "chat_history": st.session_state.chat_history
            }):
                answer += token
                container.markdown(answer + "▌")
            container.markdown(answer)

            sources = [
                {
                    "file": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "text": doc.page_content[:300]
                }
                for doc in source_docs
            ]
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"**{src['file']}** — Page {src['page']}\n\n> {src['text']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
