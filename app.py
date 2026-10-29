__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import shutil
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 1. Page Configuration
st.set_page_config(page_title="RAG Conversational Assistant", page_icon="💬", layout="wide")
st.title("💬 Conversational RAG Assistant with Source Citations")


CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")

# 2. Sidebar Controls & File Upload
st.sidebar.header("Database & Session Controls")

# Clear Chat History Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared!")

# Reset Vector Database Button
if st.sidebar.button("🔥 Reset Vector Database"):
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            st.cache_resource.clear()
            st.sidebar.success("Vector database wiped successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error resetting database: {e}")

st.sidebar.divider()
st.sidebar.header("Knowledge Base Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents (.txt or .pdf)", 
    type=["txt", "pdf"], 
    accept_multiple_files=True
)

docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                # Attach original filename to document metadata
                for d in loaded_docs:
                    d.metadata["source"] = uploaded_file.name
                docs.extend(loaded_docs)
            else:
                loader = TextLoader(tmp_path)
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = uploaded_file.name
                docs.extend(loaded_docs)
        except Exception as e:
            st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    st.sidebar.success(f"Loaded {len(uploaded_files)} file(s) successfully!")
else:
    docs = [Document(
        page_content="""Dinesh's Project Details:
Project Name: GenAI Assistant
Current Phase: Phase 9 - Production RAG App with Citations & Memory
Target Launch Date: November 15, 2026
Main Tech Stack: Python, LangChain, Gemini API, Streamlit, ChromaDB.""",
        metadata={"source": "default_project_notes.txt", "page": 1}
    )]
    st.sidebar.info("Using default project notes. Upload PDF or TXT files to customize.")

# 3. Initialize Models
@st.cache_resource
def get_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return llm, embeddings

llm, embeddings = get_models()

# 4. Process Knowledge Base & Vector Store
# 4. Process Knowledge Base & Vector Store (In-Memory persistent cache to prevent read-only disk errors)
@st.cache_resource
def get_vectorstore(docs_hash):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    # Using InMemoryVectorStore prevents SQLite/Chroma file-locking & readonly disk crashes on Streamlit Cloud
    vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
    return vectorstore

if docs:
    try:
        # Create a simple hash based on document lengths and contents to cache efficiently
        docs_hash = sum(len(d.page_content) for d in docs)
        vectorstore = get_vectorstore(docs_hash)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.sidebar.error(f"Embedding Error: {e}")

# 5. Define Conversational RAG Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant helping with document Q&A.\n"
               "Use ONLY the following retrieved context to answer the user's question accurately.\n"
               "If the context does not contain sufficient information, state:\n"
               "'The uploaded documents do not contain enough information to answer this question.'\n\n"
               "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = rag_prompt | llm | StrOutputParser()

# 6. Streamlit Chat Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history along with stored citation metadata
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Document Citations & Sources"):
                for cit in message["citations"]:
                    st.write(f"• **Source:** `{cit['source']}` | **Page:** `{cit['page']}`")
                    st.caption(f"_{cit['snippet']}_")

# Handle user input via chat bar
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                retrieved_docs = retriever.invoke(prompt)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Format session history
                history_tuples = [
                    (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                ]
                
                answer = chain.invoke({
                    "context": context_text,
                    "chat_history": history_tuples,
                    "question": prompt
                })
                
                # Parse unique citations and chunk metadata
                citations = []
                for doc in retrieved_docs:
                    source_name = doc.metadata.get("source", "Unknown Document")
                    page_num = doc.metadata.get("page", 1)
                    if isinstance(page_num, int):
                        page_num = page_num + 1  # 0-indexed to 1-indexed conversion
                    
                    citations.append({
                        "source": source_name,
                        "page": page_num,
                        "snippet": doc.page_content[:150] + "..."
                    })

                st.markdown(answer)
                
                if citations:
                    with st.expander("📚 View Document Citations & Sources"):
                        for cit in citations:
                            st.write(f"• **Source:** `{cit['source']}` | **Page:** `{cit['page']}`")
                            st.caption(f"_{cit['snippet']}_")

                # Store assistant response and citations in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")