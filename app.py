import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 1. Page Configuration
st.set_page_config(page_title="RAG AI Assistant", page_icon="📚", layout="wide")
st.title("📚 Custom Document Q&A Assistant")

# 2. Sidebar for Multi-File Upload
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
            else:
                loader = TextLoader(tmp_path)
            
            docs.extend(loader.load())
        except Exception as e:
            st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path) # Clean up temp file

    st.sidebar.success(f"Loaded {len(uploaded_files)} file(s) successfully!")
else:
    # Default fallback content when no files are uploaded
    docs = [Document(page_content="""Dinesh's Project Details:
Project Name: GenAI Assistant
Current Phase: Phase 6 - Multi-File Upload Support
Target Launch Date: November 15, 2026
Main Tech Stack: Python, LangChain, Gemini API, Streamlit, Vector DB.""")]
    st.sidebar.info("Using default project notes. Upload PDF or TXT files to customize.")

# 3. Initialize Models
@st.cache_resource
def get_models():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return llm, embeddings

llm, embeddings = get_models()

# 4. Process Knowledge Base & Create Vector Store
if docs:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.sidebar.error(f"Embedding Error: {e}")

# 5. Define RAG Prompt Template
rag_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using ONLY the provided context below.

Context:
{context}

Question: {question}
""")

chain = rag_prompt | llm | StrOutputParser()

# 6. Chat Interface
question = st.text_input("Ask a question across all uploaded documents:")

if st.button("Get Answer") and question:
    with st.spinner("Searching documents..."):
        try:
            retrieved_docs = retriever.invoke(question)
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])
            
            answer = chain.invoke({"context": context_text, "question": question})
            
            st.subheader("AI Answer:")
            st.write(answer)
            
            with st.expander("View Retrieved Context Chunks"):
                st.info(context_text)
        except Exception as e:
            st.error(f"Error generating answer: {e}")