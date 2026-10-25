__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 1. Page Configuration
st.set_page_config(page_title="RAG Conversational Assistant", page_icon="💬", layout="wide")
st.title("💬 Conversational RAG Assistant")

CHROMA_PATH = "./chroma_db"

# 2. Sidebar Management
st.sidebar.header("Knowledge Base Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents (.txt or .pdf)", 
    type=["txt", "pdf"], 
    accept_multiple_files=True
)

# Sidebar Clear History Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared!")

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
                os.unlink(tmp_path)

    st.sidebar.success(f"Loaded {len(uploaded_files)} file(s) successfully!")
else:
    docs = [Document(page_content="""Dinesh's Project Details:
Project Name: GenAI Assistant
Current Phase: Phase 8 - Conversational Chat Memory
Target Launch Date: November 15, 2026
Main Tech Stack: Python, LangChain, Gemini API, Streamlit, ChromaDB.""")]
    st.sidebar.info("Using default project notes. Upload PDF or TXT files to customize.")

# 3. Initialize Models
@st.cache_resource
def get_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return llm, embeddings

llm, embeddings = get_models()

# 4. Process Knowledge Base & Vector Store
if docs:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
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

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input via chat bar
if prompt := st.chat_input("Ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retrieved_docs = retriever.invoke(prompt)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Format session state messages for LangChain prompt
                history_tuples = [
                    (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                ]
                
                answer = chain.invoke({
                    "context": context_text,
                    "chat_history": history_tuples,
                    "question": prompt
                })
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating answer: {e}")