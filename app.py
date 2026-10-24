import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="RAG AI Assistant", page_icon="📚", layout="wide")
st.title("📚 Custom Document Q&A Assistant")

# 2. Sidebar for Document Knowledge Base
st.sidebar.header("Knowledge Base Input")
knowledge_text = st.sidebar.text_area(
    "Paste custom document text here:",
    height=250,
    value="""Dinesh's Project Details:
Project Name: GenAI Assistant
Current Phase: Phase 3 - Streamlit Web UI
Target Launch Date: November 15, 2026
Main Tech Stack: Python, LangChain, Gemini API, Streamlit, Vector DB."""
)

# 3. Initialize Models
@st.cache_resource
def get_models():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm, embeddings

llm, embeddings = get_models()

# 4. Process Knowledge Base & Create Vector Store
if knowledge_text:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    docs = text_splitter.create_documents([knowledge_text])
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    st.sidebar.success("Knowledge Base Processed Successfully!")

# 5. Define RAG Prompt Template
rag_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using ONLY the provided context below.

Context:
{context}

Question: {question}
""")

chain = rag_prompt | llm | StrOutputParser()

# 6. Chat Interface
question = st.text_input("Ask a question based on your pasted knowledge base:")

if st.button("Get Answer") and question:
    with st.spinner("Searching knowledge base..."):
        # Retrieve context matching the user question
        retrieved_docs = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate Answer
        answer = chain.invoke({"context": context_text, "question": question})
        
        # Display Results
        st.subheader("AI Answer:")
        st.write(answer)
        
        with st.expander("View Retrieved Context Chunks"):
            st.info(context_text)