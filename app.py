import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="RAG AI Assistant", page_icon="📚", layout="wide")
st.title("📚 Custom Document Q&A Assistant")

# 2. Sidebar for File Upload
st.sidebar.header("Knowledge Base Upload")
uploaded_file = st.sidebar.file_uploader("Upload a text document (.txt)", type=["txt"])

# Use uploaded text or fallback to default project info if no file is uploaded
if uploaded_file is not None:
    knowledge_text = uploaded_file.read().decode("utf-8")
    st.sidebar.success("File uploaded successfully!")
else:
    knowledge_text = """Dinesh's Project Details:
Project Name: GenAI Assistant
Current Phase: Phase 4 - Custom File Upload
Target Launch Date: November 15, 2026
Main Tech Stack: Python, LangChain, Gemini API, Streamlit, Vector DB."""
    st.sidebar.info("Using default project notes. Upload a .txt file to customize.")

# 3. Initialize Models
@st.cache_resource
def get_models():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    return llm, embeddings

llm, embeddings = get_models()

# 4. Process Knowledge Base & Create Vector Store
if knowledge_text:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        docs = text_splitter.create_documents([knowledge_text])
        vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
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
question = st.text_input("Ask a question based on your document:")

if st.button("Get Answer") and question:
    with st.spinner("Searching document..."):
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