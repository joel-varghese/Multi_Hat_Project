import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from graph_predict import predict_edges_from_text
import joblib
from langchain_core.prompts import ChatPromptTemplate

from pdfParse import extract_pdf_text  # your existing PDF parser

# Initialize the model and prompt chain
model_local = ChatOllama(model="mistral")

after_rag_template = """You are a {role}. Summarize the following content for yourself and speak in terms of first person.
Only include content relevant to that role like a resume summary.

Context:
{context}

Question: Give a one paragraph summary of the key skills a {role} can have from this document.
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

def format_query(input_dict):
    return f"Give a one paragraph summary of the key skills a {input_dict['role']} can have from this document."

def build_chain(retriever):
    return (
        {
            "context": format_query | retriever, 
            "role": lambda x: x["role"],   
        }
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

# Streamlit UI
st.set_page_config(page_title="PDF Role-Based Summarizer", layout="centered")
st.title("📄 PDF Role-Based Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
roles = [
    "Science and engineering professionals",
    "Chief executives, senior officials and legislators",
    "Health professionals",
    "Business and administration professionals",
    "Information and communications technology professionals",
    "Teaching professionals"
]
role = st.selectbox("Select an occupation role", roles)

if uploaded_file and role:
    with st.spinner("Processing..."):
        # Extract text from uploaded file
        pdf_text = extract_pdf_text(uploaded_file)

        # Prepare documents
        docs_list = [Document(page_content=pdf_text)]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        docs_splits = text_splitter.split_documents(docs_list)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=docs_splits,
            collection_name="rag-chroma-ui",  # separate name to avoid conflicts
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
        )
        retriever = vectorstore.as_retriever()

        # Run the RAG chain
        chain = build_chain(retriever)
        summary = chain.invoke({"role": role})
        le = joblib.load("label_encoder.pkl")
        skill2idx = joblib.load("skill2idx.pkl")
        skill_list = sorted(list(skill2idx.keys()))
        skills = predict_edges_from_text(pdf_text,"graphsage_model.pth",le,skill2idx,skill_list,
                                         use_semantic_mapping=True)
        
        st.subheader("Skills (GraphSAGE)")
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No skills extracted. Try uploading a different document or adjusting the role.")

        # Output the result
        st.subheader("Role-Based Summary: RAG")
        st.write(summary)
