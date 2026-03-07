from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

model_local = ChatOllama(model="mistral")

urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

print("Before RAG\n")
before_rag_template = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))

print("\n#########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""

after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("What is Ollama?"))





chromadb==1.0.8
# jsonlines==3.1.0
# jsonpatch==1.33
# jsonpointer==2.1
# jsonref==1.1.0
# keybert==0.9.0
# langchain==0.3.21
# langchain-community==0.3.20
# langchain-core==0.3.48
# langchain-ollama==0.3.0
# langchain-text-splitters==0.3.7
# langsmith==0.3.18
# latex2mathml==3.77.0
# mmh3==5.1.0
# monotonic==1.6
# ollama==0.4.7
# opencv-python-headless==4.11.0.86
# openpyxl==3.1.5
# opentelemetry-api==1.31.1
# opentelemetry-exporter-otlp-proto-common==1.31.1
# opentelemetry-exporter-otlp-proto-grpc==1.31.1
# opentelemetry-instrumentation==0.52b1
# opentelemetry-instrumentation-asgi==0.52b1
# opentelemetry-instrumentation-fastapi==0.52b1
# opentelemetry-proto==1.31.1
# opentelemetry-sdk==1.31.1
# opentelemetry-semantic-conventions==0.52b1
# opentelemetry-util-http==0.52b1
# pycurl==7.45.1
# pydantic==2.10.6
# pydantic-core==2.27.2
# pydantic-settings==2.8.1
# pyMuPDF
# streamlit==1.45.0
# tiktoken==0.9.0