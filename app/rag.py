from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def procesar_pdf(ruta_archivo: str) -> list:
    loader = PyPDFLoader(ruta_archivo)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documentos)
    return chunks

def crear_vectorstore(chunks: list) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return vectorstore

def buscar_chunks(pregunta: str, vectorstore: Chroma, k: int = 3) -> list:
    resultados = vectorstore.similarity_search(pregunta, k=k)
    return resultados