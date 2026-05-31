from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR,"vectorstore")


def procesar_pdf(ruta_archivo: str) -> list:
    loader = PyPDFLoader(ruta_archivo)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documentos)
    return chunks

def crear_vectorstore(chunks: list, nombre_coleccion: str = "dochat") -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=nombre_coleccion
    )

    return vectorstore

def buscar_chunks(pregunta: str, vectorstore: Chroma, k: int = 6) -> list:
    resultados = vectorstore.similarity_search(pregunta, k=k)
    return resultados

