from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
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

def crear_vectorstore(chunks: list, modelo: str = "llama3") -> Chroma:
    embeddings = OllamaEmbeddings(model=modelo)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embeddind=embeddings,
        persist_directory="vectorstore"
    )

    return vectorstore

def buscar_chunks(pregunta: str, vectorstore: Chroma, k: int = 3) -> list:
    resultados = vectorstore.similarity_search(pregunta, k=k)
    return resultados