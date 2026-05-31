import streamlit as st
import os
import tempfile
from rag import procesar_pdf, crear_vectorstore, buscar_chunks
from memory import crear_memoria, agregar_a_memoria, formatear_historial
from langchain_ollama import OllamaLLM

def iniciar_ollama():
    import subprocess
    import time
    try:
        subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
        )
        time.sleep(2)
    except FileNotFoundError:
        st.error("Ollama no está instalado. Descárgalo en https://ollama.com")
        st.stop()

def limpiar_voctorstore():
    import shutil
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectorstore_path = os.path.join(BASE_DIR, "vectorstore")
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
    os.makedirs(vectorstore_path)


def inicializar_estado():
    if "vectorstore" not in st.session_state:
        limpiar_voctorstore()
        st.session_state.vectorstore = None
    if "memoria" not in st.session_state:
        st.session_state.memoria = crear_memoria()
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

def procesar_archivo(archivo):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(archivo.read())
        ruta_tmp = tmp.name

    chunks = procesar_pdf(ruta_tmp)
    
    nombre = os.path.splitext(archivo.name)[0]
    nombre = nombre.replace(" ", "_").lower()
    
    vectorstore = crear_vectorstore(chunks, nombre_coleccion=nombre)
    
    os.unlink(ruta_tmp)
    
    return vectorstore

def main():
    iniciar_ollama()
    inicializar_estado()
    st.title("🤖 DoChat")
    st.caption("Chatea con tus documentos de forma local y privada")

    inicializar_estado()

    with st.sidebar:
        st.header("📄 Tus documentos")
        archivos = st.file_uploader(
            "Arrastra tus PDFs aquí",
            type="pdf",
            accept_multiple_files=True
        )

        if archivos:
            if st.button("Procesar PDFs"):
                with st.spinner("Procesando documentos..."):
                    for archivo in archivos:
                        st.session_state.vectorstore = procesar_archivo(archivo)
                st.success(f"✅ {len(archivos)} documento(s) listo(s)")

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.write(mensaje["contenido"])

    pregunta = st.chat_input("Escribe tu pregunta...")

    if pregunta:
        st.session_state.mensajes.append({
            "rol": "user",
            "contenido": pregunta
        })

        with st.chat_message("user"):
            st.write(pregunta)

        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                st.warning("Primero sube y procesa un PDF en el panel izquierdo.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    chunks_relevantes = buscar_chunks(
                        pregunta,
                        st.session_state.vectorstore
                    )
                    contexto = "\n\n".join([c.page_content for c in chunks_relevantes])
                    historial = formatear_historial(st.session_state.memoria)

                    llm = OllamaLLM(model="llama3.2")

                    prompt = f"""Eres un asistente que responde preguntas ÚNICAMENTE basándote en el contexto proporcionado.
Si la respuesta no está en el contexto, di exactamente: "No encontré esa información en el documento."
No uses conocimiento externo. No inventes información.

Contexto del documento:
{contexto}

Historial de conversación:
{historial}

Pregunta: {pregunta}
Respuesta:"""

                    respuesta = llm.invoke(prompt)

                    st.write(respuesta)

                    st.session_state.memoria = agregar_a_memoria(
                        st.session_state.memoria,
                        pregunta,
                        respuesta
                    )
                    st.session_state.mensajes.append({
                        "rol": "assistant",
                        "contenido": respuesta
                    })


if __name__ == "__main__":
    main()