from rag import procesar_pdf, crear_vectorstore, buscar_chunks
from memory import crear_memoria, agregar_a_memoria, formatear_historial
from langchain_ollama import OllamaLLM


def iniciar_chat(ruta_pdf: str):
    print("Procesando PDF...")
    chunks = procesar_pdf(ruta_pdf)

    print("Creando base de datos vectorial...")
    vectorstore = crear_vectorstore(chunks)

    memoria = crear_memoria()
    llm = OllamaLLM(model="llama3.2")

    print("¡Listo! Escribe tu pregunta (o 'salir' para terminar)\n")

    while True:
        pregunta = input("Tú: ")

        if pregunta.lower() == "salir":
            break

        chunks_relevantes = buscar_chunks(pregunta, vectorstore)
        contexto = "\n\n".join([c.page_content for c in chunks_relevantes])

        historial = formatear_historial(memoria)

        prompt = f"""Usa el siguiente contexto para responder la pregunta.
        
Contexto:
{contexto}

Historial de conversación:
{historial}

Pregunta: {pregunta}
Respuesta:"""

        respuesta = llm.invoke(prompt)
        memoria = agregar_a_memoria(memoria, pregunta, respuesta)
        print(f"\nDoChat: {respuesta}\n")


if __name__ == "__main__":
    import sys
    iniciar_chat(sys.argv[1])