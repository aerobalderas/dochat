
def crear_memoria() -> list:
    return[]

def agregar_a_memoria(memoria: list, pregunta: str, respuesta: str) -> list:
    memoria.append({"input": pregunta, "output": respuesta})
    if len(memoria) > 5:
        memoria.pop(0)
    return memoria

def formatear_historial(memoria: list) -> str:
    if not memoria:
        return "Sin historial previo."
    
    historial = ""
    for intercambio in memoria:
        historial += f"Usuario: {intercambio['input']}\n"
        historial += f"DoChat: {intercambio['output']}\n\n"
    return historial