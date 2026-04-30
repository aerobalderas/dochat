from langchain.memory import ConversationBufferWindowMemory

def crear_memoria(k: int = 5) -> ConversationBufferWindowMemory:
    memoria = ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True
    )
    return memoria