from langchain_classic.memory import ConversationBufferMemory


def criar_memoria() -> ConversationBufferMemory:
    """
    Cria a memória conversacional do chatbot de nutrição.

    A ConversationBufferMemory mantém o histórico completo
    da conversa, permitindo que o chatbot se lembre de
    informações fornecidas pelo usuário em turnos anteriores.
    """

    return ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        return_messages=False
    )


def limpar_memoria(memoria: ConversationBufferMemory) -> None:
    """
    Limpa todo o histórico armazenado na memória.
    """
    memoria.clear()
