import os

from dotenv import load_dotenv
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.memory_manager import criar_memoria
from app.prompts import NUTRITION_SYSTEM_PROMPT


# Carrega as variáveis do arquivo .env
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

if not api_key:
    raise ValueError(
        "OLLAMA_API_KEY não encontrada. "
        "Crie o arquivo .env e informe sua chave da Ollama."
    )


llm = ChatOllama(
    model="gemma4:cloud",
    temperature=0.3,
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }
)


# MEMÓRIA

memoria = criar_memoria()


# PROMPT DA CONVERSA

prompt_conversa = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            NUTRITION_SYSTEM_PROMPT
        ),
        (
            "human",
            """
<historico>
{history}
</historico>

<mensagem_atual>
{input}
</mensagem_atual>

Responda considerando a mensagem atual e, quando relevante,
as informações presentes no histórico da conversa.
"""
        )
    ]
)


# CONVERSATION CHAIN COM MEMÓRIA

chat_chain = ConversationChain(
    llm=llm,
    prompt=prompt_conversa,
    memory=memoria,
    input_key="input",
    output_key="response",
    verbose=False
)


def responder(mensagem: str) -> str:
    """
    Envia uma mensagem para o chatbot e devolve a resposta.

    A ConversationChain salva automaticamente a interação
    na memória após cada chamada.
    """

    resultado = chat_chain.invoke(
        {
            "input": mensagem
        }
    )

    return resultado["response"]


def apagar_historico() -> None:
    """
    Apaga o histórico atual da conversa.
    """

    memoria.clear()
