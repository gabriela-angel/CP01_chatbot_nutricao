import gradio as gr

from app.chain import responder


def conversar(mensagem, historico):
    """
    Função utilizada pela interface Gradio.

    O parâmetro historico pertence à interface visual.
    A memória real da IA é gerenciada pela ConversationChain.
    """

    return responder(mensagem)


demo = gr.ChatInterface(
    fn=conversar,
    title="🥗 NutriBot",
    description=(
        "Chatbot educativo sobre nutrição e alimentação saudável. "
        "As informações fornecidas são de caráter geral e não substituem "
        "acompanhamento profissional individualizado."
    )
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
