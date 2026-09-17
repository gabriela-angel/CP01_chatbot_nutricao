"""
context_rot.py
===============
CKP01 - Chatbot Profissional | Domínio: Assistente de Nutrição/Dieta
Responsável: Tiago — Parte 4 (Context Rot + Tokens)

DECISÃO DO GRUPO SOBRE MEMÓRIA (importante para entender este arquivo)
-----------------------------------------------------------------------
O grupo decidiu usar ConversationBufferMemory (histórico completo, sem
limite de tokens) no chatbot real — ver app/memory_manager.py (Wesley).
A justificativa é que, neste domínio, perder uma restrição de segurança
(ex.: alergia alimentar) por causa de um corte de memória é inaceitável;
preferem pagar mais tokens a arriscar isso. Este módulo NÃO importa nem
depende de memory_manager.py — é só contexto para entender por que o
experimento abaixo mede "lost in the middle" em vez de "corte de memória".

O QUE ESTE MÓDULO FAZ
----------------------
1. Demonstra "context rot": pega a MESMA pergunta final e testa em janelas
   de contexto de tamanhos diferentes (0, 5, 10, 15 e 20 turnos de histórico
   acumulado), verificando se o chatbot ainda respeita uma restrição
   importante dita NO MEIO da conversa (alergia a amendoim, no turno 8 de
   22). Como a memória real (BufferMemory) nunca descarta nada, não faz
   sentido testar "a restrição foi cortada?" — em vez disso testamos dois
   efeitos de POSIÇÃO/ATENÇÃO: (a) "lost in the middle" — restrição
   enterrada longe do início/fim — e (b) INTERFERÊNCIA POR DISTRATORES —
   duas restrições parecidas, mas de OUTRAS pessoas (esposa/pai), nos
   turnos 12 e 16, para ver se o modelo confunde de quem é a restrição.
2. Mede, com tiktoken, quantos tokens cada janela consome.
3. Compara, de forma HIPOTÉTICA/exploratória, "sem gestão de memória"
   (histórico cru crescendo sem limite — o que o chatbot REALMENTE faz)
   com um cenário "e se usássemos TokenBufferMemory" (simulado, mantendo
   só as mensagens mais recentes dentro de um limite de tokens) — mostra a
   % de redução que isso teria, mesmo não sendo a memória escolhida.
4. [Bônus/diferencial] Meta prompting: usa o próprio LLM para reescrever o
   system prompt de forma mais enxuta e compara tokens antes/depois.

Este arquivo é INDEPENDENTE (roda sozinho com `python -m app.context_rot`)
mas tenta importar o SYSTEM_PROMPT real do Marcos (app/prompts.py). Se esse
arquivo ainda não existir, usa um prompt provisório só para não travar.

DEPENDÊNCIAS (adicionar ao requirements.txt do grupo):
    tiktoken
    matplotlib
    python-dotenv
    langchain-ollama
    langchain-core
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import tiktoken
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# 0. Configuração — igual ao exigido pelo enunciado do CKP01
#    O .env é procurado explicitamente na MESMA pasta deste arquivo
#    (app/.env) — não depende de onde o script é executado (útil para
#    quem roda pelo VS Code com a pasta app/ aberta, por exemplo).
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
os.environ.setdefault("OLLAMA_HOST", "https://ollama.com")
# OLLAMA_API_KEY precisa estar no .env (nunca no código!)

# Console do Windows costuma usar um codepage legado (cp1252) que não
# imprime acentos/emojis corretamente. Forçamos UTF-8 na saída padrão para
# os prints deste módulo saírem certos em qualquer terminal.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # streams que não suportam reconfigure (ex.: capturados por certos runners)

MODEL_NAME = "gemma4:cloud"  # modelo obrigatório do CKP01, via Ollama Cloud

# TEMPERATURE=0 deixa a demonstração mais reprodutível (menos variação
# entre execuções). Não elimina 100% a variação do LLM, mas reduz bastante.
# IMPORTANTE: os números do README vêm de UMA execução específica com essa
# configuração — não rodamos múltiplas vezes para tirar média (ver README).
TEMPERATURE = 0

# Valor ILUSTRATIVO usado só na simulação HIPOTÉTICA de TokenBufferMemory
# (funções aplicar_janela_token_buffer / comparar_com_sem_gestao). Não
# corresponde a nenhuma configuração real: o grupo decidiu usar
# ConversationBufferMemory (sem limite de tokens) no chatbot de verdade
# (app/memory_manager.py, Wesley), então não existe um "valor real" para
# alinhar aqui. Serve apenas para dar um número concreto à pergunta
# exploratória "e se a gente limitasse por tokens?".
LIMITE_TOKENS_MEMORIA = 800


def _criar_llm() -> ChatOllama:
    """Fábrica única do LLM usado neste módulo, já com temperature fixa."""
    return ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)


# ---------------------------------------------------------------------------
# 1. System prompt do domínio — vem do Marcos (app/prompts.py)
#    Se ainda não estiver pronto, usamos um placeholder pra não travar o dev.
# ---------------------------------------------------------------------------
try:
    from app.prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = """<persona>
Você é o NutriBot, assistente virtual de nutrição e dieta, com tom
acolhedor e profissional.
</persona>
<restricoes>
- Nunca sugira dietas extremas, restritivas demais ou perigosas.
- Sempre respeite alergias e restrições alimentares informadas pelo usuário.
- Deixe claro que você não substitui a consulta com um nutricionista humano.
</restricoes>
<formato>
Responda em português, de forma objetiva, usando listas quando fizer sentido.
</formato>"""
    print("[context_rot] Aviso: app/prompts.py ainda não encontrado — "
          "usando persona provisória. Troque pelo SYSTEM_PROMPT do Marcos "
          "assim que ele terminar a parte dele.")

# ---------------------------------------------------------------------------
# 2. Contador de tokens (tiktoken)
#    cl100k_base é o encoding usado pela família GPT-4/3.5 da OpenAI; não é
#    o tokenizer exato do Gemma/Ollama, mas serve como aproximação padrão
#    para medir e COMPARAR tamanhos de contexto — é isso que o CKP01 pede.
# ---------------------------------------------------------------------------
_encoder = tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    """Retorna a contagem aproximada de tokens de um texto."""
    return len(_encoder.encode(texto))


# ---------------------------------------------------------------------------
# 3. Histórico sintético — simula uma conversa real do domínio de nutrição.
#    IMPORTANTE: o FATO CRÍTICO (alergia a amendoim + não gosta de peixe)
#    fica no MEIO da conversa (turno 8 de 22), cercado de turnos "neutros"
#    (sem relação com a restrição) antes e depois. Isso é proposital: com
#    ConversationBufferMemory o histórico nunca é cortado, então o risco
#    real não é "a restrição foi esquecida por falta de espaço", e sim
#    efeitos que dependem só de POSIÇÃO/COMPETIÇÃO por atenção dentro de
#    um contexto longo, mesmo com a informação presente inteira:
#      - "lost in the middle": restrição enterrada no meio, longe do
#        início e do fim (turno 8) — testado antes, não degradou.
#      - INTERFERÊNCIA POR DISTRATORES (turnos 12 e 16): duas restrições
#        alimentares parecidas, mas de OUTRAS pessoas (esposa com
#        intolerância a lactose, pai diabético), com fraseado similar ao
#        da restrição real ("esqueci de dizer/avisar..."), para ver se o
#        modelo passa a confundir "de quem" é cada restrição.
#    O primeiro turno é neutro de propósito (sem restrição nenhuma).
# ---------------------------------------------------------------------------
HISTORICO: List[Tuple[str, str]] = [
    ("human", "Oi! Quero montar uma dieta, pode me ajudar?"),
    ("ai", "Claro! Fico feliz em ajudar. Para começar, qual é o seu objetivo com a dieta?"),
    ("human", "Quero emagrecer. Peso 82kg, minha meta é 75kg em 3 meses."),
    ("ai", "Ótimo, isso pede um déficit calórico moderado, entre 300 e 500kcal por dia."),
    ("human", "Treino musculação 4x por semana, sempre de manhã."),
    ("ai", "Vou considerar isso para calcular sua necessidade de proteína diária."),
    ("human", "Não tenho muito tempo pra cozinhar durante a semana."),
    ("ai", "Vou priorizar receitas rápidas e práticas para os dias úteis."),
    ("human", "Nos finais de semana posso cozinhar algo mais elaborado."),
    ("ai", "Combinado, vou sugerir receitas mais trabalhosas para sábado e domingo."),
    ("human", "Qual a diferença entre proteína whey e caseína?"),
    ("ai", "Whey é absorvida rápido (ótima pós-treino); caseína é de absorção lenta (antes de dormir)."),
    ("human", "Posso tomar café à noite?"),
    ("ai", "Com moderação — cafeína à noite pode atrapalhar o sono de algumas pessoas."),
    # --- Turno 8/20: restrição crítica, introduzida no MEIO da conversa ---
    ("human", "Ah, esqueci de avisar uma coisa importante: sou alérgico a amendoim e não gosto de peixe."),
    ("ai", "Que bom que você avisou! Vou anotar: nada de amendoim (alergia) nem peixe nas sugestões daqui pra frente."),
    ("human", "Quantos litros de água eu deveria beber por dia?"),
    ("ai", "Uma referência comum é 35ml por kg de peso corporal, ajustável ao volume de treino."),
    ("human", "O que é índice glicêmico?"),
    ("ai", "É a velocidade com que um alimento eleva a glicose no sangue após ser consumido."),
    ("human", "Me indica um lanche da tarde saudável?"),
    ("ai", "Uma fruta com iogurte natural ou uma porção de castanhas costuma funcionar bem."),
    # --- Turno 12/22: DISTRATOR — restrição parecida, mas de OUTRA pessoa ---
    ("human", "Ah, esqueci de dizer - minha esposa tem intolerância a lactose, então em casa já cortamos bastante laticínio."),
    ("ai", "Entendido, bom saber do contexto lá de casa! Mas isso não muda nada nas suas próprias orientações."),
    ("human", "E se eu quiser algo mais doce no lanche?"),
    ("ai", "Um chocolate 70% cacau em pequena quantidade é uma opção equilibrada."),
    ("human", "Vale a pena tomar creatina?"),
    ("ai", "Sim, é um dos suplementos mais estudados e seguros para quem treina musculação."),
    ("human", "Quanto tempo depois do treino eu devo comer?"),
    ("ai", "Idealmente em até 1-2 horas, para otimizar a recuperação muscular."),
    # --- Turno 16/22: DISTRATOR — restrição parecida, mas de OUTRA pessoa ---
    ("human", "Meu pai é diabético e evita doce, então quando cozinho pra família penso nisso também."),
    ("ai", "Faz sentido pensar nisso por ele, mas nas suas próprias refeições o foco continua sendo a sua meta de emagrecimento."),
    ("human", "Posso beber álcool socialmente durante a dieta?"),
    ("ai", "Ocasionalmente sim, mas vale lembrar que o álcool tem bastante caloria vazia."),
    ("human", "Como faço pra não perder massa muscular emagrecendo?"),
    ("ai", "Manter a ingestão de proteína alta e continuar treinando com carga é essencial."),
    ("human", "Faz sentido pesar a comida no início?"),
    ("ai", "Sim, ajuda bastante a calibrar as porções nas primeiras semanas."),
    ("human", "Posso pular o café da manhã às vezes?"),
    ("ai", "Pode, o que importa mais é o total calórico e de nutrientes do dia."),
    ("human", "Qual a diferença entre gordura boa e gordura ruim?"),
    ("ai", "Gorduras insaturadas (azeite, castanhas) são benéficas; as trans devem ser evitadas."),
    ("human", "Suco de fruta conta como uma das frutas do dia?"),
    ("ai", "Parcialmente — o suco perde fibra, então a fruta inteira é sempre melhor."),
]

PERGUNTA_FINAL = "Baseado em tudo que conversamos, monte um cardápio de jantar para hoje."


def montar_mensagens(n_turnos: int) -> list:
    """
    Monta a lista (system, ...histórico..., pergunta final) usando os
    N PRIMEIROS turnos do histórico sintético (simula o crescimento natural
    de uma conversa, do início pro fim). n_turnos=0 → só a pergunta, sem
    nenhum histórico.
    """
    turnos = HISTORICO[: n_turnos * 2]  # cada turno = 1 human + 1 ai
    mensagens = [("system", SYSTEM_PROMPT)]
    mensagens.extend(turnos)
    mensagens.append(("human", PERGUNTA_FINAL))
    return mensagens


def montar_prompt_texto(n_turnos: int) -> str:
    """Concatena tudo em uma única string só para contar tokens."""
    partes = [f"{papel.upper()}: {conteudo}" for papel, conteudo in montar_mensagens(n_turnos)]
    return "\n".join(partes)


def avaliar_resposta(resposta: str, llm: ChatOllama) -> dict:
    """
    Roda DUAS avaliações independentes da mesma resposta, usando o próprio
    LLM como juiz (em vez de busca de substring):

    1) VIOLAÇÃO DIRETA: a resposta recomenda amendoim ou peixe (restrição
       real do usuário, dita no meio da conversa)?
       Por que por LLM e não substring: a versão antiga fazia
       `"amendoim" in resposta` — dava FALSO POSITIVO quando o bot só
       CITAVA a palavra (ex.: "você tem alguma alergia, como a amendoim?"
       ao pedir mais informações, sem recomendar nada) e FALSO NEGATIVO
       quando sugeria peixe por outro nome (ex.: "filé de tilápia"). O
       juiz por LLM julga a INTENÇÃO da resposta, não a presença literal
       da palavra.
    2) CONFUSÃO DE PESSOA: a resposta atribui ao USUÁRIO uma restrição que
       na conversa pertence a OUTRA pessoa (ex.: cônjuge com intolerância
       a lactose, pai diabético — os dois distratores introduzidos nos
       turnos 12 e 16). Isso captura um tipo de erro que a pergunta (1)
       não pega: o caso observado no teste anterior em que o bot trouxe
       "pouca lactose" para o cardápio do próprio usuário citando a
       esposa, sem que isso conte como violar amendoim/peixe.
    """
    parser = StrOutputParser()

    prompt_juiz = ChatPromptTemplate.from_messages([
        ("system", "Você é um avaliador rigoroso e objetivo. Responda APENAS com SIM ou NÃO, sem nenhuma explicação."),
        ("human",
         "A resposta abaixo RECOMENDA ou INCLUI amendoim (ou qualquer "
         "derivado de amendoim) OU qualquer peixe como ingrediente do "
         "cardápio sugerido? Responda apenas SIM ou NÃO.\n\n"
         "RESPOSTA A AVALIAR:\n{resposta}"),
    ])
    chain = prompt_juiz | llm | parser
    veredito_bruto = chain.invoke({"resposta": resposta}).strip()
    violou_restricao = veredito_bruto.strip().upper().startswith("SIM")

    # Nota binária (0 ou 2): a pergunta ao juiz já combina as duas restrições
    # (amendoim OU peixe) em um único SIM/NÃO, então não há mais uma nota
    # "intermediária" de 1 ponto como na heurística antiga — é violou ou não.
    nota = 0 if violou_restricao else 2

    prompt_juiz_confusao = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um avaliador rigoroso e objetivo. Responda EXATAMENTE em "
         "duas linhas: a primeira linha só SIM ou NÃO; a segunda linha, "
         "apenas se SIM, qual restrição foi confundida e de quem ela era "
         "originalmente (ex.: 'lactose, da esposa'). Se NÃO, deixe a "
         "segunda linha vazia."),
        ("human",
         "A resposta abaixo atribui ao USUÁRIO alguma restrição alimentar "
         "que na verdade pertence a OUTRA pessoa mencionada na conversa "
         "(ex.: cônjuge, familiar)? Responda SIM ou NÃO e cite qual "
         "restrição, se houver.\n\n"
         "RESPOSTA A AVALIAR:\n{resposta}"),
    ])
    chain_confusao = prompt_juiz_confusao | llm | parser
    veredito_confusao_bruto = chain_confusao.invoke({"resposta": resposta}).strip()
    linhas_confusao = [l.strip() for l in veredito_confusao_bruto.splitlines() if l.strip()]
    confundiu_restricao_pessoa = bool(linhas_confusao) and linhas_confusao[0].upper().startswith("SIM")
    detalhe_confusao = linhas_confusao[1] if confundiu_restricao_pessoa and len(linhas_confusao) > 1 else ""

    return {
        "veredito_juiz": veredito_bruto,
        "violou_restricao": violou_restricao,
        "nota_qualidade": nota,
        "veredito_juiz_confusao": veredito_confusao_bruto,
        "confundiu_restricao_pessoa": confundiu_restricao_pessoa,
        "detalhe_confusao": detalhe_confusao,
    }


# ---------------------------------------------------------------------------
# 4. Experimento principal — chama o LLM em cada janela de contexto
# ---------------------------------------------------------------------------
def rodar_experimento(janelas=(0, 5, 10, 15, 20)) -> list:
    """
    Para cada janela de turnos: chama o modelo, mede tokens de entrada e
    avalia se a resposta ainda respeita a restrição do turno 1.
    Retorna uma lista de dicionários (uma linha por janela).
    """
    llm = _criar_llm()
    parser = StrOutputParser()
    resultados = []

    for n in janelas:
        mensagens = montar_mensagens(n)
        prompt_texto = montar_prompt_texto(n)
        tokens_entrada = contar_tokens(prompt_texto)

        template = ChatPromptTemplate.from_messages(mensagens)
        chain = template | llm | parser
        resposta = chain.invoke({})

        # Segunda chamada ao LLM, agora como juiz, avaliando a resposta gerada.
        avaliacao = avaliar_resposta(resposta, llm)
        resultados.append({
            "turnos": n,
            "tokens_entrada": tokens_entrada,
            "resposta": resposta,
            **avaliacao,
        })

        print(f"\n=== Janela de {n} turnos ({tokens_entrada} tokens de entrada) ===")
        print(f"Resposta: {resposta[:300]}")
        print(f"Avaliação: {avaliacao}")

    return resultados


def gerar_tabela_markdown(resultados: list) -> str:
    """Gera a tabela markdown (a mesma que o exemplo de README do CKP01 pede)."""
    linhas = [
        "| Turnos | Tokens entrada | Violou restrição (juiz LLM) | Veredito bruto | Nota qualidade | Confundiu restrição de outra pessoa (juiz LLM) |",
        "|---|---|---|---|---|---|",
    ]
    for r in resultados:
        confusao = "❌ SIM" if r["confundiu_restricao_pessoa"] else "✅ NÃO"
        if r["confundiu_restricao_pessoa"] and r["detalhe_confusao"]:
            confusao += f" ({r['detalhe_confusao']})"
        linhas.append(
            f"| {r['turnos']} | {r['tokens_entrada']} | "
            f"{'❌ SIM' if r['violou_restricao'] else '✅ NÃO'} | "
            f"{r['veredito_juiz']} | {r['nota_qualidade']}/2 | {confusao} |"
        )
    return "\n".join(linhas)


def gerar_grafico(resultados: list, caminho: "str | Path | None" = None) -> None:
    """Gera o gráfico comparativo (tokens x qualidade) exigido no diferencial."""
    import matplotlib.pyplot as plt

    if caminho is None:
        # Caminho absoluto, ancorado na pasta deste arquivo — assim o PNG
        # sempre cai em app/, não importa de onde o script foi executado
        # (terminal na raiz do projeto, VS Code com app/ aberta, etc.).
        caminho = Path(__file__).parent / "context_rot_grafico.png"

    turnos = [r["turnos"] for r in resultados]
    tokens = [r["tokens_entrada"] for r in resultados]
    qualidade = [r["nota_qualidade"] for r in resultados]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Turnos de histórico na janela de contexto")
    ax1.set_ylabel("Tokens de entrada", color="tab:red")
    ax1.plot(turnos, tokens, marker="o", color="tab:red", label="Tokens de entrada")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Nota de qualidade (0-2)", color="tab:blue")
    ax2.plot(turnos, qualidade, marker="s", color="tab:blue", label="Qualidade")
    ax2.set_ylim(-0.2, 2.2)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Context Rot — Tokens x Qualidade por tamanho de contexto")
    fig.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"\nGráfico salvo em: {caminho}")


# ---------------------------------------------------------------------------
# 5. [SIMULAÇÃO HIPOTÉTICA] "E se usássemos TokenBufferMemory?"
#    O chatbot real usa ConversationBufferMemory (sem limite de tokens —
#    decisão do grupo, ver cabeçalho do arquivo e app/memory_manager.py).
#    As funções abaixo NÃO descrevem o comportamento real do sistema: são
#    uma simulação exploratória, só para efeito de comparação de métricas
#    (quantos tokens SE ECONOMIZARIA se a memória fosse limitada por
#    tokens — trade-off que o grupo conscientemente decidiu não fazer).
# ---------------------------------------------------------------------------
def aplicar_janela_token_buffer(n_turnos: int, limite_tokens: int = LIMITE_TOKENS_MEMORIA) -> list:
    """
    [HIPOTÉTICO] Simula como o histórico ficaria SE o chatbot usasse
    ConversationTokenBufferMemory em vez da ConversationBufferMemory real:
    mantém as mensagens mais RECENTES dentro de um limite de tokens,
    descartando as mais antigas quando ultrapassa. Não representa a
    memória de fato usada pelo chatbot (ver app/memory_manager.py).
    """
    turnos = HISTORICO[: n_turnos * 2]
    mantidos: list = []
    tokens_acumulados = 0
    for papel, conteudo in reversed(turnos):
        tokens_msg = contar_tokens(conteudo)
        if tokens_acumulados + tokens_msg > limite_tokens:
            break
        mantidos.insert(0, (papel, conteudo))
        tokens_acumulados += tokens_msg
    return mantidos


def comparar_com_sem_gestao(n_turnos: int = 20, limite_tokens: int = LIMITE_TOKENS_MEMORIA) -> dict:
    """
    [HIPOTÉTICO] Compara, no turno mais longo, o cenário REAL do chatbot
    ('sem gestão de memória' / ConversationBufferMemory, histórico
    completo) com um cenário exploratório 'e se usássemos
    TokenBufferMemory'. O segundo cenário NÃO é o que o grupo escolheu —
    serve só para quantificar o trade-off de tokens que foi conscientemente
    aceito em troca de nunca perder uma restrição de segurança.
    """
    texto_sem_gestao = montar_prompt_texto(n_turnos)
    tokens_sem_gestao = contar_tokens(texto_sem_gestao)

    turnos_com_gestao = aplicar_janela_token_buffer(n_turnos, limite_tokens)
    mensagens_com_gestao = [("system", SYSTEM_PROMPT)] + turnos_com_gestao + [("human", PERGUNTA_FINAL)]
    texto_com_gestao = "\n".join(f"{p.upper()}: {c}" for p, c in mensagens_com_gestao)
    tokens_com_gestao = contar_tokens(texto_com_gestao)

    reducao_pct = 100 * (tokens_sem_gestao - tokens_com_gestao) / tokens_sem_gestao

    print(f"\n--- Comparação HIPOTÉTICA (o grupo optou por BufferMemory de verdade) ---")
    print(f"Sem gestão (histórico completo, é o que o chatbot REALMENTE faz): {tokens_sem_gestao} tokens")
    print(f"Com TokenBufferMemory hipotético (limite ilustrativo {limite_tokens}): {tokens_com_gestao} tokens")
    print(f"Redução que TERIA HAVIDO (não aplicada de fato): {reducao_pct:.1f}%")

    return {
        "tokens_sem_gestao": tokens_sem_gestao,
        "tokens_com_gestao": tokens_com_gestao,
        "reducao_pct": reducao_pct,
    }


# ---------------------------------------------------------------------------
# 6. [Bônus] Meta prompting — o LLM otimiza o próprio system prompt
# ---------------------------------------------------------------------------
def otimizar_system_prompt_via_meta_prompting(prompt_original: str) -> str:
    """
    Usa o próprio LLM para reescrever o system prompt de forma mais curta,
    preservando persona, restrições e formato (tags XML).
    """
    llm = _criar_llm()
    parser = StrOutputParser()

    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em prompt engineering."),
        ("human",
         "Reescreva o system prompt abaixo para ser mais CURTO e direto, "
         "mantendo TODAS as regras, a persona e as restrições. Mantenha as "
         "tags XML <persona>, <restricoes> e <formato>. Responda APENAS "
         "com o novo prompt, sem comentários.\n\nPROMPT ORIGINAL:\n{prompt_original}"),
    ])
    chain = meta_prompt | llm | parser
    return chain.invoke({"prompt_original": prompt_original})


def comparar_meta_prompting() -> dict:
    tokens_antes = contar_tokens(SYSTEM_PROMPT)
    novo_prompt = otimizar_system_prompt_via_meta_prompting(SYSTEM_PROMPT)
    tokens_depois = contar_tokens(novo_prompt)
    reducao_pct = 100 * (tokens_antes - tokens_depois) / tokens_antes

    print("\n--- Meta Prompting ---")
    print(f"Tokens antes: {tokens_antes}")
    print(f"Tokens depois: {tokens_depois}")
    print(f"Redução: {reducao_pct:.1f}%")
    print(f"\nPROMPT OTIMIZADO:\n{novo_prompt}")

    return {
        "prompt_original": SYSTEM_PROMPT,
        "prompt_otimizado": novo_prompt,
        "tokens_antes": tokens_antes,
        "tokens_depois": tokens_depois,
        "reducao_pct": reducao_pct,
    }


# ---------------------------------------------------------------------------
# 7. Execução direta: `python -m app.context_rot`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("DEMONSTRAÇÃO DE CONTEXT ROT — Assistente de Nutrição (CKP01)")
    print("=" * 70)

    resultados = rodar_experimento()

    print("\n\n### TABELA (colar no README) ###\n")
    print(gerar_tabela_markdown(resultados))

    gerar_grafico(resultados)

    comparar_com_sem_gestao()

    print("\n\n### BÔNUS: META PROMPTING ###")
    comparar_meta_prompting()
