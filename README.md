# CP01_chatbot_nutricao

Nomes- RM:

Otavio Mancilia- 570225

Wesley Marques- 573915

Tiago Muhlmann- 569569

Gabriela Angel- 570808

Marcos Sampaio- 573987

# Justificativa da memória

O chatbot utiliza a estratégia `ConversationBufferMemory` para o
gerenciamento do histórico da conversa.

A estratégia foi escolhida porque o domínio de nutrição pode exigir
continuidade entre diferentes turnos da interação. Durante uma conversa,
o usuário pode fornecer informações como preferências alimentares,
alimentos de que não gosta e hábitos alimentares, e posteriormente
fazer perguntas que dependem dessas informações.

A `ConversationBufferMemory` mantém o histórico integral da conversa,
permitindo que a `ConversationChain` utilize informações apresentadas
em turnos anteriores.

Para o escopo deste checkpoint, no qual são esperadas conversas
relativamente curtas, essa estratégia oferece uma implementação simples
e preserva integralmente o contexto.

Como desvantagem, o histórico armazenado cresce a cada novo turno.
Consequentemente, aumenta também a quantidade de tokens enviados ao
modelo durante conversas mais longas.

Em uma aplicação com sessões extensas, estratégias como
`ConversationSummaryMemory` ou memória limitada por tokens poderiam
ser consideradas para reduzir o tamanho do contexto.

## Demonstração da memória

A memória foi testada em uma conversa com mais de cinco turnos.

Durante o teste, o usuário informou seu nome, preferências alimentares,
uma fruta favorita e hábitos relacionados à alimentação. Após cinco
turnos, o chatbot foi questionado sobre informações fornecidas no
início da conversa e conseguiu recuperá-las corretamente.

Exemplo:

Usuário: Minha fruta favorita é banana.

[Outros turnos da conversa]

Usuário: Qual fruta eu disse que era a minha favorita?

Chatbot: Você disse que sua fruta favorita é banana.

Esse comportamento demonstra que o histórico da conversa está sendo
mantido e utilizado pela `ConversationChain`.
