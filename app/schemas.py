from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class PlanoRefeicao(BaseModel):
	nome: str =  Field(description="Nome do prato ou refeição sugerida")
	calorias: int = Field(ge=0, description="Estimativa de calorias totais da refeição")
	ingredientes: List[str] = Field(description="Lista de ingredientes necessários, um por item")
	tempo_preparo: int = Field(ge=0, description="Tempo estimado de preparo em minutos")
	restricao: Optional[Literal[
	    "Alergia ao ovo",
   		"Alergia a amendoim e castanhas",
	    "Alergia a frutos do mar",
    	"Alergia a peixes",
    	"Alergia a soja",
    	"Vegetariana",
    	"Vegana",
    	"Sem açúcar",
		"Sem lactose",
    	"Sem carne suína",
    	"Sem carne vermelha",
    	"Sem glúten",
    	"Baixo teor de sódio",
		"Nenhuma"
	]] = Field(
        None, description="Restrição alimentar atendida pela refeição, se houver"
    )

parser = PydanticOutputParser(pydantic_object=PlanoRefeicao)

prompt = ChatPromptTemplate.from_messages([
	("system", "Você é um assistente de nutrição. Gere uma sugestão de refeição.\n{format_instructions}"),
	("human", "{pedido}"),
]).partial(format_instructions=parser.get_format_instructions())

# LEMBRAR DE DEFINIR NO CHAIN.PY
# llm = ChatOllama(model="gemma4:cloud", format="json")

chain_extractor = prompt | llm | parser

# Testes

print("--- Teste 1: pedido específico e bem estruturado ---")
try:
    resultado = chain_extractor.invoke({"pedido": "quero uma receita vegetariana rápida"})
    print(resultado.model_dump())
except ValidationError as e:
    print(f"Erro de schema: {e.errors()}")

print("\n--- Teste 2: pedido vago ---")
try:
	resultado = chain_extractor.invoke({"pedido": "não sei bem, algo saudável"})
	print(resultado.model_dump())
except ValidationError as e:
	print(f"Erro de schema: {e.errors()}")

print("\n--- Teste 3: restrição fora do Literal aceito ---")
try:
	resultado = chain_extractor.invoke({"pedido": "quero uma receita, mas tenho restrição a chocolate"})
	print(resultado.model_dump())
except ValidationError as e:
	print(f"Erro de schema: {e.errors()}")

print("\n--- Teste 4: validação manual forçando erro (sem depender do LLM) ---")
try:
    plano_invalido = PlanoRefeicao(
        nome="Salada de frango",
        calorias=-200,
        ingredientes=["frango", "alface"],
        tempo_preparo=15,
        restricao="Sem chocolate",
    )
except ValidationError as e:
    print(f"Erros encontrados: {e.error_count()}")
    for erro in e.errors():
        print(f"  Campo: {erro['loc']} | Erro: {erro['msg']}")