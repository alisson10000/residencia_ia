import os
import math
import requests
from dotenv import load_dotenv


# ==================================================
# CONFIGURAÇÃO
# ==================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/embeddings"
MODELO = "openai/text-embedding-3-small"


# ==================================================
# GERAR EMBEDDING + PEGAR CONSUMO DE TOKENS
# ==================================================

def gerar_embedding(texto):

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODELO,
            "input": texto
        }
    )

    response.raise_for_status()

    dados = response.json()

    embedding = dados["data"][0]["embedding"]

    # Uso de tokens retornado pela API
    usage = dados.get("usage", {})

    tokens = usage.get(
        "prompt_tokens",
        usage.get("total_tokens", 0)
    )

    return embedding, tokens


# ==================================================
# SIMILARIDADE DE COSSENO
#
# Quanto MAIOR o valor,
# mais semanticamente semelhantes.
# ==================================================

def similaridade_cosseno(vetor1, vetor2):

    produto = sum(
        a * b
        for a, b in zip(vetor1, vetor2)
    )

    norma1 = math.sqrt(
        sum(a * a for a in vetor1)
    )

    norma2 = math.sqrt(
        sum(b * b for b in vetor2)
    )

    return produto / (norma1 * norma2)


# ==================================================
# DISTÂNCIA EUCLIDIANA
#
# Quanto MENOR o valor,
# mais próximos estão os vetores.
# ==================================================

def distancia_euclidiana(vetor1, vetor2):

    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(vetor1, vetor2)
        )
    )


# ==================================================
# PALAVRAS DO EXERCÍCIO
# ==================================================

palavras = [
    "gato",
    "felino",
    "cachorro",
    "carro",
    "caminhão",
    "moto",
    "banana",
    "maçã",
    "goiaba"
]


# ==================================================
# GERAR OS EMBEDDINGS
# ==================================================

embeddings = {}
tokens_por_palavra = {}

total_tokens = 0


print("\n")
print("=" * 70)
print("GERANDO EMBEDDINGS")
print("=" * 70)


for palavra in palavras:

    embedding, tokens = gerar_embedding(palavra)

    embeddings[palavra] = embedding
    tokens_por_palavra[palavra] = tokens

    total_tokens += tokens

    print(
        f"{palavra:10} -> "
        f"{len(embedding)} dimensões | "
        f"Tokens: {tokens}"
    )


# ==================================================
# CONSUMO DE TOKENS
# ==================================================

print("\n")
print("=" * 70)
print("CONSUMO DE TOKENS")
print("=" * 70)

for palavra in palavras:

    print(
        f"{palavra:10} -> "
        f"{tokens_por_palavra[palavra]} token(s)"
    )


print("-" * 70)

print(
    f"TOTAL DE TOKENS CONSUMIDOS: {total_tokens}"
)


# ==================================================
# MOSTRAR PARTE DOS EMBEDDINGS
# ==================================================

print("\n")
print("=" * 70)
print("PRIMEIROS 5 VALORES DE CADA EMBEDDING")
print("=" * 70)


for palavra in palavras:

    print(
        f"{palavra:10}: "
        f"{embeddings[palavra][:5]}"
    )


# ==================================================
# COMPARAR TODAS AS PALAVRAS
# ==================================================

resultados = []


for i in range(len(palavras)):

    for j in range(i + 1, len(palavras)):

        palavra1 = palavras[i]
        palavra2 = palavras[j]

        embedding1 = embeddings[palavra1]
        embedding2 = embeddings[palavra2]

        similaridade = similaridade_cosseno(
            embedding1,
            embedding2
        )

        distancia = distancia_euclidiana(
            embedding1,
            embedding2
        )

        resultados.append({
            "palavra1": palavra1,
            "palavra2": palavra2,
            "similaridade": similaridade,
            "distancia": distancia
        })


# ==================================================
# ORDENAR DA MAIOR PARA A MENOR SIMILARIDADE
# ==================================================

resultados.sort(
    key=lambda x: x["similaridade"],
    reverse=True
)


# ==================================================
# MOSTRAR RESULTADOS
# ==================================================

print("\n")
print("=" * 70)
print("COMPARAÇÃO ENTRE AS PALAVRAS")
print("=" * 70)


print(
    f"{'PALAVRA 1':12}"
    f"{'PALAVRA 2':12}"
    f"{'SIMILARIDADE':16}"
    f"{'DISTÂNCIA':12}"
)

print("-" * 55)


for resultado in resultados:

    print(
        f"{resultado['palavra1']:12}"
        f"{resultado['palavra2']:12}"
        f"{resultado['similaridade']:<16.4f}"
        f"{resultado['distancia']:.4f}"
    )


# ==================================================
# TOP 5 MAIS PARECIDAS
# ==================================================

print("\n")
print("=" * 70)
print("TOP 5 PALAVRAS MAIS PARECIDAS")
print("=" * 70)


for resultado in resultados[:5]:

    print(
        f"{resultado['palavra1']} <-> "
        f"{resultado['palavra2']} | "
        f"Similaridade: {resultado['similaridade']:.4f} | "
        f"Distância: {resultado['distancia']:.4f}"
    )


# ==================================================
# TOP 5 MENOS PARECIDAS
# ==================================================

print("\n")
print("=" * 70)
print("TOP 5 PALAVRAS MENOS PARECIDAS")
print("=" * 70)


for resultado in resultados[-5:]:

    print(
        f"{resultado['palavra1']} <-> "
        f"{resultado['palavra2']} | "
        f"Similaridade: {resultado['similaridade']:.4f} | "
        f"Distância: {resultado['distancia']:.4f}"
    )