import os
import json
import math
import requests
from dotenv import load_dotenv


# ==========================================
# CONFIGURAÇÃO
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/embeddings"
MODELO = "openai/text-embedding-3-small"

ARQUIVO_EMBEDDINGS = "embeddings.json"


# ==========================================
# GERAR EMBEDDING DA PERGUNTA
# ==========================================

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

    return dados["data"][0]["embedding"]


# ==========================================
# SIMILARIDADE DE COSSENO
# ==========================================

def similaridade_cosseno(v1, v2):

    produto = sum(
        a * b
        for a, b in zip(v1, v2)
    )

    norma1 = math.sqrt(
        sum(a * a for a in v1)
    )

    norma2 = math.sqrt(
        sum(b * b for b in v2)
    )

    return produto / (
        norma1 * norma2
    )


# ==========================================
# CARREGAR EMBEDDINGS DO ARQUIVO
# ==========================================

with open(
    ARQUIVO_EMBEDDINGS,
    "r",
    encoding="utf-8"
) as arquivo:

    trechos = json.load(arquivo)


print(
    f"{len(trechos)} embeddings carregados."
)


# ==========================================
# PERGUNTA
# ==========================================

query = input(
    "\nDigite sua pergunta: "
)


# Só a pergunta precisa gerar embedding agora
embedding_query = gerar_embedding(query)


# ==========================================
# COMPARAR
# ==========================================

resultados = []

for trecho in trechos:

    score = similaridade_cosseno(
        embedding_query,
        trecho["embedding"]
    )

    resultados.append({
        "arquivo": trecho["arquivo"],
        "texto": trecho["texto"],
        "score": score
    })


# Ordenar do maior para o menor
resultados.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ==========================================
# TOP 3
# ==========================================

print("\n")
print("=" * 70)
print("TOP 3 RESULTADOS")
print("=" * 70)


for i, resultado in enumerate(
    resultados[:3],
    start=1
):

    print(
        f"\n{i}º resultado"
    )

    print(
        "Arquivo:",
        resultado["arquivo"]
    )

    print(
        "Similaridade:",
        round(
            resultado["score"],
            4
        )
    )

    print(
        "Trecho:",
        resultado["texto"]
    )

    print("-" * 70)