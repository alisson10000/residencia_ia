import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


# ==========================================
# CONFIGURAÇÃO
# ==========================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/embeddings"
MODELO = "openai/text-embedding-3-small"

PASTA_MARKDOWN = Path("markdown")
ARQUIVO_SAIDA = "embeddings.json"


# ==========================================
# GERAR EMBEDDING
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
# LER MARKDOWN
# ==========================================

trechos = []

for arquivo in PASTA_MARKDOWN.glob("*.md"):

    texto = arquivo.read_text(
        encoding="utf-8"
    )

    linhas = texto.splitlines()

    for linha in linhas:

        linha = linha.strip()

        if linha:

            trechos.append({
                "arquivo": arquivo.name,
                "texto": linha
            })


print(
    f"{len(trechos)} trechos encontrados."
)


# ==========================================
# GERAR EMBEDDINGS
# ==========================================

resultado = []

for i, trecho in enumerate(trechos):

    print(
        f"Gerando {i + 1}/{len(trechos)}"
    )

    embedding = gerar_embedding(
        trecho["texto"]
    )

    resultado.append({
        "arquivo": trecho["arquivo"],
        "texto": trecho["texto"],
        "embedding": embedding
    })


# ==========================================
# SALVAR EM JSON
# ==========================================

with open(
    ARQUIVO_SAIDA,
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        resultado,
        arquivo,
        ensure_ascii=False
    )


print("\nEmbeddings salvos em:")
print(ARQUIVO_SAIDA)