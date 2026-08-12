import os
import json
import math
import requests
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/embeddings"

# IMPORTANTE:
# Deve ser o MESMO modelo usado para gerar os embeddings
MODELO = "openai/text-embedding-3-small"

PASTA_RESULTS = Path("results")


# ============================================================
# VERIFICAR API KEY
# ============================================================

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY não encontrada no arquivo .env"
    )


# ============================================================
# GERAR EMBEDDING DA PERGUNTA
# ============================================================

def gerar_embedding(texto):

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODELO,
            "input": texto,
        },
        timeout=120,
    )

    if not response.ok:

        print("\nErro OpenRouter:")
        print(response.text)

        raise RuntimeError(
            f"Erro HTTP {response.status_code}"
        )

    dados = response.json()

    embedding = dados["data"][0]["embedding"]

    usage = dados.get("usage", {})

    tokens = usage.get(
        "prompt_tokens",
        usage.get("total_tokens", 0)
    )

    return embedding, tokens


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

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

    if norma1 == 0 or norma2 == 0:
        return 0

    return produto / (
        norma1 * norma2
    )


# ============================================================
# LOCALIZAR DOCUMENTOS
# ============================================================

documentos = sorted([
    pasta
    for pasta in PASTA_RESULTS.iterdir()
    if pasta.is_dir()
])


if not documentos:

    print("Nenhum documento encontrado.")

    exit()


# ============================================================
# MOSTRAR DOCUMENTOS
# ============================================================

print("\n")
print("=" * 70)
print("DOCUMENTOS")
print("=" * 70)


for numero, documento in enumerate(
    documentos,
    start=1
):

    print(
        f"{numero} - {documento.name}"
    )


# ============================================================
# ESCOLHER DOCUMENTO
# ============================================================

while True:

    try:

        escolha = int(
            input(
                "\nEscolha o documento: "
            )
        )

        if 1 <= escolha <= len(documentos):
            break

        print("Opção inválida.")

    except ValueError:

        print(
            "Digite somente o número."
        )


documento_escolhido = (
    documentos[escolha - 1]
)


# ============================================================
# MOSTRAR TESTES DISPONÍVEIS
# ============================================================

print("\n")
print("=" * 70)
print("TESTES DE CHUNKING")
print("=" * 70)

print("1  - 200 caracteres")
print("2  - 500 caracteres")
print("3  - 1000 caracteres")
print("4  - 2000 caracteres")
print("5  - 500 caracteres + overlap 50")
print("6  - 500 caracteres + overlap 200")
print("7  - Parágrafos")
print("8  - 3 sentenças")
print("9  - Recursive Character")
print("10 - Markdown Headers")


# ============================================================
# ESCOLHER TESTE
# ============================================================

while True:

    try:

        teste = int(
            input(
                "\nEscolha o teste: "
            )
        )

        if 1 <= teste <= 10:
            break

        print("Escolha entre 1 e 10.")

    except ValueError:

        print(
            "Digite somente o número."
        )


# ============================================================
# LOCALIZAR JSON
# ============================================================

arquivo_json = (
    documento_escolhido
    / f"test_{teste:02d}"
    / "chunks_embeddings.json"
)


if not arquivo_json.exists():

    print("\nArquivo não encontrado:")

    print(arquivo_json)

    exit()


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

print("\nCarregando embeddings...")


with open(
    arquivo_json,
    "r",
    encoding="utf-8"
) as arquivo:

    chunks = json.load(
        arquivo
    )


print(
    f"{len(chunks)} chunks carregados."
)


# ============================================================
# PERGUNTA
# ============================================================

query = input(
    "\nDigite sua pergunta: "
).strip()


if not query:

    print(
        "A pergunta não pode estar vazia."
    )

    exit()


# ============================================================
# EMBEDDING SOMENTE DA PERGUNTA
# ============================================================

print(
    "\nGerando embedding da pergunta..."
)


embedding_query, tokens = (
    gerar_embedding(query)
)


print(
    f"Tokens da pergunta: {tokens}"
)


# ============================================================
# COMPARAR COM TODOS OS CHUNKS
# ============================================================

print(
    "\nCalculando similaridades..."
)


resultados = []


for chunk in chunks:

    embedding_chunk = (
        chunk["embedding"]
    )

    score = similaridade_cosseno(
        embedding_query,
        embedding_chunk
    )

    resultados.append({

        "chunk_id":
            chunk.get("chunk_id"),

        "texto":
            chunk.get("text", ""),

        "metadata":
            chunk.get("metadata", {}),

        "score":
            score,
    })


# ============================================================
# ORDENAR
# ============================================================

resultados.sort(
    key=lambda resultado:
        resultado["score"],
    reverse=True
)


# ============================================================
# TOP 3
# ============================================================

print("\n")
print("=" * 70)
print("TOP 3 RESULTADOS")
print("=" * 70)


for posicao, resultado in enumerate(
    resultados[:3],
    start=1
):

    print("\n")
    print(
        f"{posicao}º RESULTADO"
    )

    print("-" * 70)

    print(
        "Chunk:",
        resultado["chunk_id"]
    )

    print(
        "Similaridade:",
        round(
            resultado["score"],
            4
        )
    )

    pagina = (
        resultado["metadata"]
        .get("page")
    )

    if pagina is not None:

        print(
            "Página:",
            pagina
        )


    # Headings do teste 10
    metadata = resultado["metadata"]

    for chave in [
        "heading_1",
        "heading_2",
        "heading_3",
        "heading_4",
    ]:

        if chave in metadata:

            print(
                f"{chave}:",
                metadata[chave]
            )


    print("\nTRECHO:")

    print(
        resultado["texto"]
    )


print("\n")
print("=" * 70)
print("BUSCA FINALIZADA")
print("=" * 70)