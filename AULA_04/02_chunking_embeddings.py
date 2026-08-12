import os
import re
import json
import statistics
from pathlib import Path

import requests
from dotenv import load_dotenv

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/embeddings"

# Mesmo modelo em todos os testes
MODELO = "openai/text-embedding-3-small"

PASTA_RESULTS = Path("results")

# Quantos chunks mandar por requisição
TAMANHO_LOTE = 20


# ============================================================
# CONFIGURAÇÃO DE EXECUÇÃO
# ============================================================

# False = processar todos os documentos
TESTAR_APENAS_UM_DOCUMENTO = False

# Executar os 10 testes
TESTES_A_EXECUTAR = list(range(1, 11))

# Se o JSON já existe, não gerar embedding novamente
PULAR_TESTES_EXISTENTES = True


# ============================================================
# VALIDAR API KEY
# ============================================================

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY não encontrada no arquivo .env"
    )


# ============================================================
# GERAR EMBEDDINGS
# ============================================================

def gerar_embeddings(textos):

    textos = [
        texto.strip()
        for texto in textos
        if texto and texto.strip()
    ]

    if not textos:
        return [], 0

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODELO,
            "input": textos,
        },
        timeout=120,
    )

    # --------------------------------------------------------
    # MOSTRAR ERRO REAL DA API
    # --------------------------------------------------------

    if not response.ok:

        print("\n")
        print("=" * 70)
        print("ERRO DA API OPENROUTER")
        print("=" * 70)

        print("Status:", response.status_code)

        print("\nResposta:")
        print(response.text)

        print("\nTamanho dos textos enviados:")

        for numero, texto in enumerate(
            textos,
            start=1
        ):
            print(
                f"{numero}: "
                f"{len(texto)} caracteres"
            )

        raise RuntimeError(
            f"OpenRouter retornou HTTP {response.status_code}"
        )

    dados = response.json()

    embeddings = [
        item["embedding"]
        for item in dados["data"]
    ]

    usage = dados.get("usage", {})

    total_tokens = usage.get(
        "prompt_tokens",
        usage.get("total_tokens", 0)
    )

    return embeddings, total_tokens


# ============================================================
# TESTES 1 A 6
# CHUNK FIXO POR CARACTERES
# ============================================================

def chunk_caracteres(
    texto,
    tamanho,
    overlap
):

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=tamanho,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks = splitter.split_text(texto)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# TESTE 7
# PARÁGRAFOS
# ============================================================

def chunk_paragrafos(texto):

    paragrafos = re.split(
        r"\n\s*\n",
        texto
    )

    return [
        paragrafo.strip()
        for paragrafo in paragrafos
        if paragrafo.strip()
    ]


# ============================================================
# TESTE 8
# 3 SENTENÇAS POR CHUNK
# ============================================================

def chunk_sentencas(texto):

    sentencas = re.split(
        r"(?<=[.!?])\s+",
        texto
    )

    sentencas = [
        sentenca.strip()
        for sentenca in sentencas
        if sentenca.strip()
    ]

    chunks = []

    for i in range(
        0,
        len(sentencas),
        3
    ):

        grupo = sentencas[
            i:i + 3
        ]

        chunk = " ".join(grupo)

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# TESTE 9
# RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================

def chunk_recursive(texto):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,

        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],

        length_function=len,
    )

    chunks = splitter.split_text(texto)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# ============================================================
# TESTE 10
# MARKDOWN HEADINGS
# ============================================================

def chunk_markdown(texto):

    headers = [
        ("#", "heading_1"),
        ("##", "heading_2"),
        ("###", "heading_3"),
        ("####", "heading_4"),
    ]

    # Primeiro separa por headings Markdown
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    )

    documentos = markdown_splitter.split_text(
        texto
    )

    # Se uma seção ficar muito grande,
    # subdivide preservando os metadados
    splitter_secao_grande = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
        length_function=len,
    )

    resultado = []

    for documento in documentos:

        texto_secao = (
            documento.page_content.strip()
        )

        metadata = dict(
            documento.metadata
        )

        if not texto_secao:
            continue

        # Seção pequena
        if len(texto_secao) <= 1000:

            resultado.append({
                "texto": texto_secao,
                "metadata": metadata,
            })

            continue

        # Seção grande
        subchunks = (
            splitter_secao_grande
            .split_text(texto_secao)
        )

        for indice, subchunk in enumerate(
            subchunks,
            start=1
        ):

            metadata_subchunk = dict(
                metadata
            )

            metadata_subchunk["subchunk"] = indice

            resultado.append({
                "texto": subchunk,
                "metadata": metadata_subchunk,
            })

    return resultado


# ============================================================
# DESCOBRIR PÁGINA
# ============================================================

def descobrir_pagina(
    documento_completo,
    texto_chunk
):

    trecho_busca = texto_chunk[:150]

    posicao = documento_completo.find(
        trecho_busca
    )

    if posicao == -1:
        return None

    texto_anterior = (
        documento_completo[:posicao]
    )

    paginas = re.findall(
        r"<!-- page: ([0-9]+) -->",
        texto_anterior
    )

    if paginas:
        return int(paginas[-1])

    return None


# ============================================================
# CONFIGURAÇÃO DOS 10 TESTES
# ============================================================

TESTES = {

    1: {
        "strategy": "fixed_200",
        "chunk_size": 200,
        "chunk_overlap": 0,
    },

    2: {
        "strategy": "fixed_500",
        "chunk_size": 500,
        "chunk_overlap": 0,
    },

    3: {
        "strategy": "fixed_1000",
        "chunk_size": 1000,
        "chunk_overlap": 0,
    },

    4: {
        "strategy": "fixed_2000",
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },

    5: {
        "strategy": "fixed_500_overlap_50",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    6: {
        "strategy": "fixed_500_overlap_200",
        "chunk_size": 500,
        "chunk_overlap": 200,
    },

    7: {
        "strategy": "paragraph",
        "chunk_size": None,
        "chunk_overlap": 0,
    },

    8: {
        "strategy": "three_sentences",
        "chunk_size": None,
        "chunk_overlap": 0,
    },

    9: {
        "strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    10: {
        "strategy": "markdown_headers",
        "chunk_size": None,
        "chunk_overlap": 0,
    },
}


# ============================================================
# EXECUTAR CHUNKING
# ============================================================

def executar_chunking(
    test_id,
    texto
):

    config = TESTES[test_id]

    # --------------------------------------------------------
    # TESTES 1 A 6
    # --------------------------------------------------------

    if test_id <= 6:

        chunks = chunk_caracteres(
            texto,
            config["chunk_size"],
            config["chunk_overlap"],
        )

        return [
            {
                "texto": chunk,
                "metadata": {},
            }
            for chunk in chunks
        ]

    # --------------------------------------------------------
    # TESTE 7
    # --------------------------------------------------------

    if test_id == 7:

        chunks = chunk_paragrafos(
            texto
        )

        return [
            {
                "texto": chunk,
                "metadata": {},
            }
            for chunk in chunks
        ]

    # --------------------------------------------------------
    # TESTE 8
    # --------------------------------------------------------

    if test_id == 8:

        chunks = chunk_sentencas(
            texto
        )

        return [
            {
                "texto": chunk,
                "metadata": {},
            }
            for chunk in chunks
        ]

    # --------------------------------------------------------
    # TESTE 9
    # --------------------------------------------------------

    if test_id == 9:

        chunks = chunk_recursive(
            texto
        )

        return [
            {
                "texto": chunk,
                "metadata": {},
            }
            for chunk in chunks
        ]

    # --------------------------------------------------------
    # TESTE 10
    # --------------------------------------------------------

    if test_id == 10:

        return chunk_markdown(
            texto
        )

    raise ValueError(
        f"Teste inexistente: {test_id}"
    )


# ============================================================
# PROCESSAR UM TESTE
# ============================================================

def processar_teste(
    document_id,
    document_name,
    texto_documento,
    test_id
):

    config = TESTES[test_id]

    chunks = executar_chunking(
        test_id,
        texto_documento
    )

    print(
        f"Chunks encontrados: {len(chunks)}"
    )

    # Mostrar tamanhos
    if chunks:

        maior = max(
            len(chunk["texto"])
            for chunk in chunks
        )

        menor = min(
            len(chunk["texto"])
            for chunk in chunks
        )

        print(
            f"Menor chunk antes do embedding: {menor}"
        )

        print(
            f"Maior chunk antes do embedding: {maior}"
        )

    resultados = []

    total_tokens = 0

    # ========================================================
    # PROCESSAR EM LOTES
    # ========================================================

    for inicio in range(
        0,
        len(chunks),
        TAMANHO_LOTE
    ):

        lote = chunks[
            inicio:
            inicio + TAMANHO_LOTE
        ]

        textos = [
            chunk["texto"]
            for chunk in lote
        ]

        embeddings, tokens = gerar_embeddings(
            textos
        )

        total_tokens += tokens

        # ----------------------------------------------------
        # ASSOCIAR EMBEDDING
        # ----------------------------------------------------

        for posicao, (
            chunk,
            embedding
        ) in enumerate(
            zip(
                lote,
                embeddings
            ),
            start=inicio + 1
        ):

            chunk_id = (
                f"{document_id}_"
                f"test{test_id:02d}_"
                f"chunk{posicao:04d}"
            )

            metadata = dict(
                chunk["metadata"]
            )

            # Adicionar página
            metadata["page"] = descobrir_pagina(
                texto_documento,
                chunk["texto"]
            )

            resultados.append({

                "chunk_id":
                    chunk_id,

                "document_id":
                    document_id,

                "document_name":
                    document_name,

                "test_id":
                    test_id,

                "strategy":
                    config["strategy"],

                "chunk_size":
                    config["chunk_size"],

                "chunk_overlap":
                    config["chunk_overlap"],

                "text":
                    chunk["texto"],

                "embedding":
                    embedding,

                "metadata":
                    metadata,
            })

        processados = min(
            inicio + TAMANHO_LOTE,
            len(chunks)
        )

        print(
            f"Embeddings: "
            f"{processados}/"
            f"{len(chunks)}"
        )

    return (
        resultados,
        total_tokens
    )


# ============================================================
# ESTATÍSTICAS
# ============================================================

def gerar_estatisticas(
    chunks,
    total_tokens,
    config
):

    tamanhos = [
        len(chunk["text"])
        for chunk in chunks
    ]

    if chunks:

        dimensao = len(
            chunks[0]["embedding"]
        )

    else:

        dimensao = 0

    overlap = (
        config["chunk_overlap"]
        or 0
    )

    if config["chunk_size"]:

        percentual_overlap = (
            overlap
            / config["chunk_size"]
            * 100
        )

    else:

        percentual_overlap = 0

    if overlap > 0:

        overlap_chunks = max(
            len(chunks) - 1,
            0
        )

    else:

        overlap_chunks = 0

    return {

        "num_chunks":
            len(chunks),

        "avg_chunk_size":
            round(
                statistics.mean(tamanhos),
                2
            )
            if tamanhos
            else 0,

        "min_chunk_size":
            min(tamanhos)
            if tamanhos
            else 0,

        "max_chunk_size":
            max(tamanhos)
            if tamanhos
            else 0,

        "overlap_chunks":
            overlap_chunks,

        "overlap_percent":
            round(
                percentual_overlap,
                2
            ),

        "total_tokens":
            total_tokens,

        "embedding_dimension":
            dimensao,
    }


# ============================================================
# CARREGAR STATS EXISTENTE
# ============================================================

def carregar_stats_existente(
    arquivo_stats
):

    if not arquivo_stats.exists():
        return None

    try:

        with open(
            arquivo_stats,
            "r",
            encoding="utf-8"
        ) as arquivo:

            return json.load(
                arquivo
            )

    except Exception:

        return None


# ============================================================
# RECONSTRUIR STATS DE JSON JÁ EXISTENTE
# ============================================================

def reconstruir_stats_existente(
    arquivo_json,
    arquivo_stats,
    test_id,
    config
):

    try:

        with open(
            arquivo_json,
            "r",
            encoding="utf-8"
        ) as arquivo:

            chunks = json.load(
                arquivo
            )

        tamanhos = [
            len(chunk.get("text", ""))
            for chunk in chunks
        ]

        if (
            chunks
            and chunks[0].get("embedding")
        ):

            dimensao = len(
                chunks[0]["embedding"]
            )

        else:

            dimensao = 0

        overlap = (
            config["chunk_overlap"]
            or 0
        )

        if config["chunk_size"]:

            percentual_overlap = (
                overlap
                / config["chunk_size"]
                * 100
            )

        else:

            percentual_overlap = 0

        if overlap > 0:

            overlap_chunks = max(
                len(chunks) - 1,
                0
            )

        else:

            overlap_chunks = 0

        stats = {

            "test_id":
                test_id,

            "strategy":
                config["strategy"],

            "chunk_size":
                config["chunk_size"],

            "chunk_overlap":
                config["chunk_overlap"],

            "num_chunks":
                len(chunks),

            "avg_chunk_size":
                round(
                    statistics.mean(tamanhos),
                    2
                )
                if tamanhos
                else 0,

            "min_chunk_size":
                min(tamanhos)
                if tamanhos
                else 0,

            "max_chunk_size":
                max(tamanhos)
                if tamanhos
                else 0,

            "overlap_chunks":
                overlap_chunks,

            "overlap_percent":
                round(
                    percentual_overlap,
                    2
                ),

            # Não sabemos tokens antigos
            "total_tokens":
                None,

            "embedding_dimension":
                dimensao,

            "stats_reconstructed":
                True,
        }

        with open(
            arquivo_stats,
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                stats,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )

        return stats

    except Exception as erro:

        print(
            "Erro ao reconstruir stats:",
            erro
        )

        return None


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

print("\n")
print("=" * 80)
print("EXPERIMENTOS DE CHUNKING + EMBEDDINGS")
print("=" * 80)


# ============================================================
# LOCALIZAR DOCUMENTOS
# ============================================================

pastas_documentos = sorted([
    pasta
    for pasta in PASTA_RESULTS.iterdir()
    if pasta.is_dir()
])


# ============================================================
# MODO DE TESTE
# ============================================================

if TESTAR_APENAS_UM_DOCUMENTO:

    pastas_documentos = (
        pastas_documentos[:1]
    )

    print(
        "\nMODO DE TESTE ATIVADO"
    )

    if pastas_documentos:

        print(
            "Documento:",
            pastas_documentos[0].name
        )


print(
    "\nTestes que serão executados:",
    TESTES_A_EXECUTAR
)


# ============================================================
# SUMMARY
# ============================================================

summary = []


# ============================================================
# PERCORRER DOCUMENTOS
# ============================================================

for numero, pasta_documento in enumerate(
    pastas_documentos,
    start=1
):

    pasta_markdown = (
        pasta_documento
        / "markdown"
    )

    arquivos_md = sorted(
        pasta_markdown.glob("*.md")
    )

    if not arquivos_md:

        print(
            "Markdown não encontrado:",
            pasta_documento
        )

        continue

    arquivo_md = arquivos_md[0]

    texto_documento = (
        arquivo_md.read_text(
            encoding="utf-8"
        )
    )

    document_id = (
        f"doc{numero:02d}"
    )

    document_name = (
        arquivo_md.stem
        + ".pdf"
    )

    print("\n")
    print("=" * 80)
    print(
        f"DOCUMENTO: {document_name}"
    )
    print("=" * 80)


    resumo_documento = {

        "document_id":
            document_id,

        "document":
            document_name,

        "embedding_model":
            MODELO,

        "experiments":
            [],
    }


    # ========================================================
    # EXECUTAR TESTES
    # ========================================================

    for test_id in TESTES_A_EXECUTAR:

        config = TESTES[test_id]

        pasta_teste = (
            pasta_documento
            / f"test_{test_id:02d}"
        )

        pasta_teste.mkdir(
            exist_ok=True
        )

        arquivo_json = (
            pasta_teste
            / "chunks_embeddings.json"
        )

        arquivo_stats = (
            pasta_teste
            / "stats.json"
        )


        print("\n")
        print("-" * 70)

        print(
            f"TESTE {test_id:02d}"
        )

        print(
            "Estratégia:",
            config["strategy"]
        )

        print(
            "Chunk size:",
            config["chunk_size"]
        )

        print(
            "Overlap:",
            config["chunk_overlap"]
        )

        print("-" * 70)


        # ====================================================
        # PULAR TESTE EXISTENTE
        # ====================================================

        if (
            PULAR_TESTES_EXISTENTES
            and arquivo_json.exists()
        ):

            print(
                "Teste já existe. "
                "Pulando geração de embeddings."
            )

            stats = carregar_stats_existente(
                arquivo_stats
            )

            # Se stats.json não existe,
            # reconstrói sem chamar OpenRouter
            if not stats:

                print(
                    "stats.json não encontrado."
                )

                print(
                    "Reconstruindo estatísticas..."
                )

                stats = reconstruir_stats_existente(
                    arquivo_json,
                    arquivo_stats,
                    test_id,
                    config,
                )

            if stats:

                resumo_documento[
                    "experiments"
                ].append(
                    stats
                )

            continue


        # ====================================================
        # GERAR NOVO TESTE
        # ====================================================

        try:

            chunks, total_tokens = (
                processar_teste(
                    document_id,
                    document_name,
                    texto_documento,
                    test_id,
                )
            )

        except Exception as erro:

            print("\n")
            print("=" * 70)
            print("ERRO NO TESTE")
            print("=" * 70)

            print(
                "Documento:",
                document_name
            )

            print(
                "Teste:",
                test_id
            )

            print(
                "Erro:",
                erro
            )

            raise


        # ====================================================
        # SALVAR JSON
        # ====================================================

        with open(
            arquivo_json,
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                chunks,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )


        # ====================================================
        # GERAR ESTATÍSTICAS
        # ====================================================

        estatisticas = gerar_estatisticas(
            chunks,
            total_tokens,
            config,
        )

        resumo_teste = {

            "test_id":
                test_id,

            "strategy":
                config["strategy"],

            "chunk_size":
                config["chunk_size"],

            "chunk_overlap":
                config["chunk_overlap"],

            **estatisticas,
        }


        # ====================================================
        # SALVAR STATS.JSON
        # ====================================================

        with open(
            arquivo_stats,
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                resumo_teste,
                arquivo,
                ensure_ascii=False,
                indent=2,
            )


        resumo_documento[
            "experiments"
        ].append(
            resumo_teste
        )


        # ====================================================
        # MOSTRAR RESULTADO
        # ====================================================

        print("\nRESULTADO")

        print(
            "Chunks:",
            estatisticas["num_chunks"]
        )

        print(
            "Média caracteres:",
            estatisticas[
                "avg_chunk_size"
            ]
        )

        print(
            "Menor chunk:",
            estatisticas[
                "min_chunk_size"
            ]
        )

        print(
            "Maior chunk:",
            estatisticas[
                "max_chunk_size"
            ]
        )

        print(
            "Overlap:",
            f"{estatisticas['overlap_percent']}%"
        )

        print(
            "Tokens:",
            estatisticas[
                "total_tokens"
            ]
        )

        print(
            "Dimensão embedding:",
            estatisticas[
                "embedding_dimension"
            ]
        )

        print(
            "JSON:",
            arquivo_json
        )


    summary.append(
        resumo_documento
    )


# ============================================================
# SALVAR SUMMARY.JSON
# ============================================================

arquivo_summary = (
    PASTA_RESULTS
    / "summary.json"
)


with open(
    arquivo_summary,
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        summary,
        arquivo,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSAMENTO FINALIZADO")
print("=" * 80)

print(
    "\nResumo salvo em:",
    arquivo_summary
)