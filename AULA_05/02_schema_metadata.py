import json
from pathlib import Path

from langchain_core.documents import Document


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Escolha um JSON real que já foi gerado na Aula 04
ARQUIVO_JSON = Path(
    "results/"
    "attention_is_all_you_need/"
    "test_09/"
    "chunks_embeddings.json"
)


# ============================================================
# VERIFICAR SE O ARQUIVO EXISTE
# ============================================================

if not ARQUIVO_JSON.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_JSON}"
    )


# ============================================================
# CARREGAR OS CHUNKS
# ============================================================

with open(
    ARQUIVO_JSON,
    "r",
    encoding="utf-8"
) as arquivo:

    chunks = json.load(arquivo)


print("\n")
print("=" * 70)
print("ARQUIVO CARREGADO")
print("=" * 70)

print(
    f"Quantidade de chunks encontrados: {len(chunks)}"
)


# ============================================================
# PEGAR UM CHUNK REAL
# ============================================================

chunk = chunks[0]


print("\n")
print("=" * 70)
print("CHUNK ORIGINAL DA AULA 04")
print("=" * 70)

print(
    json.dumps(
        chunk,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# PEGAR O CHUNK_INDEX
# ============================================================

# Exemplo:
#
# doc01_test09_chunk0042
#
# vira:
#
# 42

try:

    chunk_index = int(
        chunk["chunk_id"]
        .split("chunk")[-1]
    )

except Exception:

    chunk_index = 0


# ============================================================
# PEGAR METADATA ANTIGO
# ============================================================

metadata_antigo = chunk.get(
    "metadata",
    {}
)


# ============================================================
# IDENTIFICAR SEÇÃO
# ============================================================

secao = (
    metadata_antigo.get("heading_4")
    or metadata_antigo.get("heading_3")
    or metadata_antigo.get("heading_2")
    or metadata_antigo.get("heading_1")
)


# ============================================================
# DESCOBRIR O NOME DO .MD
# ============================================================

nome_pdf = chunk.get(
    "document_name",
    "documento.pdf"
)

fonte_md = nome_pdf.replace(
    ".pdf",
    ".md"
)


# ============================================================
# DEFINIR IDIOMA
# ============================================================

# Como este arquivo é o Attention Is All You Need,
# o idioma é inglês.
#
# Em uma etapa futura isso pode ser automatizado.

idioma = "en"


# ============================================================
# CRIAR O NOVO SCHEMA DE METADADOS
# ============================================================

metadata = {

    # --------------------------------------------------------
    # CAMPOS OBRIGATÓRIOS
    # --------------------------------------------------------

    "fonte":
        fonte_md,

    "documento_id":
        chunk.get("document_id"),

    "chunk_index":
        chunk_index,

    "estrategia":
        chunk.get("strategy"),

    "chunk_size":
        chunk.get("chunk_size"),

    "chunk_overlap":
        chunk.get("chunk_overlap"),

    "n_caracteres":
        len(
            chunk.get("text", "")
        ),


    # --------------------------------------------------------
    # CAMPOS PRÓPRIOS
    # --------------------------------------------------------

    # Campo próprio 1
    "pagina":
        metadata_antigo.get("page"),

    # Campo próprio 2
    "secao":
        secao,

    # Campo próprio 3
    "idioma":
        idioma,


    # --------------------------------------------------------
    # CAMPO EXTRA
    # --------------------------------------------------------

    "chunk_id":
        chunk.get("chunk_id"),
}


# ============================================================
# MOSTRAR O NOVO SCHEMA
# ============================================================

print("\n")
print("=" * 70)
print("NOVO SCHEMA DE METADADOS")
print("=" * 70)

print(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2
    )
)


# ============================================================
# CRIAR DOCUMENT DO LANGCHAIN
# ============================================================

documento = Document(

    page_content=
        chunk.get(
            "text",
            ""
        ),

    metadata=
        metadata
)


# ============================================================
# MOSTRAR DOCUMENT
# ============================================================

print("\n")
print("=" * 70)
print("DOCUMENT LANGCHAIN")
print("=" * 70)

print("\nPAGE CONTENT:\n")

print(
    documento.page_content
)

print("\nMETADATA:\n")

print(
    documento.metadata
)


# ============================================================
# MOSTRAR QUE EMBEDDING NÃO ESTÁ NO DOCUMENT
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANTE")
print("=" * 70)

print(
    "O Document possui:"
)

print(
    "- page_content"
)

print(
    "- metadata"
)

print(
    "\nO embedding NÃO fica dentro do Document."
)


# ============================================================
# RESPOSTAS DA ATIVIDADE
# ============================================================

print("\n")
print("=" * 70)
print("RESPOSTAS")
print("=" * 70)


print("\n1. Campos próprios adicionados:")

print(
    "- pagina"
)

print(
    "- secao"
)

print(
    "- idioma"
)


print("\n2. Justificativas:")

print(
    "pagina -> permite saber em qual página "
    "do documento original a informação foi encontrada."
)

print(
    "secao -> permite saber em qual seção ou capítulo "
    "o trecho estava localizado."
)

print(
    "idioma -> permite identificar o idioma do conteúdo "
    "e aplicar filtros posteriormente."
)


print("\n3. Para citar a fonte no RAG:")

print(
    "Usaria principalmente os campos fonte, pagina e secao."
)


print("\n4. Utilidade do chunk_index:")

print(
    "O chunk_index permite localizar a posição do trecho "
    "dentro do documento."
)

print(
    "Se a explicação estiver cortada, podemos recuperar "
    "o chunk anterior e o próximo para reconstruir o contexto."
)


# ============================================================
# EXEMPLO JSON FINAL
# ============================================================

print("\n")
print("=" * 70)
print("EXEMPLO JSON FINAL")
print("=" * 70)

exemplo_json = {

    "page_content":
        documento.page_content,

    "metadata":
        documento.metadata
}


print(
    json.dumps(
        exemplo_json,
        ensure_ascii=False,
        indent=2
    )
)