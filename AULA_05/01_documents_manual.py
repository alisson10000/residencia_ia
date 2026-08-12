from langchain_core.documents import Document


# ============================================================
# EXERCICIO 1 - CRIANDO DOCUMENTS NA MAO
# ============================================================

documentos = [
    Document(
        page_content="Embeddings sao representacoes vetoriais densas de texto.",
        metadata={
            "fonte": "arquivo_01.md",
            "pagina": 1,
            "tipo": "teoria",
            "tema": "embeddings",
            "autor": "Alisson",
        },
    ),
    Document(
        page_content="Chunking divide textos grandes em partes menores.",
        metadata={
            "fonte": "arquivo_02.md",
            "pagina": 2,
            "tipo": "teoria",
            "tema": "chunking",
            "autor": "Alisson",
        },
    ),
    Document(
        page_content="RAG combina busca de contexto com modelo de linguagem.",
        metadata={
            "fonte": "arquivo_03.md",
            "pagina": 3,
            "tipo": "teoria",
            "tema": "rag",
            "autor": "Alisson",
        },
    ),
    Document(
        page_content="Tokenizacao separa o texto em unidades chamadas tokens.",
        metadata={
            "fonte": "arquivo_04.md",
            "pagina": 4,
            "tipo": "teoria",
            "tema": "tokenizacao",
            "autor": "Alisson",
        },
    ),
    Document(
        page_content="Embeddings ajudam a medir similaridade entre textos.",
        metadata={
            "fonte": "arquivo_05.md",
            "pagina": 5,
            "tipo": "pratica",
            "tema": "embeddings",
            "autor": "Alisson",
            "tags": ["vetores", "similaridade", "busca"],
            "detalhes": {"nivel": "iniciante", "aula": 5},
        },
    ),
]


# ============================================================
# 1. EXIBIR PAGE_CONTENT E METADATA DE CADA DOCUMENTO
# ============================================================

for indice, documento in enumerate(documentos, start=1):
    print("=" * 60)
    print(f"DOCUMENTO {indice}")
    print("page_content:", documento.page_content)
    print("metadata:", documento.metadata)


# ============================================================
# 2. RESULTADO DE LEN(DOCUMENTOS)
# ============================================================

print("=" * 60)
print("len(documentos) =", len(documentos))


# ============================================================
# 3. TESTES PEDIDOS NO ENUNCIADO
# ============================================================

documento_sem_metadata = Document(
    page_content="Este documento foi criado sem metadata."
)

print("=" * 60)
print("DOCUMENTO SEM METADATA")
print("page_content:", documento_sem_metadata.page_content)
print("metadata:", documento_sem_metadata.metadata)

print("=" * 60)
print("OBSERVACOES")
print(
    "O metadata aceitou tipos simples como string e int, e tambem "
    "aceitou lista e dicionario aninhado neste teste."
)
print(
    "Quando o Document e criado sem metadata, o atributo metadata "
    "fica como um dicionario vazio: {}"
)
