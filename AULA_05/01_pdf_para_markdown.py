from pathlib import Path
import pymupdf4llm


# ==================================================
# DESATIVAR LAYOUT AVANÇADO
# ==================================================

pymupdf4llm.use_layout(False)


# ==================================================
# CONFIGURAÇÃO
# ==================================================

PASTA_PDF = Path("pdf")
PASTA_RESULTS = Path("results")

PASTA_RESULTS.mkdir(exist_ok=True)


# ==================================================
# LOCALIZAR PDFs
# ==================================================

pdfs = sorted(
    PASTA_PDF.glob("*.pdf")
)

print(
    f"\nPDFs encontrados: {len(pdfs)}\n"
)


# ==================================================
# PROCESSAR PDFs
# ==================================================

for pdf in pdfs:

    print("=" * 70)
    print(f"Convertendo: {pdf.name}")
    print("=" * 70)

    nome_documento = pdf.stem

    pasta_documento = (
        PASTA_RESULTS /
        nome_documento
    )

    pasta_markdown = (
        pasta_documento /
        "markdown"
    )

    pasta_imagens = (
        pasta_documento /
        "images"
    )

    pasta_markdown.mkdir(
        parents=True,
        exist_ok=True
    )

    pasta_imagens.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ==========================================
        # PDF -> MARKDOWN
        # ==========================================

        paginas = pymupdf4llm.to_markdown(
            str(pdf),

            # Separar resultado por página
            page_chunks=True,

            # Salvar imagens encontradas
            write_images=True,

            image_path=str(
                pasta_imagens
            ),

            # Não mostrar barra de progresso
            show_progress=False
        )


        # ==========================================
        # MONTAR MARKDOWN
        # ==========================================

        partes = []


        for pagina in paginas:

            metadata = pagina.get(
                "metadata",
                {}
            )

            numero_pagina = metadata.get(
                "page_number",
                "?"
            )

            texto = pagina.get(
                "text",
                ""
            )


            # Guardar informação da página
            partes.append(
                f"\n\n"
                f"<!-- page: {numero_pagina} -->"
                f"\n\n"
            )

            partes.append(
                texto
            )


        markdown_final = "".join(
            partes
        )


        # ==========================================
        # SALVAR MARKDOWN
        # ==========================================

        arquivo_saida = (
            pasta_markdown /
            f"{nome_documento}.md"
        )


        arquivo_saida.write_text(
            markdown_final,
            encoding="utf-8"
        )


        print(
            f"Markdown criado: "
            f"{arquivo_saida}"
        )

        print(
            f"Páginas processadas: "
            f"{len(paginas)}"
        )


    except Exception as erro:

        print(
            f"\nERRO REAL em "
            f"{pdf.name}:"
        )

        print(erro)


print("\n")
print("=" * 70)
print("CONVERSÃO FINALIZADA")
print("=" * 70)