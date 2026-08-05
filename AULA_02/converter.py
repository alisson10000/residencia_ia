import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib import error, request

os.environ.setdefault("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "false")


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "markdown"
METADATA_DIR = BASE_DIR / "metadata"

OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

ARQUIVOS_PDF = [
    "bioetica_e_ia.pdf",
    "escrita_academica_ia.pdf",
    "twitter_algoritmo.pdf",
]

METADATA_SCHEMA = {
    "name": "documento_metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "titulo": {
                "type": "string",
                "description": "Titulo principal do trabalho.",
            },
            "autores": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Lista de autores do trabalho.",
            },
            "ano": {
                "type": ["integer", "null"],
                "description": "Ano de publicacao do trabalho.",
            },
        },
        "required": ["titulo", "autores", "ano"],
        "additionalProperties": False,
    },
}


def carregar_arquivo_env(caminho_env: Path) -> None:
    if not caminho_env.exists():
        return

    for linha in caminho_env.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()

        if not linha or linha.startswith("#") or "=" not in linha:
            continue

        chave, valor = linha.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")

        if chave:
            os.environ.setdefault(chave, valor)


def normalizar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def construir_contexto_metadata(conteudo_markdown: str) -> str:
    linhas = conteudo_markdown.splitlines()
    cabecalho = "\n".join(linhas[:80])
    rodape = "\n".join(linhas[-40:])

    linhas_relevantes: list[str] = []
    padrao = re.compile(
        r"doi|recebido|revisado|aprovado|aceito|publicado|published|journal|revista",
        flags=re.IGNORECASE,
    )

    for linha in linhas:
        if padrao.search(linha):
            linhas_relevantes.append(linha)

    linhas_relevantes = linhas_relevantes[:30]

    partes = [
        "=== CABECALHO DO DOCUMENTO ===",
        cabecalho,
        "=== LINHAS BIBLIOGRAFICAS E EDITORIAIS ===",
        "\n".join(linhas_relevantes),
        "=== RODAPE DO DOCUMENTO ===",
        rodape,
    ]
    return "\n".join(partes)


def linha_eh_secao(linha: str) -> bool:
    secoes = {
        "resumo",
        "abstract",
        "resumen",
        "introducao",
        "introducao",
        "metodo",
        "metodos",
        "palavras-chave",
        "keywords",
        "doi",
    }
    base = (
        linha.lower()
        .replace("##", "")
        .replace(":", "")
        .replace(".", "")
        .strip()
    )
    return base in secoes


def listar_arquivos_markdown() -> list[Path]:
    return sorted(OUTPUT_DIR.glob("*.md"))


def converter_pdfs_para_markdown() -> list[Path]:
    try:
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError:
        print(
            "Aviso: a biblioteca 'docling' nao esta instalada no Python em uso.\n"
            f"Interpretador atual: {sys.executable}\n"
            "A conversao dos PDFs sera ignorada, mas a extracao dos JSONs a partir dos .md continuara."
        )
        return listar_arquivos_markdown()

    converter = DocumentConverter()
    arquivos_convertidos: list[Path] = []

    for nome_arquivo in ARQUIVOS_PDF:
        caminho_pdf = BASE_DIR / nome_arquivo

        if not caminho_pdf.exists():
            print(f"Arquivo nao encontrado: {caminho_pdf}")
            continue

        print(f"\nProcessando PDF: {nome_arquivo}")

        try:
            resultado = converter.convert(caminho_pdf)
            documento = resultado.document
            markdown = documento.export_to_markdown()
            caminho_md = OUTPUT_DIR / f"{caminho_pdf.stem}.md"
            caminho_md.write_text(markdown, encoding="utf-8")
            arquivos_convertidos.append(caminho_md)
            print(f"Convertido: {caminho_md.name}")
        except Exception as erro:
            print(f"Erro ao converter {nome_arquivo}: {erro}")

    return arquivos_convertidos


def detectar_configuracao_llm() -> dict[str, str] | None:
    api_key = (
        os.getenv("METADATA_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    base_url = os.getenv("METADATA_BASE_URL")
    model = os.getenv("METADATA_MODEL") or os.getenv("OPENAI_MODEL")

    if os.getenv("OPENROUTER_API_KEY") and not base_url:
        base_url = "https://openrouter.ai/api/v1"
    elif os.getenv("GROQ_API_KEY") and not base_url:
        base_url = "https://api.groq.com/openai/v1"
    elif os.getenv("OPENAI_API_KEY") and not base_url:
        base_url = "https://api.openai.com/v1"

    if not api_key or not base_url or not model:
        return None

    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
    }


def chamar_structured_outputs(
    conteudo_markdown: str,
    nome_arquivo: str,
    config: dict[str, str],
) -> dict[str, Any]:
    contexto_metadata = construir_contexto_metadata(conteudo_markdown)

    prompt_sistema = (
        "Voce extrai metadados bibliograficos de artigos em Markdown. "
        "Retorne apenas os campos do schema. "
        "Use o titulo principal do artigo, a lista de autores e o ano de publicacao. "
        "Use apenas evidencias do cabecalho, DOI, notas editoriais e rodape. "
        "Nunca use anos de referencias bibliograficas ou citacoes do corpo do texto como ano do artigo. "
        "Se o ano nao puder ser determinado com seguranca, retorne null."
    )

    prompt_usuario = (
        f"Arquivo: {nome_arquivo}\n"
        "Extraia os metadados do documento abaixo.\n"
        "Priorize o ano do DOI, linha de publicacao ou datas editoriais do proprio artigo.\n\n"
        f"{contexto_metadata}"
    )

    payload = {
        "model": config["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": METADATA_SCHEMA,
        },
    }

    corpo = json.dumps(payload).encode("utf-8")
    requisicao = request.Request(
        url=f"{config['base_url']}/chat/completions",
        data=corpo,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(requisicao, timeout=120) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
    except error.HTTPError as erro_http:
        detalhe = erro_http.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Falha HTTP na extracao estruturada: {detalhe}") from erro_http
    except error.URLError as erro_url:
        raise RuntimeError(f"Falha de rede na extracao estruturada: {erro_url}") from erro_url

    mensagem = dados["choices"][0]["message"]
    conteudo = mensagem.get("content", "")

    if isinstance(conteudo, list):
        texto_json = "".join(
            parte.get("text", "")
            for parte in conteudo
            if isinstance(parte, dict)
        )
    else:
        texto_json = conteudo

    if not texto_json:
        raise RuntimeError("A resposta do modelo veio sem conteudo JSON.")

    metadata = json.loads(texto_json)
    return validar_metadata(metadata)


def extrair_titulo_heuristico(linhas: list[str]) -> str:
    candidatos: list[str] = []

    for linha in linhas:
        linha_limpa = normalizar_texto(linha.replace("*", ""))
        if not linha_limpa:
            continue
        if linha_limpa.startswith("<!--"):
            continue
        if re.fullmatch(r"\d+", linha_limpa):
            continue
        if "doi" in linha_limpa.lower():
            continue
        if linha_eh_secao(linha_limpa):
            continue
        if linha_limpa.startswith("## "):
            conteudo = normalizar_texto(linha_limpa[3:])
            if not linha_eh_secao(conteudo):
                candidatos.append(conteudo)

    if candidatos:
        return candidatos[0]

    for linha in linhas:
        linha_limpa = normalizar_texto(linha)
        if linha_limpa:
            return linha_limpa

    return ""


def limpar_linha_autores(linha: str) -> str:
    linha = linha.replace("*", " ")
    linha = re.sub(r"https?://\S+", " ", linha)
    linha = re.sub(r"\bORCID\b.*", " ", linha, flags=re.IGNORECASE)
    linha = re.sub(r"\biD\b", " ", linha)
    linha = re.sub(r"\s+\d+\s*", " ", linha)
    linha = re.sub(r"[;|]+", ",", linha)
    return normalizar_texto(linha)


def separar_autores(bloco: str) -> list[str]:
    bloco = bloco.replace(" e ", ", ")
    partes = [normalizar_texto(parte) for parte in bloco.split(",")]
    autores: list[str] = []

    for parte in partes:
        if not parte:
            continue
        if len(parte.split()) < 2:
            continue
        if any(ch.isdigit() for ch in parte):
            continue
        autores.append(parte)

    return autores


def extrair_autores_heuristico(linhas: list[str], titulo: str) -> list[str]:
    for indice, linha in enumerate(linhas):
        linha_limpa = normalizar_texto(linha)
        if titulo and titulo in linha_limpa:
            for proxima in linhas[indice + 1 : indice + 8]:
                candidata = limpar_linha_autores(proxima)
                if not candidata:
                    continue
                if candidata.lower().startswith(("resumo", "abstract", "resumen")):
                    break
                if "universidade" in candidata.lower() or "faculdade" in candidata.lower():
                    break
                autores = separar_autores(candidata)
                if autores:
                    return autores

    for linha in linhas[:20]:
        autores = separar_autores(limpar_linha_autores(linha))
        if autores:
            return autores

    return []


def extrair_ano_heuristico(conteudo: str) -> int | None:
    contexto = construir_contexto_metadata(conteudo)

    padrao_doi_ano = re.search(
        r"doi[:\s]+[^\n]*?(19\d{2}|20\d{2})",
        contexto,
        flags=re.IGNORECASE,
    )
    if padrao_doi_ano:
        return int(padrao_doi_ano.group(1))

    padrao_publicacao = re.search(
        r"(publicado em|publicada em|published in|published on)[^\d]*(19\d{2}|20\d{2})",
        contexto,
        flags=re.IGNORECASE,
    )
    if padrao_publicacao:
        return int(padrao_publicacao.group(2))

    editorial_matches = re.findall(
        r"(recebido|revisado|aprovado|aceito|published|publicado)[^\d]*(19\d{2}|20\d{2})",
        contexto,
        flags=re.IGNORECASE,
    )
    anos_editoriais = [int(ano) for _, ano in editorial_matches]
    if anos_editoriais and len(set(anos_editoriais)) == 1:
        return anos_editoriais[0]

    return None


def extrair_metadata_heuristica(conteudo_markdown: str) -> dict[str, Any]:
    linhas = conteudo_markdown.splitlines()
    titulo = extrair_titulo_heuristico(linhas)
    autores = extrair_autores_heuristico(linhas, titulo)
    ano = extrair_ano_heuristico(conteudo_markdown)
    return validar_metadata(
        {
            "titulo": titulo,
            "autores": autores,
            "ano": ano,
        }
    )


def validar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    titulo = normalizar_texto(str(metadata.get("titulo", "")))

    autores_brutos = metadata.get("autores", [])
    if not isinstance(autores_brutos, list):
        autores_brutos = []

    autores = [
        normalizar_texto(str(autor))
        for autor in autores_brutos
        if normalizar_texto(str(autor))
    ]

    ano_bruto = metadata.get("ano")
    ano: int | None

    if ano_bruto in (None, ""):
        ano = None
    else:
        try:
            ano = int(ano_bruto)
        except (TypeError, ValueError):
            ano = None

    if ano is not None and not (1900 <= ano <= 2100):
        ano = None

    return {
        "titulo": titulo,
        "autores": autores,
        "ano": ano,
    }


def extrair_metadata_markdown(caminho_md: Path) -> dict[str, Any]:
    conteudo = caminho_md.read_text(encoding="utf-8")
    config = detectar_configuracao_llm()

    if config is not None:
        try:
            metadata = chamar_structured_outputs(conteudo, caminho_md.name, config)
            print(f"Metadados extraidos via Structured Outputs: {caminho_md.name}")
            return metadata
        except Exception as erro:
            print(
                f"Falha na extracao estruturada de {caminho_md.name}. "
                f"Usando fallback heuristico. Motivo: {erro}"
            )

    print(f"Metadados extraidos via heuristica local: {caminho_md.name}")
    return extrair_metadata_heuristica(conteudo)


def salvar_metadata_json(caminho_md: Path, metadata: dict[str, Any]) -> Path:
    caminho_json = METADATA_DIR / f"{caminho_md.stem}.json"
    caminho_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return caminho_json


def processar_metadados_markdown() -> list[dict[str, Any]]:
    documentos: list[dict[str, Any]] = []

    for caminho_md in listar_arquivos_markdown():
        metadata = extrair_metadata_markdown(caminho_md)
        salvar_metadata_json(caminho_md, metadata)

        documentos.append(
            {
                "arquivo": caminho_md.name,
                **metadata,
            }
        )

    indice_path = METADATA_DIR / "documentos.json"
    indice_path.write_text(
        json.dumps(documentos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return documentos


def main() -> None:
    carregar_arquivo_env(BASE_DIR / ".env")

    print("Iniciando conversao dos PDFs para Markdown...")
    converter_pdfs_para_markdown()

    print("\nIniciando extracao de metadados dos arquivos Markdown...")
    documentos = processar_metadados_markdown()

    print("\nMetadados extraidos:")
    for documento in documentos:
        print(
            f"- {documento['arquivo']}: "
            f"titulo={documento['titulo']!r}, "
            f"autores={len(documento['autores'])}, "
            f"ano={documento['ano']}"
        )

    print("\nProcessamento finalizado.")


if __name__ == "__main__":
    main()
