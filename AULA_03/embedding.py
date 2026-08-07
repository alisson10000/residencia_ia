import os
import requests
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Busca a chave no .env
API_KEY = os.getenv("OPENROUTER_API_KEY")


def gerar_embedding(texto):
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/text-embedding-3-small",
            "input": texto
        }
    )

    # Mostra um erro mais claro se a requisição falhar
    response.raise_for_status()

    dados = response.json()

    return dados["data"][0]["embedding"]


texto1 = "Eu gosto de estudar programação utilizando o meu pc gamer."
texto2 = "Eu gosto de aprender desenvolvimento de software com o meu pc gamer."

embedding1 = gerar_embedding(texto1)
embedding2 = gerar_embedding(texto2)

print("Texto 1:", texto1)
print("Texto 2:", texto2)

print("\nPrimeiros 10 números do embedding 1:")
print(embedding1[:10])

print("\nPrimeiros 10 números do embedding 2:")
print(embedding2[:10])

print("\nDimensão:")
print(len(embedding1))