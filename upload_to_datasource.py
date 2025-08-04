import os
import httpx
from dotenv import load_dotenv
from pathlib import Path
from typing import List

load_dotenv()
BEARER_TOKEN = os.getenv("CHATVOLT_BEARER_TOKEN")
DATASOURCE_ID = "cmd615eoc02hpp8n4001uc2y3"
DOCS_PATH = Path("./docs")
EXTENSIONS = [".txt", ".pdf", ".docx"]

def listar_arquivos_validos(pasta: Path) -> List[Path]:
    return [f for f in pasta.glob("*") if f.suffix.lower() in EXTENSIONS]

async def enviar_arquivo(arquivo: Path):
    url = "https://api.chatvolt.ai/datasources"  # endpoint correto
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    files = {
        "file": (arquivo.name, open(arquivo, "rb")),
        "type": (None, "file"),
        "datastoreId": (None, DATASOURCE_ID),
        "fileName": (None, arquivo.name)  # opcional mas recomendável
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"[📤] Enviando: {arquivo.name}")
            resp = await client.post(url, headers=headers, files=files)
            resp.raise_for_status()
            print(f"[✅] Sucesso: {arquivo.name}")
        except httpx.HTTPStatusError as e:
            print(f"[❌] Erro {e.response.status_code} ao enviar {arquivo.name}: {e.response.text}")
        except Exception as e:
            print(f"[❌] Fail geral no envio de {arquivo.name}: {e}")

async def main():
    arquivos = listar_arquivos_validos(DOCS_PATH)
    if not arquivos:
        print("[ℹ️] Nenhum arquivo válido na pasta ./docs/")
        return
    print(f"[📁] {len(arquivos)} arquivo(s) encontrados. Iniciando uploads...")
    for a in arquivos:
        await enviar_arquivo(a)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
