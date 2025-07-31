from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 🔐 Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 📁 Pastas obrigatórias
os.makedirs("docs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 🧠 Embeddings (InstructorEmbedding - compreensão semântica superior)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

# 🔧 Utilitários
def fragmentar_documentos(pasta, chunk_size=1000, chunk_overlap=150):
    documentos_resultantes = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith(".txt"):
            caminho = os.path.join(pasta, nome_arquivo)
            loader = TextLoader(caminho, encoding="utf-8")
            documentos = loader.load()
            chunks = splitter.split_documents(documentos)

            for chunk in chunks:
                chunk.metadata["arquivo"] = nome_arquivo
                documentos_resultantes.append(chunk)

    return documentos_resultantes

def log_pergunta(pergunta: str, resposta: str):
    log_path = "logs/perguntas_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta
        }, ensure_ascii=False) + "\n")

# 📥 Carga inicial do vetor
docs = fragmentar_documentos("docs")
db = FAISS.from_documents(docs, embeddings)

# 🧠 Prompt personalizado
prompt_template = PromptTemplate(
    input_variables=["context_str", "question"],
    template="""
Você é um analista de suporte com conhecimento senior, especialista em RH e no uso dos sistema de RH da Senior sistemas e especializado no eConsignado.

Responda à pergunta com base **exclusivamente** nas informações abaixo:

{context_str}

Pergunta: {question}

Regras:
- Quando houver documentações de sistema com o passo a passo você deve levar os passos descritos de forma clara e objetiva após a introdução da resposta.
- Não invente informações.
- Se não tiver certeza, diga: "Não encontrei informações suficientes com segurança sobre esse tema."
- Seja direto, técnico e claro.
Você **não deve fornecer interpretações legais**.
"""
)

# 🔧 LLM + QA Chain com refinamento
llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai_key)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 6}),
    chain_type="refine",
    chain_type_kwargs={
        "question_prompt": prompt_template,
        "refine_prompt": prompt_template,
        "document_variable_name": "context_str"
    },
    return_source_documents=True
)

# 🔁 FastAPI
app = FastAPI(title="Agente eConsignado (Senior Sistemas)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pergunta(BaseModel):
    pergunta: str

@app.post("/perguntar/")
def perguntar(pergunta: Pergunta):
    resultado = qa.invoke(pergunta.pergunta)
    resposta = resultado["result"]
    fontes = resultado["source_documents"]

    log_pergunta(pergunta.pergunta, resposta)

    if not fontes or resposta.strip().lower() in ["", "não sei", "não encontrado"]:
        return {
            "resposta": "Não encontrei informações suficientes com segurança sobre esse tema.",
            "fonte": None
        }

    origem = [
        {
            "arquivo": doc.metadata.get("arquivo", "Desconhecido"),
            "trecho": doc.page_content.strip()[:300] + "..."
        }
        for doc in fontes
    ]

    return {
        "resposta": resposta.strip(),
        "fonte": origem
    }

# 📤 Novo endpoint: upload de .txt em tempo real
@app.post("/upload/")
async def upload_txt(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return {"erro": "Apenas arquivos .txt são suportados."}

    caminho = os.path.join("docs", file.filename)
    with open(caminho, "wb") as f:
        f.write(await file.read())

    novos_docs = fragmentar_documentos("docs")
    global db, qa
    db = FAISS.from_documents(novos_docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        chain_type="refine",
        chain_type_kwargs={
            "question_prompt": prompt_template,
            "refine_prompt": prompt_template,
            "document_variable_name": "context_str"
        },
        return_source_documents=True
    )

    return {"mensagem": f"Arquivo '{file.filename}' processado e indexado com sucesso."}
