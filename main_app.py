#Este modelo utiliza instructor-base para busca Instrução não semantica

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

import shutil

# 🔐 Variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 📁 Diretórios necessários
os.makedirs("docs", exist_ok=True)
os.makedirs("summaries", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("index", exist_ok=True)

# 🧠 Modelo de linguagem
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=openai_key)

# 📝 Prompt para sumarização
template_sumario = PromptTemplate(
    input_variables=["documento"],
    template="""
Você é um assistente especializado em sistemas da Senior Sistemas e em eConsignado, você receberá uma lista de arquivos com dúvidas e possíveis soluções para as mesmas.
Gere um resumo técnico, didático, mas bastante detalhado, embora e segmentado do conteúdo abaixo enviado. Este conteúdo será utilizado posteriormente para outra instância de uma IA com GPT, para responder dúvidas de usuários.
Evite explicar o motivo do resumo, foque apenas no conteúdo.
Considere que as palavras Credito do Trabalhado, eConsgnado, consignado são utilizadas para referencia ao mesmo contexto.

{documento}

Resumo:
-"""
)
summarizer = template_sumario | llm

# ✂️ Fragmentador
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# 🔎 Embedding
embedding = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

# 📄 Carregamento do cache de hash
hash_cache_path = "summaries/hash_cache.json"
hash_cache = {}
try:
    if os.path.exists(hash_cache_path):
        with open(hash_cache_path, "r", encoding="utf-8") as f:
            hash_cache = json.load(f)
except Exception as e:
    print(f"[⚠️ ERRO] Falha ao carregar cache: {e}. Iniciando cache vazio.")
    hash_cache = {}

# 🧠 Função para calcular hash de um texto
def calcular_hash(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

# 🧪 Função para validar integridade do resumo comparando o hash no cabeçalho
def validar_resumo_integridade(resumo_completo, hash_esperado):
    if not resumo_completo.startswith("<!--hash="):
        return False
    linha_hash = resumo_completo.splitlines()[0].strip()
    hash_embutido = linha_hash.replace("<!--hash=", "").replace("-->", "")
    return hash_embutido == hash_esperado

# 📄 Gerar resumo para documento

def gerar_sumario(texto: str, nome: str, hash_atual: str):
    print(f"[📜 RESUMO] Gerando resumo para: {nome}")
    resultado = summarizer.invoke({"documento": texto})
    resumo = resultado.content if hasattr(resultado, "content") else str(resultado)
    conteudo_final = f"<!--hash={hash_atual}-->\n{resumo.strip()}"
    with open(f"summaries/{nome}.resumo.txt", "w", encoding="utf-8") as f:
        f.write(conteudo_final)
    return conteudo_final

# 🔁 Processar e indexar documentos com cache

def processar_documentos():
    print("[🔁 PROCESSANDO DOCUMENTOS]")
    documentos = []
    atualizou = False

    for nome in os.listdir("docs"):
        if nome.endswith(".txt"):
            caminho = os.path.join("docs", nome)
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()
            hash_atual = calcular_hash(texto)

            resumo_path = f"summaries/{nome}.resumo.txt"

            if nome not in hash_cache or hash_cache[nome] != hash_atual:
                print(f"[🔄 ATUALIZAÇÃO] {nome} alterado ou novo. Resumindo...")
                resumo = gerar_sumario(texto, nome, hash_atual)
                hash_cache[nome] = hash_atual
                atualizou = True
            elif os.path.exists(resumo_path):
                with open(resumo_path, "r", encoding="utf-8") as f:
                    resumo = f.read()
                if not validar_resumo_integridade(resumo, hash_cache[nome]):
                    print(f"[⚠️ AVISO] Resumo de {nome} parece desatualizado. Gerando novo...")
                    resumo = gerar_sumario(texto, nome, hash_atual)
                    hash_cache[nome] = hash_atual
                    atualizou = True
                else:
                    print(f"[✅ CACHE] Resumo de {nome} reutilizado.")
            else:
                print(f"[❓ FALTANDO] Resumo não encontrado para {nome}. Gerando...")
                resumo = gerar_sumario(texto, nome, hash_atual)
                hash_cache[nome] = hash_atual
                atualizou = True

            documentos.append(resumo)

    with open(hash_cache_path, "w", encoding="utf-8") as f:
        json.dump(hash_cache, f, ensure_ascii=False, indent=2)

    if os.path.exists("index/faiss_index") and not atualizou:
        print("[📦 INDEX] Carregando índice FAISS existente...")
        return FAISS.load_local("index/faiss_index", embedding, allow_dangerous_deserialization=True)

    print("[⚙️ INDEXAÇÃO] Fragmentando e criando novo índice FAISS...")
    textos = splitter.create_documents(documentos)
    faiss_index = FAISS.from_documents(textos, embedding)
    faiss_index.save_local("index/faiss_index")
    return faiss_index

print("[🚦 STARTUP] Carregando documentos e sumários...")
retriever = processar_documentos().as_retriever()

template_qa = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um analista de suporte com conhecimento senior, especialista em RH e nos sistemas da Senior Sistemas, com domínio do tema eConsignado/Credito do trabalhador
 
Sua função é responder dúvidas de clientes com base **exclusiva** no conteúdo abaixo:
Considere que as palavras Credito do Trabalhado, eConsgnado, consignado são utilizadas para referencia ao mesmo contexto
 
{context}
 
Pergunta recebida: {question}
 
Regras:
- Responda com base no contexto acima, mesmo que a pergunta pareça ambígua.
- Se as instruções forem parciais, diga isso e oriente o cliente a revisar os parâmetros ou consultar a documentação completa.
- Nunca invente informações.
- Não forneça interpretação legal. Oriente a buscar o setor jurídico ou canais oficiais.
- Seja direto, claro e técnico.
- Responda de forma cordial, com linguagem de fácil entendimento, de forma direta, com entonação humana, sem uso de girias.
- Se você encontrar um "passo a passo" documentado, envie-o de forma integral como retorno.
- Utilize linguagem positiva e instrutiva
 


"""
)
chain = template_qa | llm

app = FastAPI(title="Agente eConsignado - RAG Hierárquico", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pergunta(BaseModel):
    pergunta: str

def log_pergunta(pergunta: str, resposta: str):
    try:
        with open("logs/perguntas_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "pergunta": pergunta,
                "resposta": resposta
            }, ensure_ascii=False) + "\n")
        print("[📝 LOG] Pergunta registrada no arquivo de log.")
    except Exception as e:
        print(f"[⚠️ ERRO LOG] Falha ao registrar log: {e}")

@app.post("/perguntar/")
def perguntar(pergunta: Pergunta):
    print(f"[🟡 RECEBIDA] Pergunta: {pergunta.pergunta}")
    inicio = time.time()

    print("[🔍 BUSCA] Recuperando trechos relevantes...")
    docs = retriever.get_relevant_documents(pergunta.pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    print("[💬 LLM] Gerando resposta com base no contexto...")
    resposta_raw = chain.invoke({"context": contexto, "question": pergunta.pergunta})
    resposta_texto = resposta_raw.content if hasattr(resposta_raw, "content") else str(resposta_raw)

    if any(sinal in resposta_texto.lower() for sinal in ["não sei", "não encontrei", "não tenho informação", "não posso", "não há informação"]):
        resposta_texto = "Não encontrei informações suficientes com segurança para responder sobre esse tema. Recomendo buscar suporte oficial."

    fim = time.time()
    print(f"[✅ RESPOSTA] Pronta em {fim - inicio:.2f}s | Preview: {resposta_texto[:100]}...")

    log_pergunta(pergunta.pergunta, resposta_texto)
    return {"resposta": resposta_texto.strip()}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return {"erro": "Apenas arquivos .txt são aceitos."}

    caminho = os.path.join("docs", file.filename)
    with open(caminho, "wb") as f:
        f.write(await file.read())

    print(f"[📅 NOVO ARQUIVO] {file.filename} recebido. Processando...")
    global retriever
    retriever = processar_documentos().as_retriever()
    return {"mensagem": f"Arquivo '{file.filename}' carregado e processado com sucesso."}

@app.post("/limpar_cache/")
def limpar_cache():
    print("[🧹 RESET] Limpando cache, resumos e índice...")

    for arquivo in os.listdir("summaries"):
        if arquivo.endswith(".resumo.txt"):
            os.remove(os.path.join("summaries", arquivo))

    if os.path.exists("summaries/hash_cache.json"):
        os.remove("summaries/hash_cache.json")

    if os.path.exists("index/faiss_index"):
        shutil.rmtree("index/faiss_index", ignore_errors=True)


    print("[✅ RESET CONCLUÍDO] Tudo pronto para reprocessamento completo.")
    return {"mensagem": "Cache, resumos e índice apagados com sucesso."}
