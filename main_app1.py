#Este modelo utiliza paraphrase-multilingual-mpnet-base-v2 para busca semantica

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

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
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=openai_key)

# 📝 Prompt para sumarização
template_sumario = PromptTemplate(
    input_variables=["documento"],
    template="""
Você é um assistente especializado em sistemas da Senior Sistemas, seu objetivo é resumir os documentos abaixo com precisão, para que sejam utilizados em um vetor Faiss, que será re-consumido por uma IA de resposta de tickets.
Importante, sempre que você encontrar links, mantenha-os, pois são muito importantes para o usuário final.
Ademais, gere um resumo executivo, didático e segmentado do conteúdo abaixo:

{documento}

Resumo:
-"""
)
summarizer = template_sumario | llm

# ✂️ Fragmentador
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

# 🔎 Embedding multilíngue usando modelo robusto para PT-BR
print("[🧠 EMBEDDING] Carregando modelo 'paraphrase-multilingual-mpnet-base-v2'...")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# 📄 Cache de hash
hash_cache_path = "summaries/hash_cache.json"
hash_cache = {}
try:
    if os.path.exists(hash_cache_path):
        with open(hash_cache_path, "r", encoding="utf-8") as f:
            hash_cache = json.load(f)
except Exception as e:
    print(f"[⚠️ ERRO] Falha ao carregar cache: {e}. Iniciando cache vazio.")

# 🔍 Hash de conteúdo
def calcular_hash(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

def validar_resumo_integridade(resumo_completo, hash_esperado):
    if not resumo_completo.startswith("<!--hash="):
        return False
    linha_hash = resumo_completo.splitlines()[0].strip()
    hash_embutido = linha_hash.replace("<!--hash=", "").replace("-->", "")
    return hash_embutido == hash_esperado

# 🦾 Gerar resumo
def gerar_sumario(texto, nome, hash_atual):
    print(f"[📜 RESUMO] Gerando resumo para: {nome}")
    resultado = summarizer.invoke({"documento": texto})
    resumo = resultado.content if hasattr(resultado, "content") else str(resultado)
    conteudo_final = f"<!--hash={hash_atual}-->\n{resumo.strip()}"
    with open(f"summaries/{nome}.resumo.txt", "w", encoding="utf-8") as f:
        f.write(conteudo_final)
    return conteudo_final

# 🔁 Processar documentos
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
                    print(f"[⚠️ AVISO] Resumo de {nome} parece desatualizado. Regenerando...")
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

# 🚦 Inicialização
print("[🚦 STARTUP] Carregando documentos e sumários...")
retriever = processar_documentos().as_retriever()

# 🧠 Prompt para responder perguntas
template_qa = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um analista técnico da Senior Sistemas com expertise no módulo eConsignado. Com base nos informações a seguir, responda a questão com precisão e clareza:

{context}

Questão: {question}

Regras:
- A pergutna normalmente apresenta um cenário, e um dos contextos será a solução. Você só pode unir dois contextos se eles tiverem com certeza o mesmo cenário;
- Seja direto e didático;
- Você é o suporte oficial da empresa, portanto NUNCA deve sugerir que o cliente faça contato com outro suporte;
- Você não deve responder perguntas que não façam parte do contexto acima, responda de forma educada que o conteúdo não é relacionado ao tema tratado;
- Se houver mais de uma possível solução, responda somente a que se aplica diretamente ao cenário apresentado na pergunta.
"""
)
chain = template_qa | llm

# 🚀 FastAPI
app = FastAPI(title="Agente eConsignado - RAG Multilingual MPNet", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pergunta(BaseModel):
    pergunta: str
    versao: Optional[str] = "(Versão não informada)"


# 📚 Log de perguntas
def log_pergunta(pergunta, resposta):
    try:
        with open("logs/perguntas_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "pergunta": pergunta,
                "resposta": resposta
            }, ensure_ascii=False) + "\n")
        print("[📝 LOG] Pergunta registrada.")
    except Exception as e:
        print(f"[⚠️ ERRO LOG] Falha ao registrar log: {e}")

def avaliar_confiabilidade_resposta(resposta: str) -> bool:
    """
    Avalia se a resposta gerada tem nível adequado de confiabilidade
    com base em sinais de incerteza, presença de palavras-chave do domínio
    e tamanho mínimo de conteúdo.

    Retorna True se for considerada confiável, False caso contrário.
    """

    resposta_lower = resposta.lower()

    sinais_baixa_confianca = [
        "não sei", "não encontrei", "não tenho informação", "sem informações suficientes",
        "consulte o suporte", "entre em contato com o suporte", "não há informação suficiente",
        "não há detalhes", "não está claro", "não posso afirmar", "não posso garantir",
        "não está disponível", "não tenho detalhes", "precisa fornecer mais detalhes",
        "documentação oficial", "contate o suporte oficial", "recomendo que entre em contato"
    ]

    palavras_chave_dominio = [
        "econsignado", "folha de pagamento", "empréstimo", "desconto em folha",
        "colaborador", "provisão", "consignado", "convênio"
    ]

    pontuacao_confianca = sum(1 for sinal in sinais_baixa_confianca if sinal in resposta_lower)
    tem_palavras_chave = any(p in resposta_lower for p in palavras_chave_dominio)
    tamanho_resposta = len(resposta.split())

    responder = pontuacao_confianca < 2 and tamanho_resposta >= 30 and tem_palavras_chave

    # 📋 Logging detalhado
    print("[🧪 AVALIAÇÃO DE CONFIABILIDADE]")
    print(f"  • Sinais de baixa confiança: {pontuacao_confianca}")
    print(f"  • Contém palavras-chave de domínio: {tem_palavras_chave}")
    print(f"  • Tamanho da resposta: {tamanho_resposta} palavras")
    print(f"  • Responder: {responder}")

    return responder

def pergunta_e_relevante(pergunta: str) -> bool:
    """
    Verifica se a pergunta menciona termos relevantes ao contexto do eConsignado.
    Se não contiver palavras-chave do domínio, será considerada irrelevante.
    """
    termos_chave = [
        "econsignado", "consignado", "empréstimo", "folha de pagamento",
        "provisão", "desconto em folha", "convênio", "margem consignável"
    ]
    pergunta_lower = pergunta.lower()
    return any(termo in pergunta_lower for termo in termos_chave)

def filtrar_docs_por_palavras_chave(docs, pergunta):
    """
    Filtra os documentos mantendo apenas aqueles que contêm pelo menos uma
    palavra significativa (excluindo stopwords) presente na pergunta.
    """
    stopwords = {"como", "de", "o", "a", "e", "em", "para", "com", "que", "não", "do", "da"}
    pergunta_lower = pergunta.lower()
    palavras_chave = [p for p in pergunta_lower.split() if p not in stopwords]

    docs_filtrados = [
        doc for doc in docs
        if any(p in doc.page_content.lower() for p in palavras_chave)
    ]
    return docs_filtrados



def resposta_para_pergunta(pergunta: str, versao: str = "Versão não informada") -> dict:
    if not pergunta_e_relevante(pergunta):
        print("[⚠️ RELEVÂNCIA] Pergunta fora do escopo do eConsignado.")
        return {
            "resposta": "A pergunta enviada não parece estar relacionada ao módulo eConsignado. Reformule ou envie uma dúvida mais específica.",
            "responder": False
        }

    if versao != "Versão não informada":
        pergunta = f"{pergunta} (considerar para esta pergunta a versão: {versao})"

    print("[🔍 BUSCA] Selecionando trechos relevantes...")
    docs = retriever.invoke(pergunta)
    docs_filtrados = filtrar_docs_por_palavras_chave(docs, pergunta)

    contexto = "\n\n".join([doc.page_content for doc in docs_filtrados])
    contexto_com_versao = f"[Versão informada na pergunta: {versao}]\n\n{contexto}"

    print("[💬 LLM] Gerando resposta...")
    resposta_raw = chain.invoke({"context": contexto_com_versao, "question": pergunta})
    resposta_texto = resposta_raw.content if hasattr(resposta_raw, "content") else str(resposta_raw)

    def validar_resposta(pergunta: str, resposta: str) -> dict:
        """
        Valida a resposta de uma IA com base em três critérios:
        - Relevância da pergunta para o escopo do eConsignado
        - Confiabilidade da resposta (conteúdo técnico, ausência de incertezas, palavras-chave)
        - Coerência entre a intenção da pergunta e o conteúdo da resposta

        Retorna um dicionário com o status final e detalhes de cada verificação.
        """
        pergunta_lower = pergunta.lower()
        resposta_lower = resposta.lower()

        # 1. Relevância da pergunta (domínio eConsignado)
        termos_relevantes = [
            "econsignado", "consignado", "empréstimo", "folha de pagamento",
            "provisão", "desconto em folha", "convênio", "margem consignável"
        ]
        pergunta_relevante = any(termo in pergunta_lower for termo in termos_relevantes)

        # 2. Confiabilidade da resposta
        sinais_baixa_confianca = [
            "não sei", "não encontrei", "não tenho informação", "sem informações suficientes",
            "consulte o suporte", "entre em contato com o suporte", "não há informação suficiente",
            "não há detalhes", "não está claro", "não posso afirmar", "não posso garantir",
            "não está disponível", "não tenho detalhes", "precisa fornecer mais detalhes",
            "documentação oficial", "contate o suporte oficial", "recomendo que entre em contato"
        ]
        palavras_chave_resposta = [
            "econsignado", "folha de pagamento", "empréstimo", "desconto em folha",
            "colaborador", "provisão", "consignado", "convênio"
        ]
        sinais_confianca = sum(1 for sinal in sinais_baixa_confianca if sinal in resposta_lower)
        tem_palavra_chave = any(p in resposta_lower for p in palavras_chave_resposta)
        tamanho_suficiente = len(resposta.split()) >= 30
        resposta_confiavel = sinais_confianca < 2 and tem_palavra_chave and tamanho_suficiente

        # 3. Intenção da pergunta (evita responder temas de "atualização de sistema")
        termos_excluidos = ["como atualizar", "atualizar o sistema", "instalar versão", "aplicar patch"]
        cobertura_intencao = not any(t in pergunta_lower for t in termos_excluidos)

        responder = pergunta_relevante and resposta_confiavel and cobertura_intencao

        print("[✅ VALIDAÇÃO FINALIZADA]")
        print(f"  • Pergunta relevante: {pergunta_relevante}")
        print(f"  • Resposta confiável: {resposta_confiavel}")
        print(f"  • Intenção coberta: {cobertura_intencao}")
        print(f"  • Responder: {responder}")

        return {
            "responder": responder,
            "pergunta_relevante": pergunta_relevante,
            "resposta_confiavel": resposta_confiavel,
            "intencao_coberta": cobertura_intencao
        }


    validacao = validar_resposta(pergunta, resposta_texto)
    responder = validacao["responder"]

    if not responder:
        resposta_texto = (
            "A IA não encontrou instruções claras ou a pergunta parece fora do escopo do eConsignado. "
            "Recomendo que consulte a documentação oficial ou o suporte técnico para orientações específicas."
        )


    log_pergunta(pergunta, resposta_texto)
    return {
        "resposta": resposta_texto.strip(),
        "responder": responder
    }





# 🎯 Endpoint de pergunta
@app.post("/perguntar/")
def perguntar(p: Pergunta):
    print(f"[🟡 PERGUNTA] {p.pergunta} | Versão: {p.versao}")
    inicio = time.time()

    resultado = resposta_para_pergunta(p.pergunta, p.versao)

    fim = time.time()
    print(f"[✅ FINALIZADO] Tempo: {fim - inicio:.2f}s | Preview: {resultado['resposta'][:80]}...")
    return resultado



@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        return {"erro": "Apenas arquivos .txt são aceitos."}

    caminho = os.path.join("docs", file.filename)
    with open(caminho, "wb") as f:
        f.write(await file.read())

    print(f"[📄 UPLOAD] {file.filename} recebido. Reindexando...")
    global retriever
    retriever = processar_documentos().as_retriever()
    return {"mensagem": f"Arquivo '{file.filename}' carregado e processado com sucesso."}

@app.post("/limpar_cache/")
def limpar_cache():
    print("[🛉 RESET] Limpando cache, resumos, logs e índice...")

    for pasta, cond in [("summaries", ".resumo.txt"), ("logs", ".jsonl")]:
        for arq in os.listdir(pasta):
            if arq.endswith(cond):
                os.remove(os.path.join(pasta, arq))

    for path in ["summaries/hash_cache.json", "index/faiss_index"]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

    print("[✅ LIMPEZA COMPLETA] Pronto para reprocessamento.")
    processar_documentos()
    return {"mensagem": "Cache e dados removidos com sucesso."}
