#Este modelo utiliza text-embedding-3-large da Open IA para busca avançada, abaixo os custos:

#| Tipo                       | Preço por 1.000 tokens |
#| -------------------------- | ---------------------- |
#| **text-embedding-3-large** | **\$0.00013**          | <-- usando este
#| **text-embedding-3-small** | \$0.00002              |


#Um documento com 1.000 palavras tem aproximadamente 750 a 1.200 tokens, dependendo da linguagem e estilo.

#A cobrança é apenas na geração do embedding, não durante a busca.

#main_appOpenIA.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from config_assistentes import CATEGORIAS_CONFIG, CATEGORIA_PADRAO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tiktoken
import json
import time
import hashlib
import re
import spacy
import shutil
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


nlp = spacy.load("pt_core_news_sm")


# 🔐 Variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# 📁 Diretórios necessários
os.makedirs("docs", exist_ok=True)
os.makedirs("summaries", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("index", exist_ok=True)

# 🧠 Modelo de linguagem
llmModel = "gpt-4o"
llm = ChatOpenAI(model=llmModel, temperature=0.0, openai_api_key=openai_key)

def custo_total_acumulado(limite_usd=50.0):
    total = 0.0
    try:
        if os.path.exists("logs/custos_log.jsonl"):
            with open("logs/custos_log.jsonl", "r", encoding="utf-8") as f:
                for linha in f:
                    try:
                        dado = json.loads(linha)
                        total += dado.get("custo_usd", 0.0)
                        print("[CONSULTA CUSTOS] Custo total acumulado: $ {total}")
                        if total >= limite_usd:
                            return total  # Early exit
                    except Exception:
                        continue
    except Exception as e:
        print(f"[ERRO] Falha ao calcular custo acumulado: {e}")
        return limite_usd  # Travar em caso de erro
    return total

# 📝 Prompt para sumarização
template_sumario = PromptTemplate(
    input_variables=["documento"],
    template="""
Você é um assistente especializado em sistemas da Senior Sistemas, seu objetivo é resumir os documentos abaixo com precisão, para que sejam utilizados em um vetor Faiss utilizando chunk_size=1500 e chunk_overlap=100, que será re-consumido por uma IA de resposta de tickets;
Importante, sempre que você encontrar links, mantenha-os, pois são muito importantes para o usuário final;

Os artigos que você está resumindo estão em português do Brasil e em contextos que são muito parecidos entre si, mas diferentes, por exemplo:
“Falha ao calcular impostos de férias” é diferente de “falha ao calcular impostos na demissão”. Entretanto, “o sistema não está calculando corretamente os impostos nas férias” é igual a “Falha ao calcular impostos de férias”, portanto resuma de modo que o contexto fique claro e facilmente distinguivel no vetor FAISS para o chunk informado acima.

Os resumos serão consumidos por uma IA, portanto, remova contradicções e ambiguidades semanticas, tanto quanto possível for sem alterar o real contexto, tema e solução.
Você eventualmente encontrará informações sobre "Versão", você deve considerar o entendimento abaixo como guia. Entenda que o comportamento muda em novas versões e builds, portanto, é crucial manter essa informação em seu resumo.
Regra para entendimento de versões:
- Versões seguem o formato 'N.NN.N.NN' ou 'N.NN.N.NNN' (com 4 blocos de números).
- Os dois primeiros blocos são fixos e normalmente terão o valor '6.10'.
- O terceiro bloco representa a linha de versão funcional, podendo ser '3' ou '4':
    - A versão '4' é mais nova que a '3', mas ambas coexistem.
    - Compare builds sempre dentro da mesma linha de versão (ou seja, 6.10.4.X com 6.10.4.X).
- O quarto bloco é a build: um número incremental que indica a evolução semanal da versão.
    - Builds são lançadas toda semana, normalmente às sextas-feiras.
    - Exemplo: a build 6.10.3.112 foi lançada em 03/01/2025 e a 6.10.3.138 em 28/06/2025 (indicando uma build por semana).
    - O mesmo se aplica à linha 6.10.4: a build 6.10.4.49 foi lançada em 03/01/2025 e a 6.10.4.76 em 28/06/2025.
- Sempre que for dito algo como "a partir da versão 6.10.4.70", considere todas as builds da linha 6.10.4 com o último número maior ou igual a 70 (ex: 70, 71, 72...).

Gere um resumo técnico, didático e segmentado do conteúdo abaixo:

{documento}

Resumo:
-"""
)
summarizer = template_sumario | llm

# ✂️ Fragmentador
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

# 🔎 Embedding com modelo OpenAI text-embedding-3-large
print("[🧠 EMBEDDING] Carregando modelo 'text-embedding-3-large' via OpenAI...")
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_key,
    dimensions=3072
)

# 💰 Contador de consumo

def contar_tokens(texto: str, modelo: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(modelo)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(enc.encode(texto))

def custo_por_tokens(input_tokens: int, output_tokens: int, modelo: str) -> float:
    if modelo == "gpt-4o":
        preco_input = 0.005 / 1000
        preco_output = 0.015 / 1000
    elif modelo == "gpt-4":
        preco_input = 0.03 / 1000
        preco_output = 0.06 / 1000
    elif modelo == "gpt-4o-mini":
        preco_input = 0.0005 / 1000  # $0.0005 por mil tokens de entrada
        preco_output = 0.0015 / 1000 # $0.0015 por mil tokens de saída
    else:
        raise ValueError(f"Modelo '{modelo}' não suportado")

    return round((input_tokens * preco_input) + (output_tokens * preco_output), 6)


def registrar_custo(tipo: str, tokens_entrada: int, tokens_saida: int, custo_usd: float):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "tipo": tipo,
        "tokens_entrada": tokens_entrada,
        "tokens_saida": tokens_saida,
        "custo_usd": round(custo_usd, 6)
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/custos_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro) + "\n")

def custo_total_mes(mes: int = None, ano: int = None):
    if not os.path.exists("logs/custos_log.jsonl"):
        print("Nenhum registro encontrado.")
        return

    mes = mes or datetime.now().month
    ano = ano or datetime.now().year
    total = 0.0

    with open("logs/custos_log.jsonl", "r", encoding="utf-8") as f:
        for linha in f:
            try:
                dado = json.loads(linha)
                data = datetime.fromisoformat(dado["timestamp"])
                if data.month == mes and data.year == ano:
                    total += dado["custo_usd"]
            except Exception:
                continue

    print(f"[📊 ACUMULADO {mes:02d}/{ano}] Custo total (USD): ${total:.6f}")


# 📄 Cache de hash
hash_cache_path = "summaries/hash_cache.json"
hash_cache = {}
if os.path.exists(hash_cache_path):
    with open(hash_cache_path, "r", encoding="utf-8") as f:
        hash_cache = json.load(f)
def calcular_hash(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def validar_resumo_integridade(resumo_completo, hash_esperado):
    """
    Verifica se algum trecho do resumo contém o hash embutido e se ele bate com o esperado.
    Mais robusto contra variações no formato do arquivo.
    """
    for linha in resumo_completo.splitlines():
        linha = linha.strip()
        if linha.startswith("<!--hash=") and linha.endswith("-->"):
            hash_embutido = linha.replace("<!--hash=", "").replace("-->", "")
            return hash_embutido == hash_esperado
    return False


def gerar_sumario(texto, nome, hash_atual):
    print(f"[📜 RESUMO] Gerando resumo para: {nome}")
    resultado = summarizer.invoke({"documento": texto})
    resumo = resultado.content if hasattr(resultado, "content") else str(resultado)
    
    # Quebra de linha após o hash para garantir validação
    conteudo_final = f"<!--hash={hash_atual}-->\n{resumo.strip()}"
    
    with open(f"summaries/{nome}.resumo.txt", "w", encoding="utf-8") as f:
        f.write(conteudo_final)
    entrada_prompt = template_sumario.format(documento=texto)
    tokens_input = contar_tokens(entrada_prompt, llmModel)
    tokens_output = contar_tokens(resumo, llmModel)
    custo_sumario = custo_por_tokens(tokens_input, tokens_output, llmModel)

    print(f"[💰 CUSTO SUMARIZAÇÃO]")
    print(f"  • Tokens entrada: {tokens_input}")
    print(f"  • Tokens saída: {tokens_output}")
    print(f"  • Custo estimado (USD): ${custo_sumario}")
    registrar_custo("sumarizacao", tokens_input, tokens_output, custo_sumario)

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

    # Nova chamada para criar e logar embeddings e o índice FAISS
    faiss_index = criar_faiss_com_log(textos, embedding, index_path="index/faiss_index")
    return faiss_index


# Função auxiliar para gerar embeddings, logar custo e criar índice FAISS
def criar_faiss_com_log(textos, embedding, index_path="index/faiss_index"):
    docs_texts = [d.page_content for d in textos]
    # 1. Gera os embeddings
    embs = embedding.embed_documents(docs_texts)
    # 2. Calcula e loga o custo
    tokens_input = sum([contar_tokens(t, "text-embedding-3-large") for t in docs_texts])
    tokens_output = 0  # Embeddings não têm tokens de saída
    preco_input = 0.13 / 1_000_000  # preço do embedding-3-large
    custo_emb = round(tokens_input * preco_input, 6)
    registrar_custo("embedding_indexacao", tokens_input, tokens_output, custo_emb)
    print(f"Custo do Embedding: {custo_emb}")
    # 3. Cria o índice FAISS manualmente
    from langchain_community.vectorstores.faiss import FAISS
    faiss_index = FAISS(embs, textos)
    faiss_index.save_local(index_path)
    return faiss_index



# 🚦 Inicialização
print("[🚦 STARTUP] Carregando documentos e sumários...")
retriever = processar_documentos().as_retriever()

# 🧠 Prompt para responder perguntas
template_qa = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é a Sara, analista de suporte da Senior Sistemas, especialista no módulo eConsignado. Com base no conteúdo abaixo, responda à pergunta como se fosse uma orientação formal enviada a um cliente estratégico:

{context}

Pergunta: {question}

Instruções obrigatórias:

- Assine suas instruções como "Sara - Analista de Suporte"
- Se houver link relevante no conteúdo, priorize-o como parte essencial da resposta.
- Só combine informações de artigos diferentes se eles tratarem exatamente do mesmo processo, tela e funcionalidade.
- Nunca misture temas semelhantes com escopos distintos (ex: criação de CIF ≠ múltiplos contratos).
- Responda apenas ao que se aplica diretamente à pergunta. Seja claro, formal e objetivo.
- Se o conteúdo for omisso ou insuficiente, explique isso com transparência e oriente o que pode ser verificado — sem redirecionar.
- Nunca assuma que uma versão já atende, mesmo que seja superior. Sempre valide se a funcionalidade está de fato liberada.
- É proibido:
  • Inventar, induzir ou completar informações não presentes no conteúdo.
  • Transferir o atendimento (“entre em contato com o suporte” ou equivalentes).
  • Assumir limitações ou funcionalidades que não estejam explicitamente descritas.



Sobre versões:
- Formato: '6.10.3.120', '6.10.4.72', etc.
- Linha 6.10.4 é mais nova que 6.10.3, mas ambas coexistem.
- Compare builds apenas dentro da mesma linha.
- Builds são incrementais semanais. Ex: 6.10.4.49 (03/01/2025) → 6.10.4.76 (28/06/2025).
- “A partir da 6.10.4.70” significa qualquer 6.10.4.X com X ≥ 70.
"""
)
chain = template_qa | llm



class Pergunta(BaseModel):
    pergunta: str
    versao: Optional[str] = "(Versão não informada)"

def normalizar_texto(texto: str) -> str:
    return re.sub(r"[^\w\s]", " ", texto.lower())


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


def lematizar(texto: str) -> set:
    doc = nlp(texto.lower())
    return {token.lemma_ for token in doc if not token.is_stop and token.is_alpha}

def pergunta_e_relevante(pergunta: str, categoria: str = CATEGORIA_PADRAO) -> bool:
    """
    Verifica se a pergunta contém conceitos e funções relevantes lematizados
    com base na configuração por categoria.
    """
    config = CATEGORIAS_CONFIG.get(categoria)
    if not config:
        print(f"[⚠️ AVISO] Categoria '{categoria}' não encontrada. Usando categoria padrão '{CATEGORIA_PADRAO}'.")
        config = CATEGORIAS_CONFIG[CATEGORIA_PADRAO]

    conceitos_cfg = config.get("conceitos_relevantes", set())
    funcoes_cfg = config.get("verbos_funcionais", set())

    lemas_pergunta = lematizar(pergunta)

    conceitos_encontrados = lemas_pergunta & conceitos_cfg
    funcoes_encontradas = lemas_pergunta & funcoes_cfg

    print(f"[🔎 LEMAS] Categoria: {categoria}")
    print(f"[🔎 LEMAS] Pergunta lematizada: {lemas_pergunta}")
    print(f"[🔎 LEMAS] Conceitos encontrados: {conceitos_encontrados}")
    print(f"[🔎 LEMAS] Funções encontradas: {funcoes_encontradas}")

    return (
        len(conceitos_encontrados) >= 2
        or (len(conceitos_encontrados) >= 1 and len(funcoes_encontradas) >= 1)
    )




def filtrar_docs_por_palavras_chave(docs, pergunta: str, limite: int = 5):
    """
    Reordena documentos retornados pelo FAISS com base na similaridade semântica
    (via lemas) entre a pergunta e o conteúdo do documento.

    Mantém todos os documentos, mas prioriza os mais relevantes.
    """
    doc_pergunta = nlp(pergunta.lower())
    lemas_pergunta = {token.lemma_ for token in doc_pergunta if not token.is_stop and not token.is_punct}

    ranking = []
    for doc in docs:
        doc_text = doc.page_content.lower()
        lemas_doc = {token.lemma_ for token in nlp(doc_text) if not token.is_stop and not token.is_punct}
        intersecao = lemas_pergunta.intersection(lemas_doc)
        score = len(intersecao)
        ranking.append((score, doc))

    # Ordena por número de lemas em comum (decrescente)
    ranking.sort(reverse=True, key=lambda x: x[0])

    docs_ordenados = [doc for score, doc in ranking if score > 0]

    # fallback: se nada tiver intersecção, retorna tudo como veio
    return docs_ordenados if docs_ordenados else docs


# Bloco de validação do conteúdo da pergunta:

def pergunta_tem_objeto_valido(pergunta: str, categoria: str = CATEGORIA_PADRAO) -> bool:
    """
    Avalia se a pergunta apresenta intenção funcional válida para a categoria escolhida.
    Usa:
    - Verbos funcionais (ações permitidas).
    - Termos técnicos relacionados à funcionalidade.

    A categoria define quais expressões e verbos são válidos.
    """
    config = CATEGORIAS_CONFIG.get(categoria)
    if not config:
        print(f"[⚠️ AVISO] Categoria '{categoria}' não encontrada. Usando categoria padrão '{CATEGORIA_PADRAO}'.")
        config = CATEGORIAS_CONFIG[CATEGORIA_PADRAO]

    verbos_funcionais = config.get("verbos_funcionais", set())
    termos_tecnicos = config.get("termos_tecnicos", set())

    doc = nlp(pergunta.lower())
    lemas = {token.lemma_ for token in doc if not token.is_stop and not token.is_punct}

    # 🔍 1. Verifica se há verbo funcional explícito
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in verbos_funcionais:
            return True
        if token.pos_ == "AUX":
            for child in token.head.children:
                if child.pos_ == "VERB" and child.lemma_ in verbos_funcionais:
                    return True


    # 🔍 2. Verifica se há termos técnicos relevantes
    if lemas.intersection(termos_tecnicos):
        return True

    return False
def normalizar_pergunta(pergunta):
    if len(pergunta) <= 350:
        return pergunta

    prompt = PromptTemplate(
        input_variables=["pergunta"],
        template="""
Você é um assistente técnico da Senior Sistemas.

Reescreva a pergunta abaixo de forma objetiva e técnica, mantendo o foco funcional no módulo eConsignado. 
Preserve qualquer número de versão informado (ex: 6.10.4.71), sem alterar seu formato. 
Não adicione termos que não estavam no texto original.

Pergunta original:
{pergunta}

Pergunta reescrita:"""
    )

    from langchain_openai import ChatOpenAI
    mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=openai_key)
    chain = prompt | mini_llm

    resultado = chain.invoke({"pergunta": pergunta})
    pergunta_normalizada = resultado.content.strip() if hasattr(resultado, "content") else str(resultado).strip()

    # Correção de inserção indevida de "eConsignado" pela LLM
    if "econsignado" not in pergunta.lower() and "econsignado" in pergunta_normalizada.lower():
        print("[⚠️ AJUSTE] 'eConsignado' adicionado indevidamente. Removendo...")
        pergunta_normalizada = re.sub(r"\b(do|no)?\s*econsignado\b", "", pergunta_normalizada, flags=re.IGNORECASE).strip()

    entrada_prompt = prompt.format(pergunta=pergunta)
    tokens_input = contar_tokens(entrada_prompt, modelo="gpt-4o-mini")
    tokens_output = contar_tokens(pergunta_normalizada, modelo="gpt-4o-mini")
    custo = custo_por_tokens(tokens_input, tokens_output, modelo="gpt-4o-mini")
    registrar_custo("normalizacao", tokens_input, tokens_output, custo)

    print(f"[✂️ NORMALIZADA] {pergunta_normalizada}")
    return pergunta_normalizada





# ⚠️ Frases proibidas que indicam transferência de responsabilidade para o suporte
sinais_baixa_confianca = [
    "não sei", "não encontrei", "não tenho informação", "sem informações suficientes",
    "consulte o suporte", "entre em contato com o suporte", "não há informação suficiente",
    "não há detalhes", "não está claro", "não posso afirmar", "não posso garantir",
    "não está disponível", "não tenho detalhes", "precisa fornecer mais detalhes",
    "contate o suporte oficial", "recomendo que entre em contato",
    "abra um chamado", "encaminhe para o suporte", "fale com o suporte", "entre em contato com nosso suporte"
]

def resposta_para_pergunta(pergunta: str, versao: str = "Versão não informada") -> dict:
    #print(f"[🟡 PERGUNTA] {pergunta} | Versão: {versao}")

    pergunta = normalizar_pergunta(pergunta)
    print(f"[🟡PERGUNTA Normalizada] {pergunta} | Versão: {versao}")

    # 🚫 Bloqueio de perguntas relacionadas a versões instáveis (alfa, beta etc.)
#    termos_proibidos = [" alfa ", " alpha ", " aufa ", " beta "]

#    pergunta_lower = pergunta.lower()
#    if any(termo in pergunta_lower for termo in termos_proibidos):
#        print("[🚫 BLOQUEIO] Pergunta sobre versão instável detectada.")
#        log_pergunta(pergunta, "[Motivo: Pergunta sobre versão instável (alfa/beta) bloqueada por política interna]")
#        return {
#            "resposta": (
#                "Perguntas relacionadas a versões instáveis (como 'versão alfa' ou 'versão beta') "
#                "não são processadas por este assistente. Aguarde a liberação oficial da versão antes de solicitar suporte automatizado."
#            ),
#            "responder": False
#        }


    pergunta_relevante = pergunta_e_relevante(pergunta, categoria="eConsignado")
    if not pergunta_relevante:
        print("[⚠️ RELEVÂNCIA] Pergunta fora do escopo do eConsignado.")
        log_pergunta(pergunta, "[Motivo: Pergunta fora do escopo do eConsignado]")
        return {
            "resposta": (
                "A pergunta enviada não parece estar relacionada ao módulo eConsignado. "
                "Reformule ou envie uma dúvida mais específica."
            ),
            "responder": False
        }

    pergunta_tem_objeto_valido(pergunta, categoria="eConsignado")
    if not pergunta_tem_objeto_valido:
        print("[❌ OBJETO] Pergunta com intenção fora do escopo funcional permitido.")
        log_pergunta(pergunta, "[Motivo: Intenção fora do escopo funcional permitido]")
        return {
            "resposta": (
                "Essa pergunta trata de uma ação que não está coberta pelo escopo do assistente. "
                "O foco são dúvidas sobre o uso funcional do eConsignado, como lançar, cadastrar, processar ou consultar dados do colaborador."
            ),
            "responder": False
        }

    if versao != "Versão não informada":
        pergunta = f"{pergunta} (considerar para esta pergunta a versão: {versao})"

    print("[🔍 BUSCA] Selecionando trechos relevantes...")
    docs = retriever.invoke(pergunta)
    tokens_input = contar_tokens(pergunta, "text-embedding-3-large")
    tokens_output = 0
    preco_input = 0.13 / 1_000_000  # preço do embedding-3-large
    custo_emb = round(tokens_input * preco_input, 6)
    registrar_custo("embedding_consulta", tokens_input, tokens_output, custo_emb)
    print(f"Custo da consulta no Embedding: {custo_emb}")
    docs_filtrados = filtrar_docs_por_palavras_chave(docs, pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs_filtrados])
    contexto_com_versao = f"[Versão informada na pergunta: {versao}]\n\n{contexto}"

    print("[💬 LLM] Gerando resposta...")
    resposta_raw = chain.invoke({"context": contexto_com_versao, "question": pergunta})
    resposta_texto = resposta_raw.content if hasattr(resposta_raw, "content") else str(resposta_raw)
# Medir tokens de entrada e saída para cálculo de custo

    entrada_prompt = template_qa.format(context=contexto_com_versao, question=pergunta)
    saida_resposta = resposta_raw.content if hasattr(resposta_raw, "content") else str(resposta_raw)

    tokens_input = contar_tokens(entrada_prompt, llmModel)
    tokens_output = contar_tokens(saida_resposta, llmModel)
    custo_total = custo_por_tokens(tokens_input, tokens_output, llmModel)

    print(f"[💰 CUSTO GPT]")
    print(f"  • Tokens entrada: {tokens_input}")
    print(f"  • Tokens saída: {tokens_output}")
    print(f"  • Custo estimado (USD): ${custo_total}")
    registrar_custo("resposta", tokens_input, tokens_output, custo_total)


    def validar_resposta(pergunta: str, resposta: str) -> dict:
        resposta_lower = resposta.lower()

        sinais_confianca = sum(1 for sinal in sinais_baixa_confianca if sinal in resposta_lower)
        palavras_chave_resposta = [
            "econsignado", "folha de pagamento", "empréstimo", "desconto em folha",
            "colaborador", "provisão", "consignado", "convênio", "crédito do trabalhador",
            "ficha financeira", "evento"
        ]
        tem_palavra_chave = any(p in resposta_lower for p in palavras_chave_resposta)
        tamanho_suficiente = len(resposta.split()) >= 30

        resposta_confiavel = sinais_confianca < 2 and tem_palavra_chave and tamanho_suficiente
        cobertura_intencao = True  # já validado antes
        responder = resposta_confiavel and cobertura_intencao and pergunta_relevante

        print("[✅ VALIDAÇÃO FINALIZADA]")
        print(f"  • Resposta confiável: {resposta_confiavel}")
        print(f"  • Intenção coberta: {cobertura_intencao}")
        print(f"  • Responder: {responder}")

        return {
            "responder": responder,
            "resposta_confiavel": resposta_confiavel,
            "intencao_coberta": cobertura_intencao
        }

    validacao = validar_resposta(pergunta, resposta_texto)
    responder = validacao["responder"]

    # 🚫 Validação extra para garantir que nenhuma frase proibida passe despercebida
    if any(x in resposta_texto.lower() for x in sinais_baixa_confianca):
        print("[🚫 BLOQUEIO FINAL] Resposta continha frases proibidas.")
        resposta_texto = (
            "A IA não encontrou uma solução clara com base no conteúdo disponível. "
            "Por favor, revise os parâmetros da pergunta ou verifique se a situação se encaixa em funcionalidades já documentadas no eConsignado."
        )
        responder = False

    log_pergunta(pergunta, resposta_texto)

    return {
        "resposta": resposta_texto.strip(),
        "responder": responder
    }


# 🚀 FastAPI
app = FastAPI(title="Agente eConsignado - RAG Multilingual MPNet", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 🎯 Endpoint de pergunta
@app.post("/perguntar/")
def perguntar(p: Pergunta):
    print(f"[🟡 PERGUNTA] {p.pergunta} | Versão: {p.versao}")
    inicio = time.time()

    # 🚦 BLOQUEIO DE POC
    limite_poc_usd = 50.0
    total_custo = custo_total_acumulado(limite_usd=limite_poc_usd)
    if total_custo >= limite_poc_usd:
        print(f"[🚫 LIMITE POC ATINGIDO] Custo total acumulado: ${total_custo:.2f}")
        return {
            "resposta": "Período de POC expirado, desativar o assistente ou rever o período $",
            "responder": False
        }
    else:
        print(f"[CUSTO ATUAL] Custo total acumulado: ${total_custo:.6f}")
    

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

#Para buscar custo mensal: http://localhost:8000/custo_mes?mes=7&ano=2025
@app.get("/custo_mes")
def custo_mes(mes: int = None, ano: int = None):
    from fastapi.responses import JSONResponse

    mes = mes or datetime.now().month
    ano = ano or datetime.now().year
    total = 0.0

    if os.path.exists("logs/custos_log.jsonl"):
        with open("logs/custos_log.jsonl", "r", encoding="utf-8") as f:
            for linha in f:
                try:
                    dado = json.loads(linha)
                    data = datetime.fromisoformat(dado["timestamp"])
                    if data.month == mes and data.year == ano:
                        total += dado["custo_usd"]
                except Exception:
                    continue

    return JSONResponse({
        "mes": mes,
        "ano": ano,
        "custo_total_usd": round(total, 6)
    })