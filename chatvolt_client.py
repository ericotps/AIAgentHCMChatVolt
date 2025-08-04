import httpx
import os
import re
from dotenv import load_dotenv

load_dotenv()

CHATVOLT_API_URL = "https://api.chatvolt.ai/agents"
AGENT_ID = "cmd60wtvr01nb6cuw8jzv5lxs"
BEARER_TOKEN = os.getenv("CHATVOLT_BEARER_TOKEN")

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

# ⚠️ Frases que indicam baixa confiabilidade
sinais_baixa_confianca = [
    "não sei", "não encontrei", "não tenho informação", "sem informações suficientes",
    "consulte o suporte", "entre em contato com o suporte", "não há informação suficiente",
    "não há detalhes", "não está claro", "não posso afirmar", "não posso garantir",
    "não está disponível", "não tenho detalhes", "precisa fornecer mais detalhes",
    "contate o suporte oficial", "recomendo que entre em contato",
    "abra um chamado", "encaminhe para o suporte", "fale com o suporte", "entre em contato com nosso suporte"
]

palavras_chave_resposta = [
    "econsignado", "folha de pagamento", "empréstimo", "desconto em folha",
    "colaborador", "provisão", "consignado", "convênio", "crédito do trabalhador",
    "ficha financeira", "evento", "wildfly", "glassfish", "servidor", "processo"
]

def validar_resposta(pergunta: str, resposta: str) -> dict:
    resposta_lower = resposta.lower()
    sinais_confianca = sum(1 for sinal in sinais_baixa_confianca if sinal in resposta_lower)
    tem_palavra_chave = any(p in resposta_lower for p in palavras_chave_resposta)
    tamanho_suficiente = len(resposta.split()) >= 30

    resposta_confiavel = sinais_confianca < 2 and tem_palavra_chave and tamanho_suficiente
    cobertura_intencao = True  # aqui você pode adicionar uma lógica real se quiser
    pergunta_relevante = True  # mesma coisa

    responder = resposta_confiavel and cobertura_intencao and pergunta_relevante

    # Validação final para bloquear certas frases
    if any(x in resposta_lower for x in sinais_baixa_confianca):
        resposta = (
            "A IA não encontrou uma solução clara com base no conteúdo disponível. "
            "Por favor, revise os parâmetros da pergunta ou verifique se a situação se encaixa em funcionalidades já documentadas."
        )
        responder = False

    print("[✅ VALIDAÇÃO FINALIZADA]")
    print(f"  • Resposta confiável: {resposta_confiavel}")
    print(f"  • Responder: {responder}")

    return {
        "responder": responder,
        "resposta_texto": resposta
    }

def limpar_imagens_markdown(texto):
    return re.sub(r'!\[.*?\]\(.*?\)', '', texto)

async def perguntar_chatvolt(pergunta, visitor_id=None, conversation_id=None):
    pergunta_limpa = limpar_imagens_markdown(pergunta)
    payload = {"query": pergunta_limpa}

    if visitor_id and conversation_id:
        payload["visitorId"] = visitor_id
        payload["conversationId"] = conversation_id

    url = f"{CHATVOLT_API_URL}/{AGENT_ID}/query"
    print(f"[🔍] URL: {url}")
    print(f"[🔍] Payload: {payload}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            print(f"[🔍] Status code: {response.status_code}")
            print(f"[🔍] Response body: {response.text}")
            response.raise_for_status()
            dados = response.json()

            texto_resposta = dados.get("answer", "")
            validacao = validar_resposta(pergunta_limpa, texto_resposta)

            return {
                "resposta": validacao["resposta_texto"],
                "responder": validacao["responder"],
                "visitorId": dados.get("visitorId"),
                "conversationId": dados.get("conversationId")
            }

        except httpx.HTTPStatusError as e:
            print("[❌] Erro de status HTTP:")
            print("Status:", e.response.status_code)
            print("Conteúdo:", e.response.text)
            raise
        except Exception as e:
            print("[❌] Erro inesperado:")
            import traceback
            traceback.print_exc()
            raise
