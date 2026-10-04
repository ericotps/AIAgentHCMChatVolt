#zendesk_bot_prod.py					

from datetime import datetime, timedelta, timezone
import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# 🔐 Carregar variáveis de ambiente
load_dotenv()
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_TOKEN = os.getenv("ZENDESK_API_TOKEN")
ZENDESK_SUBDOMAIN = "cxsenior"
IA_ENDPOINT = "http://localhost:8000/perguntar/"

# 🕒 Calcula o timestamp para 20 minutos atrás
agora = datetime.now(timezone.utc)
vinte_minutos_atras = agora - timedelta(minutes=10)
created_after = vinte_minutos_atras.strftime("%Y-%m-%dT%H:%M:%SZ")
# 🔎 Query com filtro por data de criação e tag
query = (
    f"type:ticket tags:prod_gestao_de_pessoas_hcm_proc_windows created>={created_after}"
)
url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/search.json?query={query}"

print(f"[🔗] Executando chamada para: {url}")
print("[Autenticação] Credenciais carregadas do ambiente.")
if ZENDESK_TOKEN:
    print("[Autenticação] Token configurado.")
else:
    print("[❌] Token não carregado. Verifique seu arquivo .env")

response = requests.get(url, auth=(f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN))

tickets_filtrados = []
if response.status_code == 200:
    data = response.json()
    print(f"[✅] {len(data['results'])} tickets retornados pela API.")

    grupos_validos = {
        "360018327392",
        "360018327352",
        "10758177846164",
        "32882571659668",
        "360018333271"
    }

    for ticket in data["results"]:
        campos = {str(field["id"]): field["value"] for field in ticket.get("custom_fields", [])}

        # ⚠️ Verificações adicionais: assignee_id == None e group_id válido
        if (
            campos.get("360044037571") == "produto_gestao_de_pessoas_hcm" and
            campos.get("360044429192") == "prod_gestao_de_pessoas_hcm_proc_windows" and
            campos.get("1900001430487") == "Administração de Pessoal" and
            campos.get("360047792451") == "Crédito do Trabalhador - eConsignado" and 
            ticket.get("assignee_id") is None and
            str(ticket.get("group_id")) in grupos_validos
        ):
			# Ignorar tickets que já possuem a tag 'Resposta_IA'													 
            if "resposta_ia" in ticket.get("tags", []) or "resposta_ia_false" in ticket.get("tags", []):
                print(f"[⏩] Ticket #{ticket['id']} já possui tag 'Resposta_IA' ou 'Resposta_IA_False'. Pulando...")
                continue
            tickets_filtrados.append(ticket)

    print(f"[🎯] {len(tickets_filtrados)} tickets mantidos após filtro por campos personalizados e grupo.")
else:
    print(f"[❌] Erro ao buscar tickets: {response.status_code} - {response.text}")

# 📋 Exibindo dados relevantes dos tickets filtrados e chamando a IA
class Pergunta(BaseModel):
    pergunta: str

for ticket in tickets_filtrados:
    ticket_id = ticket["id"]
    titulo = ticket.get("raw_subject", "(Sem título)")
    descricao = ticket.get("description", "(Sem descrição)")
    assignee = ticket.get("assignee_id", "(Sem responsável)")
    versao = next(
        (field["value"] for field in ticket.get("custom_fields", []) if str(field["id"]) == "360044024752"),
        "(Versão não informada)"
    )
    print(f"[📄] Ticket #{ticket_id}\n  • Título: {titulo}\n  • Descrição: {descricao}\n  • Versão: {versao} Responsável: {assignee}")

    pergunta_formatada = f"{titulo}\n{descricao}\nVersão: {versao}"
    print(f"[🤖] Enviando para IA: {pergunta_formatada}")

    try:
        response_ia = requests.post(IA_ENDPOINT, json={"pergunta": pergunta_formatada})
        if response_ia.status_code == 200:
            resposta_ia = response_ia.json().get("resposta", "(Sem resposta)")
            print(f"[✅] Resposta da IA: {resposta_ia}\n")

            # 🚀 Enviando resposta de volta ao Zendesk
            resposta_ia = response_ia.json().get("resposta", "(Sem resposta)")														   
            responder = response_ia.json().get("responder", False)
            print(f"[✅] Resposta da IA: {resposta_ia}\n")
            print(f"[🧠] Confiabilidade da IA: {'Alta' if responder else 'Baixa'}")

            update_url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}.json"
            headers = {"Content-Type": "application/json"}

			# 🎯 Campos comuns para ambos os cenários											
            custom_fields = [
                {
                    "id": 360044030251,
                    "value": "qualificacao_linha_de_suporte_produto_conhecimento_sistema_negocio_hd1"
                },
                {
                    "id": 13243462675732,
                    "value": "abertura_completa_sim"
                }
            ]

			# ✏️ Define conteúdo conforme confiabilidade															 
            if responder:
                # 🔀 Definir assignee_id com base no group_id
                group_id = str(ticket.get("group_id"))
                assignee_id = None
                if group_id in {"360018327352", "10758177846164", "32882571659668", "360018333271"}:
                    assignee_id = 419911387612
                elif group_id == "360018327392":
                    assignee_id = 419931542512

                payload = {
                    "ticket": {
                        "status": "solved",
                        "assignee_id": assignee_id,
                        "comment": {
                            "body": resposta_ia,
                            "public": True
                        },
                        "tags": list(set(ticket.get("tags", []) + ["Resposta_IA"])),
                        "custom_fields": custom_fields
                    }
                }
            else:
                payload = {
                    "ticket": {
                        "comment": {
                            "body": f"A IA não conseguiu responder com confiança. Encaminhe para análise manual.\n\nResposta sugerida pela IA:\n{resposta_ia}",
                            "public": False
                        },
                        "tags": list(set(ticket.get("tags", []) + ["resposta_ia_false"])),
                        "custom_fields": custom_fields
                    }
                }

            response_update = requests.put(update_url, json=payload, auth=(f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN), headers=headers)
            if response_update.status_code == 200:
                print(f"[📬] Atualização registrada para o ticket #{ticket_id} ({'Resolvido' if responder else 'Comentado internamente'}).")
            else:
                print(f"[⚠️] Falha ao atualizar ticket #{ticket_id}: {response_update.status_code} - {response_update.text}")



        else:
            print(f"[❌] Erro ao consultar IA: {response_ia.status_code} - {response_ia.text}")
    except Exception as e:
        print(f"[⚠️] Falha na chamada HTTP para IA: {e}")
