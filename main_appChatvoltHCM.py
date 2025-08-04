from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ticket_storeHCM import get_ticket_info, save_ticket_info
from chatvolt_client import perguntar_chatvolt
import uvicorn

app = FastAPI()

class PerguntaRequest(BaseModel):
    ticket: str
    pergunta: str

@app.post("/perguntarChatVolt")
async def perguntar_chatvolt_endpoint(data: PerguntaRequest):
    ticket_info = get_ticket_info(data.ticket)

    try:
        if ticket_info:
            response = await perguntar_chatvolt(
                data.pergunta,
                visitor_id=ticket_info["visitorId"],
                conversation_id=ticket_info["conversationId"]
            )
        else:
            response = await perguntar_chatvolt(data.pergunta)
            save_ticket_info(
                data.ticket,
                response["visitorId"],
                response["conversationId"]
            )

        return {
            "resposta": response.get("resposta", ""),
            "responder": response.get("responder", False),
            "visitorId": response.get("visitorId"),
            "conversationId": response.get("conversationId")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
