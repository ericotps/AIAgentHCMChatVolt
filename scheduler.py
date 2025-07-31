
# scheduler.py - Responsável pelo serviço de agendamento do processo do BOT
import time
import subprocess
from datetime import datetime
import pytz

# Fuso horário de Brasília
br_tz = pytz.timezone('America/Sao_Paulo')

while True:
    now_br = datetime.now(br_tz)
    weekday = now_br.weekday()  # 0 = segunda, 6 = domingo
    hour = now_br.hour

    # Segunda a sexta (0 a 4), entre 7h e 19h (inclusive 19:00)
    if 0 <= weekday <= 4 and 7 <= hour < 19:
        print(f"[{now_br.strftime('%Y-%m-%d %H:%M:%S')}] INFO Executando zendesk_bot_prod.py...")
        subprocess.run(["python", "zendesk_bot_prod.py"])
        print(f"[{now_br.strftime('%Y-%m-%d %H:%M:%S')}] INFO Aguardando 10 minutos para próxima execução...\n")
    else:
        print(f"[{now_br.strftime('%Y-%m-%d %H:%M:%S')}] INFO Fora do horário de execução (seg a sex, 07-19h).")
        print(f"[{now_br.strftime('%Y-%m-%d %H:%M:%S')}] INFO Aguardando 10 minutos para nova verificação...\n")

    time.sleep(600)  # 10 minutos = 600 segundos

