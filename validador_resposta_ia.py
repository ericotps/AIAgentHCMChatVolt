
# ✅ Validador unificado para respostas de assistentes IA
from config_assistentes import get_config_validacao_resposta

def validar_resposta_final(
    resposta: str,
    contexto: str,
    categoria: str,
    pergunta_relevante: bool,
    intencao_coberta: bool = True
) -> dict:
    cfg = get_config_validacao_resposta(categoria)
    resposta_lower = resposta.lower()
    contexto_lower = contexto.lower()

    frases_proibidas = [s for s in cfg["frases_proibidas"] if s in resposta_lower]
    tem_palavras_chave = any(p in resposta_lower for p in cfg["palavras_chave"])
    tamanho_minimo = len(resposta.split()) >= cfg.get("tamanho_minimo", 30)
    negacoes_sem_contexto = [n for n in cfg["negacoes_proibidas"] if n in resposta_lower and n not in contexto_lower]

    resposta_confiavel = (
        len(frases_proibidas) < 2 and tem_palavras_chave and tamanho_minimo and not negacoes_sem_contexto
    )

    responder = resposta_confiavel and pergunta_relevante and intencao_coberta

    print("[✅ VALIDAÇÃO FINALIZADA]")
    print(f"  • Frases proibidas detectadas: {frases_proibidas}")
    print(f"  • Alucinações negativas detectadas: {negacoes_sem_contexto}")
    print(f"  • Palavras-chave encontradas: {tem_palavras_chave}")
    print(f"  • Tamanho suficiente: {tamanho_minimo}")
    print(f"  • Resposta confiável: {resposta_confiavel}")
    print(f"  • Deve responder: {responder}")

    return {
        "responder": responder,
        "resposta_confiavel": resposta_confiavel,
        "frases_proibidas": frases_proibidas,
        "alucinacoes_negativas": negacoes_sem_contexto
    }
