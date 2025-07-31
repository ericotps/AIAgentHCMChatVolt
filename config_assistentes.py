# config_assistentes.py

# 📝 Este arquivo centraliza os termos e expressões usados para validar perguntas por categoria de atendimento.
# Para adicionar uma nova frente, crie uma nova entrada no dicionário `CATEGORIAS_CONFIG`, usando o exemplo abaixo como modelo.

CATEGORIAS_CONFIG = {
    "eConsignado": {
        # ✅ Termos lematizados que indicam contexto relevante ao eConsignado
        "conceitos_relevantes": {
            "econsignado", "consignado", "empréstimo", "folha", "pagamento", "cif", 
            "provisão", "desconto", "convênio", "margem", "crédito", "trabalhador",
            "evento", "colaborador", "rescisão", "parcela", "financeira", "ficha", 
            "descontar", "consignável", "automático", "complementar", "diferença", 
            "criar", "rescindir", "reajuste", "ajuste", "ajustar", "parametrizar",
            "comportamento", "mudança", "liberação", "adiantamento", "adt salarial", 
            "documentação"
        },


        # ✅ Verbos lematizados que indicam ações esperadas na pergunta
        "verbos_funcionais": {
            "cadastrar", "lançar", "registrar", "corrigir", "excluir", "visualizar",
            "ajustar", "consultar", "emitir", "gerar", "processar", "validar", "importar",
            "criar", "precisar", "transitar", "utilizar", "ocorrer", "acontecer",
            "movimentar", "indicar", "tratar", "impactar", "descontar", "parametrizar",
            "configurar", "definir"
        },


        # ✅ Termos técnicos que reforçam a presença de contexto funcional
        "termos_tecnicos": {
            "evento", "regra", "característica", "parametrização", "parâmetro", 
            "crédito", "provisão", "econsignado", "desconto", "rescisão",
            "recibo", "férias", "configuração", "lançamento", "cadastro",
            "verba", "código", "colaborador", "versão", "documentação", 
            "orientação", "cif", "título"
        }
    },

    # Exemplo: outra frente
    # "folha_pagamento": {
    #     "termos_lematizados": {...},
    #     "verbos_funcionais": {...},
    #     "termos_tecnicos": {...}
    # }
}

# 🔧 Define qual categoria será usada por padrão se não for especificada
CATEGORIA_PADRAO = "eConsignado"

# 🔐 Regras de validação de respostas por categoria
VALIDACAO_RESPOSTA = {
    "eConsignado": {
        "frases_proibidas": [
            "não sei", "não encontrei", "não tenho informação", "sem informações suficientes",
            "consulte o suporte", "entre em contato com o suporte", "não há informação suficiente",
            "não há detalhes", "não está claro", "não posso afirmar", "não posso garantir",
            "não está disponível", "não tenho detalhes", "precisa fornecer mais detalhes",
            "documentação oficial", "contate o suporte oficial", "recomendo que entre em contato",
            "abra um chamado", "encaminhe para o suporte", "fale com o suporte", "entre em contato com nosso suporte"
        ],
        "palavras_chave": [
            "econsignado", "folha de pagamento", "empréstimo", "desconto em folha",
            "colaborador", "provisão", "consignado", "convênio", "adiantamento", "adt salarial"
        ],
        "negacoes_proibidas": [
            "o sistema não", "não possui", "não oferece", "não especifica",
            "não permite", "não existe", "não há suporte", "não é possível"
        ],
        "tamanho_minimo": 30
    }
}

# Função utilitária para recuperar configuração segura por categoria
def get_config_validacao_resposta(categoria: str) -> dict:
    return VALIDACAO_RESPOSTA.get(categoria, VALIDACAO_RESPOSTA["eConsignado"])
