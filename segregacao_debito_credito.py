"""
Segregação de Débito e Crédito (com Partidas Dobradas)
------------------------------------------------------
Junta as colunas "Conta Débito" e "Conta Crédito" em uma única coluna W_CONTA
e gera o valor segregado W_VALOR:
    - Débito  -> valor POSITIVO
    - Crédito -> valor NEGATIVO

Regras:
    1) PARTIDA DOBRADA (Conta Débito E Conta Crédito preenchidas na mesma linha):
         a linha é desdobrada em 2 linhas de saída:
           - perna de débito : W_CONTA = Conta Débito  | W_TIPO = D | W_VALOR = +Valor
           - perna de crédito: W_CONTA = Conta Crédito | W_TIPO = C | W_VALOR = -Valor
         (a coluna DC é ignorada, pois a linha traz os dois lados)

    2) LINHA SIMPLES (só um dos lados preenchido): 1 linha de entrada = 1 de saída
         - DC = "D" -> W_CONTA = Conta Débito  | W_VALOR = +Valor
         - DC = "C" -> W_CONTA = Conta Crédito | W_VALOR = -Valor
         - DC vazia/inválida -> usa o lado que estiver preenchido

    3) Linhas que não puderem ser classificadas vão para a aba "Pendencias",
       com o motivo na coluna W_MOTIVO.

Colunas novas: W_TIPO, W_CONTA, W_VALOR, W_ORIGEM (PARTIDA_DOBRADA / LINHA_SIMPLES).
"""

import sys
import numpy as np
import pandas as pd

# CONFIGURAÇÃO - ajuste os nomes conforme as colunas do seu arquivo
COL_DEBITO = "Conta Débito"
COL_CREDITO = "Conta Crédito"
COL_VALOR = "Valor"
COL_DC = "DC"

COL_CONTA_SAIDA = "W_CONTA"
COL_VALOR_SAIDA = "W_VALOR"
COL_TIPO_SAIDA = "W_TIPO"
COL_ORIGEM_SAIDA = "W_ORIGEM"
COL_MOTIVO_SAIDA = "W_MOTIVO"

ORIGEM_PARTIDA_DOBRADA = "PARTIDA_DOBRADA"
ORIGEM_LINHA_SIMPLES = "LINHA_SIMPLES"


def _normalizar_conta(serie: pd.Series) -> pd.Series:
    """Converte a conta para inteiro (sem '.0') quando ela é numérica;
    se houver contas em texto (ex.: '1.1.01'), mantém como estão."""
    num = pd.to_numeric(serie, errors="coerce")
    if num.notna().sum() != serie.notna().sum():
        return serie
    try:
        return num.astype("Int64")
    except (TypeError, ValueError):
        return serie


def segregar_debito_credito(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (dados_segregados, pendencias)."""
    for col in (COL_DEBITO, COL_CREDITO, COL_VALOR):
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    base = df.copy()
    valor = pd.to_numeric(base[COL_VALOR], errors="coerce")

    tem_deb = base[COL_DEBITO].notna()
    tem_cred = base[COL_CREDITO].notna()
    eh_partida_dobrada = tem_deb & tem_cred

    # A) PARTIDAS DOBRADAS: cada linha gera duas (perna de débito + perna de crédito)
    pd_base = base[eh_partida_dobrada]
    pd_valor = valor[eh_partida_dobrada].abs()

    perna_deb = pd_base.copy()
    perna_deb[COL_TIPO_SAIDA] = "D"
    perna_deb[COL_CONTA_SAIDA] = pd_base[COL_DEBITO]
    perna_deb[COL_VALOR_SAIDA] = pd_valor
    perna_deb[COL_ORIGEM_SAIDA] = ORIGEM_PARTIDA_DOBRADA

    perna_cred = pd_base.copy()
    perna_cred[COL_TIPO_SAIDA] = "C"
    perna_cred[COL_CONTA_SAIDA] = pd_base[COL_CREDITO]
    perna_cred[COL_VALOR_SAIDA] = -pd_valor
    perna_cred[COL_ORIGEM_SAIDA] = ORIGEM_PARTIDA_DOBRADA

    # B) LINHAS SIMPLES: só um lado preenchido -> tipo pela coluna DC
    simples = base[~eh_partida_dobrada].copy()
    v_abs = valor[~eh_partida_dobrada].abs()

    if COL_DC in simples.columns:
        tipo = simples[COL_DC].astype(str).str.strip().str.upper().where(lambda s: s.isin(["D", "C"]))
    else:
        tipo = pd.Series(np.nan, index=simples.index, dtype="object")

    # Plano B quando DC está vazio/inválido: usa o lado que estiver preenchido
    lado = pd.Series(
        np.where(tem_deb[~eh_partida_dobrada], "D",
                 np.where(tem_cred[~eh_partida_dobrada], "C", None)),
        index=simples.index, dtype="object",
    )
    tipo = tipo.fillna(lado)

    simples[COL_TIPO_SAIDA] = tipo
    simples[COL_CONTA_SAIDA] = simples[COL_DEBITO].where(
        tipo == "D", simples[COL_CREDITO].where(tipo == "C")
    )
    simples[COL_VALOR_SAIDA] = v_abs.where(tipo == "D", (-v_abs).where(tipo == "C"))
    simples[COL_ORIGEM_SAIDA] = ORIGEM_LINHA_SIMPLES

    # C) Junta tudo, mantém as pernas do mesmo lançamento lado a lado
    partes = [p for p in (perna_deb, perna_cred, simples) if not p.empty]
    out = pd.concat(partes) if partes else simples
    out = out.sort_index(kind="stable")
    out[COL_CONTA_SAIDA] = _normalizar_conta(out[COL_CONTA_SAIDA])

    # D) Separa pendências (com o motivo)
    sem_conta = out[COL_CONTA_SAIDA].isna()
    sem_valor = out[COL_VALOR_SAIDA].isna()
    invalido = sem_conta | sem_valor

    pend = out[invalido].copy()
    pend[COL_MOTIVO_SAIDA] = np.select(
        [sem_conta[invalido].to_numpy() & sem_valor[invalido].to_numpy(),
         sem_conta[invalido].to_numpy()],
        ["Sem conta identificada e sem valor válido",
         "Conta não identificada (nenhuma conta preenchida ou DC diverge da conta)"],
        default="Valor ausente ou não numérico",
    )

    return out[~invalido].reset_index(drop=True), pend.reset_index(drop=True)


def main(caminho_entrada: str, caminho_saida: str):
    df = pd.read_excel(r"ARQUIVOS\filial-data-6.xlsx")
    ok, pend = segregar_debito_credito(df)

    print(ok.to_string(index=False))

    n_pd = int((ok[COL_ORIGEM_SAIDA] == ORIGEM_PARTIDA_DOBRADA).sum())
    print(f"\nLinhas geradas: {len(ok)} (partida dobrada: {n_pd} | simples: {len(ok) - n_pd})"
          f" | Pendências: {len(pend)}")
    print("Soma de W_VALOR geral (deve ser 0):", round(ok[COL_VALOR_SAIDA].sum(), 2))

    with pd.ExcelWriter(caminho_saida) as xw:
        ok.to_excel(xw, sheet_name="Segregado", index=False)
        if len(pend):
            pend.to_excel(xw, sheet_name="Pendencias", index=False)


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else r"ARQUIVOS\filial-data-6.xlsx"
    saida = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\user\Desktop\Python\ARQUIVOS\resultado_segregado.xlsx"
    main(entrada, saida)