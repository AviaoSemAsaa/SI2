import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from treino import SAIDA, gravidade_para_classe


def ler_entradas(caminho):
    df = pd.read_csv(caminho, header=None, skipinitialspace=True)
    if df.shape[1] == 4:
        return df[0].astype(int).to_numpy(), df[[1, 2, 3]].to_numpy()
    return df[0].astype(int).to_numpy(), df[[3, 4, 5]].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arquivo")
    ap.add_argument("--classe", choices=["gravidade", "direta"], default="gravidade")
    args = ap.parse_args()
    ids, X = ler_entradas(args.arquivo)

    for nome in ("rf", "mlp"):
        m = joblib.load(SAIDA / "modelos" / f"{nome}.joblib")
        grav = np.clip(m["regressor"].predict(X), 0, 100)
        classe = gravidade_para_classe(grav, m["limiares"]) if args.classe == "gravidade" \
            else m["classificador"].predict(X)
        destino = SAIDA / f"predicoes_{Path(args.arquivo).stem}_{nome}.txt"
        with open(destino, "w") as f:
            for i, g, c in zip(ids, grav, classe):
                f.write(f"{i}, {g:.4f}, {c}\n")
        print(f"{nome.upper()}: {len(ids)} predições -> {destino}")


if __name__ == "__main__":
    main()
