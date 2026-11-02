import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from random_forest import RandomForestClassifierDoZero, RandomForestRegressorDoZero

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "02_treino_sinais_vitais_com_label.txt"
SAIDA = RAIZ / "resultados"
COLUNAS = ["i", "pSist", "pDiast", "qPA", "pulso", "resp", "gravidade", "classe"]
ENTRADAS = ["qPA", "pulso", "resp"]
SEMENTE = 42
CORES = {"RF": "#2a78d6", "MLP": "#eb6834"}


def carregar():
    df = pd.read_csv(DADOS, header=None, names=COLUNAS)
    return df[ENTRADAS].to_numpy(), df["gravidade"].to_numpy(), df["classe"].to_numpy()


def ajustar(nome, modelo, grade, X, y, cv, scoring):
    t0 = time.time()
    busca = GridSearchCV(modelo, grade, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    busca.fit(X, y)
    cvr = pd.DataFrame(busca.cv_results_)
    cols = [c for c in cvr.columns if c.startswith("param_")]
    cols += ["mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    cvr[cols].sort_values("rank_test_score").head(5).to_csv(SAIDA / f"ajuste_{nome}.csv", index=False)
    print(f"[{nome}] melhor={busca.best_params_} cv={busca.best_score_:.4f} ({time.time() - t0:.1f}s)")
    return busca


def metricas_reg(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p))}


def metricas_clf(y, p):
    rel = classification_report(y, p, labels=[1, 2, 3, 4], output_dict=True, zero_division=0)
    return {"acuracia": float(accuracy_score(y, p)),
            "macro": {k: float(rel["macro avg"][k]) for k in ("precision", "recall", "f1-score")},
            "por_classe": {c: {k: float(rel[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                           for c in ("1", "2", "3", "4")},
            "matriz_confusao": confusion_matrix(y, p, labels=[1, 2, 3, 4]).tolist()}


def limiares_da_gravidade(g, c):
    return [float((g[c == k].max() + g[c == k + 1].min()) / 2) for k in (1, 2, 3)]


def gravidade_para_classe(g, limiares):
    return np.digitize(g, limiares) + 1


def main():
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / "modelos").mkdir(exist_ok=True)
    X, g, c = carregar()

    X_tr, X_te, g_tr, g_te, c_tr, c_te = train_test_split(
        X, g, c, test_size=0.2, stratify=c, random_state=SEMENTE)
    kf = KFold(5, shuffle=True, random_state=SEMENTE)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)

    rf_reg = ajustar("rf_regressao", RandomForestRegressorDoZero(n_estimators=100, random_state=SEMENTE),
                     {"max_depth": [None, 10], "min_samples_leaf": [1, 3], "max_features": [1, None]},
                     X_tr, g_tr, kf, "neg_root_mean_squared_error")
    rf_clf = ajustar("rf_classificacao", RandomForestClassifierDoZero(n_estimators=100, random_state=SEMENTE),
                     {"criterion": ["gini", "entropy"], "max_depth": [None, 10], "min_samples_leaf": [1, 3],
                      "max_features": [1, None], "class_weight": [None, "balanced"]},
                     X_tr, c_tr, skf, "f1_macro")

    mlp_base = dict(max_iter=3000, early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=50, random_state=SEMENTE)
    mlp_reg = ajustar(
        "mlp_regressao",
        TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(**mlp_base)),
                                   transformer=StandardScaler()),
        {"regressor__mlpregressor__hidden_layer_sizes": [(32,), (64, 32), (64, 64, 32)],
         "regressor__mlpregressor__alpha": [1e-4, 1e-2],
         "regressor__mlpregressor__learning_rate_init": [1e-3, 1e-2]},
        X_tr, g_tr, kf, "neg_root_mean_squared_error")
    mlp_clf = ajustar(
        "mlp_classificacao", make_pipeline(StandardScaler(), MLPClassifier(**mlp_base)),
        {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32), (64, 64, 32)],
         "mlpclassifier__alpha": [1e-4, 1e-2],
         "mlpclassifier__learning_rate_init": [1e-3, 1e-2]},
        X_tr, c_tr, skf, "f1_macro")

    limiares = limiares_da_gravidade(g_tr, c_tr)
    regs = {"RF": rf_reg.best_estimator_, "MLP": mlp_reg.best_estimator_}
    clfs = {"RF": rf_clf.best_estimator_, "MLP": mlp_clf.best_estimator_}

    res = {"divisao": {"treino": len(X_tr), "teste": len(X_te),
                       "classes_teste": {str(k): int((c_te == k).sum()) for k in (1, 2, 3, 4)}},
           "limiares_gravidade": limiares, "regressao": {}, "classificacao": {},
           "classificacao_via_regressao": {}, "melhores_parametros": {}}
    pred_te = {}
    for m in ("RF", "MLP"):
        pg_tr, pg_te = regs[m].predict(X_tr), regs[m].predict(X_te)
        pred_te[m] = pg_te
        res["regressao"][m] = {"treino": metricas_reg(g_tr, pg_tr), "teste": metricas_reg(g_te, pg_te),
                               "cv_rmse": float(-(rf_reg if m == "RF" else mlp_reg).best_score_)}
        busca_c = rf_clf if m == "RF" else mlp_clf
        res["classificacao"][m] = {"treino": metricas_clf(c_tr, clfs[m].predict(X_tr)),
                                   "teste": metricas_clf(c_te, clfs[m].predict(X_te)),
                                   "cv_f1_macro": float(busca_c.best_score_)}
        res["classificacao_via_regressao"][m] = {"teste": metricas_clf(c_te, gravidade_para_classe(pg_te, limiares))}
        res["melhores_parametros"][m] = {
            "regressao": {k.split("__")[-1]: str(v) for k, v in (rf_reg if m == "RF" else mlp_reg).best_params_.items()},
            "classificacao": {k.split("__")[-1]: str(v) for k, v in busca_c.best_params_.items()}}

    res["importancia_permutacao"] = {}
    for m in ("RF", "MLP"):
        imp = permutation_importance(regs[m], X_te, g_te, n_repeats=20, random_state=SEMENTE,
                                     scoring="neg_root_mean_squared_error")
        res["importancia_permutacao"][m] = dict(zip(ENTRADAS, imp.importances_mean.round(3).tolist()))

    for m in ("RF", "MLP"):
        regs[m].fit(X, g)
        clfs[m].fit(X, c)
        joblib.dump({"regressor": regs[m], "classificador": clfs[m], "limiares": limiares, "entradas": ENTRADAS},
                    SAIDA / "modelos" / f"{m.lower()}.joblib")

    (SAIDA / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    figuras(g_te, pred_te, res, limiares)
    resumo(res)


def figuras(g_te, pred_te, res, limiares):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, m in zip(axs, ("RF", "MLP")):
        cm = np.array(res["classificacao"][m]["teste"]["matriz_confusao"])
        ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{m} (acurácia {res['classificacao'][m]['teste']['acuracia']:.3f})".replace(".", ","))
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
    fig.tight_layout()
    fig.savefig(SAIDA / "fig_matrizes_confusao.png", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, m in zip(axs, ("RF", "MLP")):
        ax.scatter(g_te, pred_te[m], s=10, color=CORES[m], alpha=0.7, linewidths=0)
        ax.plot([10, 90], [10, 90], color="#52514e", lw=1, ls="--")
        for t in limiares:
            ax.axvline(t, color="#c3c2b7", lw=0.8)
        ax.set_title(f"{m} (RMSE teste {res['regressao'][m]['teste']['RMSE']:.2f})".replace(".", ","))
        ax.set_xlabel("Gravidade real")
    axs[0].set_ylabel("Gravidade predita")
    fig.tight_layout()
    fig.savefig(SAIDA / "fig_gravidade_real_predita.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3))
    y = np.arange(len(ENTRADAS))
    for k, m in enumerate(("RF", "MLP")):
        v = [res["importancia_permutacao"][m][a] for a in ENTRADAS]
        ax.barh(y + (k - 0.5) * 0.38, v, height=0.36, color=CORES[m], label=m)
    ax.set_yticks(y, ENTRADAS)
    ax.set_xlabel("Aumento do RMSE ao embaralhar o atributo")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SAIDA / "fig_importancia.png", dpi=200)
    plt.close(fig)


def resumo(res):
    print("\n=== TESTE (hold-out 20%) ===")
    for m in ("RF", "MLP"):
        r, cl, cr = res["regressao"][m]["teste"], res["classificacao"][m]["teste"], \
            res["classificacao_via_regressao"][m]["teste"]
        print(f"{m}: RMSE={r['RMSE']:.3f} MAE={r['MAE']:.3f} R2={r['R2']:.4f} | "
              f"acc={cl['acuracia']:.3f} F1macro={cl['macro']['f1-score']:.3f} | "
              f"via regressão acc={cr['acuracia']:.3f} F1macro={cr['macro']['f1-score']:.3f}")
    print("limiares:", [round(t, 2) for t in res["limiares_gravidade"]])


if __name__ == "__main__":
    main()
