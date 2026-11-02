import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def entropia(contagens, total):
    p = contagens / np.maximum(total, 1e-12)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        termo = np.where(p > 0, p * np.log2(p), 0.0)
    return -termo.sum(axis=1)


def gini(contagens, total):
    p = contagens / np.maximum(total, 1e-12)[:, None]
    return 1.0 - (p ** 2).sum(axis=1)


class ArvoreCART:
    def __init__(self, classificacao, criterio, max_depth, min_samples_leaf, max_features, rng):
        self.classificacao = classificacao
        self.impureza = entropia if criterio == "entropy" else gini
        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng

    def fit(self, X, y, pesos=None, n_classes=None):
        self.atributo, self.limiar, self.esq, self.dir, self.valor = [], [], [], [], []
        self.X = X
        self.Y = (np.eye(n_classes)[y] * pesos[:, None]) if self.classificacao else y.astype(float)
        self._construir(np.arange(len(y)))
        for nome in ("atributo", "limiar", "esq", "dir"):
            setattr(self, nome, np.array(getattr(self, nome)))
        self.valor = np.array(self.valor)
        del self.X, self.Y
        return self

    def _construir(self, idx):
        pilha = [(idx, 0, None, None)]
        while pilha:
            idx, prof, pai, lado = pilha.pop()
            no = len(self.atributo)
            if pai is not None:
                getattr(self, lado)[pai] = no
            Y = self.Y[idx]
            self.valor.append(Y.sum(axis=0) / Y.sum() if self.classificacao else Y.mean())
            self.atributo.append(-1)
            self.limiar.append(0.0)
            self.esq.append(-1)
            self.dir.append(-1)
            if prof >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
                continue
            divisao = self._melhor_divisao(idx)
            if divisao is None:
                continue
            f, t = divisao
            self.atributo[no], self.limiar[no] = f, t
            vai_esq = self.X[idx, f] <= t
            pilha.append((idx[~vai_esq], prof + 1, no, "dir"))
            pilha.append((idx[vai_esq], prof + 1, no, "esq"))

    def _melhor_divisao(self, idx):
        n, n_atr = len(idx), self.X.shape[1]
        k = n_atr if self.max_features is None else self.max_features
        atributos = self.rng.choice(n_atr, k, replace=False)
        Y = self.Y[idx]
        if self.classificacao:
            custo_pai = Y.sum() * self.impureza(Y.sum(axis=0)[None, :], np.array([Y.sum()]))[0]
        else:
            custo_pai = ((Y - Y.mean()) ** 2).sum()
        if custo_pai <= 1e-12:
            return None
        melhor, melhor_custo = None, custo_pai - 1e-12
        m = self.min_samples_leaf
        for f in atributos:
            coluna = self.X[idx, f]
            ordem = np.argsort(coluna, kind="stable")
            xs, Ys = coluna[ordem], Y[ordem]
            validos = np.zeros(n - 1, dtype=bool)
            validos[m - 1:n - m] = True
            validos &= xs[:-1] < xs[1:]
            if not validos.any():
                continue
            if self.classificacao:
                acum = np.cumsum(Ys, axis=0)
                esq, total = acum[:-1], acum[-1]
                dir_ = total - esq
                ne, nd = esq.sum(axis=1), dir_.sum(axis=1)
                custo = ne * self.impureza(esq, ne) + nd * self.impureza(dir_, nd)
            else:
                s, s2 = np.cumsum(Ys), np.cumsum(Ys ** 2)
                ne = np.arange(1, n)
                nd = n - ne
                se, se2 = s[:-1], s2[:-1]
                sd, sd2 = s[-1] - se, s2[-1] - se2
                custo = (se2 - se ** 2 / ne) + (sd2 - sd ** 2 / nd)
            custo = np.where(validos, custo, np.inf)
            i = int(np.argmin(custo))
            if custo[i] < melhor_custo:
                melhor_custo, melhor = custo[i], (int(f), float((xs[i] + xs[i + 1]) / 2))
        return melhor

    def predict(self, X):
        no = np.zeros(len(X), dtype=int)
        ativos = self.atributo[no] >= 0
        while ativos.any():
            n = no[ativos]
            esquerda = X[ativos, self.atributo[n]] <= self.limiar[n]
            no[ativos] = np.where(esquerda, self.esq[n], self.dir[n])
            ativos = self.atributo[no] >= 0
        return self.valor[no]


class _FlorestaBase(BaseEstimator):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_leaf=1,
                 max_features=None, class_weight=None, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def _treinar(self, X, y, classificacao, pesos=None, n_classes=None):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        self.arvores_ = []
        for _ in range(self.n_estimators):
            amostra = rng.integers(0, len(X), len(X))
            arvore = ArvoreCART(classificacao, self.criterion, self.max_depth, self.min_samples_leaf,
                                self.max_features, rng)
            arvore.fit(X[amostra], y[amostra], None if pesos is None else pesos[amostra], n_classes)
            self.arvores_.append(arvore)
        return self


class RandomForestRegressorDoZero(RegressorMixin, _FlorestaBase):
    def fit(self, X, y):
        return self._treinar(X, np.asarray(y, dtype=float), classificacao=False)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([a.predict(X) for a in self.arvores_], axis=0)


class RandomForestClassifierDoZero(ClassifierMixin, _FlorestaBase):
    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        contagem = np.bincount(y_idx)
        if self.class_weight == "balanced":
            pesos = (len(y_idx) / (len(contagem) * contagem))[y_idx]
        else:
            pesos = np.ones(len(y_idx))
        return self._treinar(X, y_idx, classificacao=True, pesos=pesos, n_classes=len(contagem))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votos = np.zeros((len(X), len(self.classes_)))
        for a in self.arvores_:
            votos[np.arange(len(X)), a.predict(X).argmax(axis=1)] += 1
        return self.classes_[votos.argmax(axis=1)]
