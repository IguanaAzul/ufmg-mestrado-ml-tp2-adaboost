from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class Stump:
    """
        Classificador fraco para utilizar no Boosting
    """
    def __init__(self):
        # Árvore de decisão de apenas um nível.
        self.decision_stump = DecisionTreeClassifier(max_depth=1)
        # Coeficiente Alpha.
        self.alpha = 0.0

    def fit(self, x, y, weights):
        self.decision_stump.fit(x, y, sample_weight=weights)

    def predict(self, x):
        return self.decision_stump.predict(x)


class AdaBoost:
    """
        Implementação do AdaBoost
    """
    def __init__(self, n_stumps):
        self.n_stumps = n_stumps
        self.stumps = list()

    def fit(self, x, y, x_test=None, y_test=None):
        # Inicialização dos pesos com valores uniformes.
        weights = np.ones(x.shape[0]) / x.shape[0]
        training_score_history = list()
        test_score_history = list()
        for _ in range(self.n_stumps):
            stump = Stump()
            # Ajuste do stump aos dados ponderados.
            stump.fit(x, y, weights)
            # Previsões do stump nos dados de treinamento.
            predictions = stump.predict(x)
            # Identificação dos exemplos classificados incorretamente.
            incorrect = predictions != y
            # Cálculo do erro ponderado cometido pelo stump.
            error = np.sum(weights[incorrect])
            # Cálculo do coeficiente alpha.
            stump.alpha = np.log((1.0 - error) / error) / 2.0
            # Atualização dos pesos com base nas previsões do stump.
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)
            # Adiciona Stump no modelo.
            self.stumps.append(stump)
            # Adiciona score de treino à lista.
            training_score_history.append(self.score(x, y))
            # Adiciona score de teste à lista, se teste tiver sido fornecido.
            if x_test is not None and y_test is not None:
                test_score_history.append(self.score(x_test, y_test))
        if test_score_history:
            return training_score_history, test_score_history
        else:
            return training_score_history

    def predict(self, x):
        # Previsões iniciais zeradas.
        predictions = np.zeros(x.shape[0])
        for stump in self.stumps:
            # Previsões do stump.
            stump_predictions = stump.predict(x)
            # Atualização das previsões com base no stump ponderado.
            predictions += stump.alpha * stump_predictions
        # Aplicação da função de sinal para obter as previsões finais.
        return np.sign(predictions)

    def score(self, x, y):
        # Cálculo da precisão das previsões em relação aos rótulos verdadeiros.
        return accuracy_score(y, self.predict(x))
