import numpy as np

class AdalineGD(object):
    """Klasyfikator — ADAptacyjny LIniowy NEuron.

    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0)
    n_iter : liczba całkowita
        Liczba przebiegów po zestawie uczącym.

    Atrybuty
    -----------
   w_ : jednowymiarowa tablica
        Wagi po dopasowaniu.
    errors_ : lista
        Liczba niewłaściwych klasyfikacji w każdej epoce.

    """
    def __init__(self, eta=0.01, n_iter=50):
        if(eta < 0.0 or eta > 1.0):
            raise ValueError(f"eta should be in range 0 - 1 (was {eta})")
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Trenowanie za pomocą danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczenia,
            gdzie n_próbek oznacza liczbę próbek, a
            n_cech – liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        """
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0]  += self.eta * errors.sum()
            y_pred = self.predict(X)
            misclassifications = np.sum(y_pred != y)
            self.errors.append(misclassifications)
        return self

    def net_input(self, X):
        """oblicza całkowite pobudzenie"""
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        """oblicza liniową funkcję aktywacji"""
        return X

    def predict(self, X):
        """zwraca etykietę klas po wykonaniu skoku jednostkowego"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)