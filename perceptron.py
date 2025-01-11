import numpy as np


class Perceptron(object):
    """Klasyfikator - perceptron.

    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (w przedziale pomiędzy 0.0 a 1.0)
    n_iter : liczba całkowita
        Liczba przebiegów po zestawach uczących.

    Atrybuty
    -----------
    w_ : jednowymiarowa tablica
        Wagi po dopasowaniu.
    errors_ : lista
        Liczba nieprawidłowych klasyfikacji w każdej epoce.

    """
    def __init__(self, eta=0.01, n_iter=10):
        if(eta < 0.0 or eta > 1.0):
            raise ValueError(f"eta should be in range 0 - 1 (was {eta})")
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Dopasowanie danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczące, gdzie n_próbek
            oznacza liczbę próbek, a
            n_cech — liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        UWAGA: wykorzystuje metodę predict

        """
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += 1 if update != 0.0 else 0
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Oblicza całkowite pobudzenie sieci
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        """Zwraca etykietę klas po obliczeniu funkcji skoku jednostkowego
        UWAGA: Wykorzystuje metodę net_input
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)