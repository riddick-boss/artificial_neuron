import numpy as np

class AdalineSGD(object):
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
        Liczba nieprawidłowych klasyfikacji w każdej epoce.
    shuffle : wartość boolowska (domyślnie: True)
        Jeżeli jest ustalona wartość True,
        tasuje dane uczące przed każdą epoką w celu zapobiegnięcia cykliczności.
    random_state : liczba całkowita (domyślnie: None)
        Ustanawia przypadkowy stan dla operacji tasowania
        oraz inicjacji wag.
        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        if(eta < 0.0 or eta > 1.0):
            raise ValueError(f"eta should be in range 0 - 1 (was {eta})")
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        """ Dopasowanie danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczące, gdzie n_próbek
            oznacza liczbę próbek, a
            n_cech określa liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        """
        self._initialize_weights(X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
            
            y_pred = self.predict(X)
            misclassifications = np.sum(y_pred != y)
            self.errors.append(misclassifications)
        return self

    def partial_fit(self, X, y):
        """dopasowuje dane uczące bez ponownej inicjacji wag"""
        pass #?

    def _shuffle(self, X, y):
        """tasuje dane uczące"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """inicjuje wagi przydzielając im wartości zerowe"""
        self.rgen = np.random.RandomState(self.random_state)
        self.weigths = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """wykorzystuje regułę uczenia Adaline do aktualizacji wag"""
        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = (target - output)
        self.weigths[1:] += self.eta * xi * error
        self.weigths[0] += self.eta * error
    
    def net_input(self, X):
        """oblicza całkowite pobudzenie"""
        return np.dot(X, self.weigths[1:]) + self.weigths[0]

    def activation(self, X):
        """oblicza liniową funkcję aktywacji"""
        return X

    def predict(self, X):
        """zwraca etykietę klas po wykonaniu skoku jednostkowego"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)