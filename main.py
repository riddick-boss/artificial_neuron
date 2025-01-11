import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from plot import plot_decision_regions

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None, encoding='utf-8')
    df.dropna(how="any", inplace=True)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    x = df.iloc[0:100, [0, 2]].values

    x_setosa = x[y == -1]
    x_versicolor = x[y == 1]
    plt.figure(figsize=(6, 4))
    plt.scatter(x_setosa[:, 0], x_setosa[:, 1], color='red', marker='o', label='Setosa (-1)')
    plt.scatter(x_versicolor[:, 0], x_versicolor[:, 1], color='blue', marker='x', label='Versicolor (1)')
    plt.xlabel('Dlugosc dzialki [cm]')
    plt.ylabel('Dlugosc platka [cm]')
    plt.title('Wizualizacja zbioru danych')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    #Perceptron
    etas = [0.1, 0.01, 0.5]
    for eta in etas:
        perceptron = Perceptron(eta=eta, n_iter=10)
        perceptron.fit(x, y)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
        plt.xlabel('Epoka')
        plt.ylabel('Liczba bledow')
        plt.title(f'Perceptron - liczba blednych klasyfikacji (eta={eta})')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plot_decision_regions(x, y, classifier=perceptron)
        plt.title(f'Granica decyzyjna - Perceptron (eta={eta})')
        plt.xlabel('Długość działki [cm]')
        plt.ylabel('Długość płatka [cm]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()