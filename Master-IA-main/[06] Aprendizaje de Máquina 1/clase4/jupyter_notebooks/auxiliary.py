import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def plot_boundary(X, y, model, 
                  step_x=(0.1, 0.1),
                  max_x=(1, 1),
                  min_x=(-1, -1),
                  point_size=5,
                  figsize=(7, 5), 
                  label_point=("1", "0"),
                  colormap_frontier=('#7aa5fb', '#f8b389'),
                  colormap_points=('#5471ab', '#d1885c'),
                  labels_axis=("x1", "x2"),
                  legend=True,
                  legend_title=None
                  ):

    # Crear la malla de puntos para el gráfico
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() + min_x[0], stop=X[:, 0].max() + max_x[0], step=step_x[0]),
        np.arange(start=X[:, 1].min() + min_x[1], stop=X[:, 1].max() + max_x[1], step=step_x[1])
    )
    X_cont = np.array([X1.ravel(), X2.ravel()]).T

    # Crear el gráfico de contorno
    plt.figure(figsize=figsize)
    plt.contourf(
        X1, X2, model.predict(X_cont).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(colormap_frontier)
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # Graficar los puntos de entrenamiento
    for i, j in enumerate(np.unique(y)):
        plt.scatter(
            X[y == j, 0], X[y == j, 1],
            c=colormap_points[i], label=label_point[i],
            s=point_size
        )

    plt.xlabel(labels_axis[0]) 
    plt.ylabel(labels_axis[1]) 
    if legend:
        plt.legend(title=legend_title)
