import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE


def run_demo(output_path="tmp.png"):
    iris = datasets.load_iris()
    target = iris.target
    tsne = TSNE(learning_rate=1000, init="random", random_state=0)
    transformed = tsne.fit_transform(iris.data)

    fig = plt.figure()
    colors = ("r", "g", "b")
    for index, label in enumerate(np.unique(target)):
        plt.scatter(
            transformed[target == label, 0],
            transformed[target == label, 1],
            label=label,
            alpha=0.8,
            color=colors[index % len(colors)],
        )
    plt.legend(loc="upper right")
    fig.savefig(output_path, dpi=200)
    return transformed


if __name__ == "__main__":
    run_demo()
