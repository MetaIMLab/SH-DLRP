import matplotlib.pyplot as plt


def plot_roc(
    fpr,
    tpr,
    auc,
    label_names=("逻辑回归", "SVM", "神经网络", "随机森林", "决策树"),
    colors=("r", "b", "g", "m", "k"),
    linestyles=("-", "--", "-.", ":", "-"),
    zoom_ylim=(0.7, 1.01),
):
    """Plot multiple ROC curves from precomputed false/true positive rates."""
    if not (len(fpr) == len(tpr) == len(auc) == len(label_names)):
        raise ValueError("fpr, tpr, auc, and label_names must have the same length")

    fig = plt.figure(figsize=(8, 7), dpi=150)
    for index in range(len(fpr)):
        plt.plot(
            fpr[index],
            tpr[index],
            color=colors[index % len(colors)],
            linewidth=2,
            linestyle=linestyles[index % len(linestyles)],
            label="AUC={:.4f} {}".format(auc[index], label_names[index]),
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.title("不同模型的ROC曲线")

    inset_ax = fig.add_axes([0.3, 0.45, 0.4, 0.4], facecolor="white")
    for index in range(len(fpr)):
        inset_ax.plot(
            fpr[index],
            tpr[index],
            color=colors[index % len(colors)],
            linewidth=2,
            linestyle=linestyles[index % len(linestyles)],
        )
    inset_ax.set_xlim(-0.1, 1)
    inset_ax.set_ylim(*zoom_ylim)
    inset_ax.grid()
    return fig


if __name__ == "__main__":
    raise SystemExit("Provide precomputed fpr, tpr, and auc arrays and call plot_roc().")
