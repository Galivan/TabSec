import matplotlib.pyplot as plt
import numpy as np


def show_finetuning_metrics(pre_defense_data, post_defense_data, tag):
    """
    Show a table with metrics to compare our methods of defense.
    Data is: "Success Rate", "Perturbation Norms, "Perturbation W. Norms", "Test Loss", "Test Acc."
    :param pre_defense_data: Data before applying defense methods.
    :param post_defense_data: Data after applying defense methods.
    :param tag: T
    :return:
    """
    fig_background_color = 'white'
    fig_border = 'steelblue'
    column_headers = ["Success Rate", "Norm Mean", "Norm STD", "W. Norm Mean", "W. Norm STD", "Test Loss", "Test Acc."]
    row_headers = ['Before', 'After', 'After/Before']
    pre_defense_row = get_row(pre_defense_data)
    post_defense_row = get_row(post_defense_data)
    with np.errstate(divide='ignore',invalid='ignore'):
        scale = (post_defense_row/pre_defense_row).round(5)
    data = [pre_defense_row, post_defense_row, scale]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    plt.figure(linewidth=2,
               edgecolor=fig_border,
               facecolor=fig_background_color,
               tight_layout={'pad': 1},
               figsize=(8, 2)
               )
    the_table = plt.table(cellText=data,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')
    the_table.scale(1, 1.5)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.suptitle(tag, y=0.8)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.show()


def get_row(metrics):
    sr = metrics[0]
    mean = np.mean(metrics[1])
    std = np.std(metrics[1])
    w_mean = np.mean(metrics[2])
    w_std = np.std(metrics[2])
    test_loss = metrics[-2]
    test_acc = metrics[-1]
    row = [sr, mean, std, w_mean, w_std, test_loss, test_acc]
    return np.array([f'{x:1.5f}' for x in row], dtype=float)


