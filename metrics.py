import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(pre_defense_data, post_defense_data, tag):
    fig_background_color = 'white'
    fig_border = 'steelblue'
    column_headers = ["Success Rate", "W. Norm Mean", "W. Norm STD", "Norm Mean", "Norm STD"]
    row_headers = ['Before', 'After']
    pre_defense_row = get_row(pre_defense_data)
    post_defense_row = get_row(post_defense_data)
    data = [[f'{x:1.5f}' for x in pre_defense_row], [f'{x:1.5f}' for x in post_defense_row]]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    plt.figure(linewidth=2,
               edgecolor=fig_border,
               facecolor=fig_background_color,
               tight_layout={'pad':1},
               figsize=(8,2)
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
    the_table.set_fontsize(8)
    plt.show()



def get_row(metrics):
    success_rate, norms, weighted_norms = metrics
    norms_mean = np.mean(norms)
    norms_std = np.std(norms)
    weighted_norms_mean = np.mean(weighted_norms)
    weighted_norms_std = np.std(weighted_norms)
    return [success_rate, weighted_norms_mean, weighted_norms_std, norms_mean, norms_std]


pre_defense_data = (1, [1,2,3,4,5], [1,3,5,7,9])
post_defense_data = (0.5, [0.1, 0.2, 0.3], [1,2,3,4,5])
tag = 'Fine Tune'
plot_metrics(pre_defense_data, post_defense_data, tag)