import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(pre_ft_data, post_ft_data, tag):
    fig, ax = plt.subplots()
    tags = ["Success Rate", "Mean Norm", "Mean Weighted  Norm"]
    x = np.arange(len(pre_ft_data))
    width = 0.4
    ax.set_title(tag)
    b1 = plt.bar(x - width/2, pre_ft_data, width=width, edgecolor='black', label="Before Finetuning")
    b2 = plt.bar(x + width/2, post_ft_data, width=width, edgecolor='black', label="After Finetuning")
    ax.bar_label(b1)
    ax.bar_label(b2)
    plt.xticks(x, tags)
    plt.legend()

