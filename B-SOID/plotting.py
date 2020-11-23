import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import *

FIGSIZE = (12, 9)
FONTDICT = {
    'fontsize': 14
    }



def prettify_figure(ax, xticks, yticks, xlabels=None, ylabels=None, plot_xticks=True, plot_yticks=True):
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["right"].set_visible(False)   

    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    xlabels = [str(x) for x in xticks] if xlabels is None else xlabels
    ylabels = [str(y) for y in yticks] if ylabels is None else ylabels

    # tick labels
    ax.set_xticklabels(xlabels, fontdict=FONTDICT)
    ax.set_yticklabels(ylabels, fontdict=FONTDICT)

    # tick lines
    if plot_xticks:
        [ax.plot([x]*len(yticks), yticks, "--", lw=0.5, color="black", alpha=0.3) for x in xticks]
    if plot_yticks:
        [ax.plot(xticks, [y]*len(xticks), "--", lw=0.5, color="black", alpha=0.3) for y in yticks]
    
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    ax.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")  

    return ax

def plot_behaviour_usage(usage_data, savefile):
    n_bvr = usage_data.shape[1]
    mean_usg, stddev_usg = usage_data.mean(axis=0), usage_data.std(axis=0)

    fig = plt.figure(figsize=FIGSIZE)
    plt.errorbar(x=np.arange(n_bvr), y=mean_usg, yerr=stddev_usg)

    xticks = np.arange(n_bvr)
    yticks = np.linspace(0, 1.0, 10)

    for i in range(yticks.size):
        yticks[i] = round(yticks[i], 2)

    ax = fig.get_axes()
    ax = [prettify_figure(axes, xticks, yticks, plot_xticks=False) for axes in ax]

    plt.savefig(savefile)

def plot_behaviour_info(info: pd.DataFrame, behaviour_metric):
    