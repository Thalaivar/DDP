import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (12, 9)

def prettify_figure(ax):
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)   

    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    return ax

def plot_behaviour_usage(usage_data):
    n_bvr = usage_data.shape[1]

    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)

    ax = prettify_figure(ax)

    