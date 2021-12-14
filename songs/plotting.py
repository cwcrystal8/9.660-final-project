import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def plot_mds(songs_to_mds_coords, genre_label, ellipse_sets):
    for s in songs_to_mds_coords:
        x, y = songs_to_mds_coords[s][0],songs_to_mds_coords[s][1]
        plt.scatter([x],[y])
        plt.annotate(s, (x,y), fontsize='small', wrap=True)
    colors = ['black', 'tab:red', 'tab:orange', 'tab:green']
    for i, (group, label) in enumerate(ellipse_sets):   
        xs = np.asarray([songs_to_mds_coords[song][0] for song in group])
        ys = np.asarray([songs_to_mds_coords[song][1] for song in group])
        confidence_ellipse(xs, ys, plt.gca(), 1, edgecolor=colors[i], label=label)
    plt.legend(fontsize='small')
    plt.xticks(plt.xticks()[0], [])
    plt.yticks(plt.yticks()[0], [])
    plt.title("MDS Space for {} Songs".format(genre_label))
    return songs_to_mds_coords

def plot_scatter_comparisons(human_data, bayesian_data, likelihood_data, max_sim_data, sum_sim_data, genre_label, plot=False, verbose=True):
    ax = [[None, None], [None, None]]
    if plot:
        fig, ax = plt.subplots(2, 2)
        plt.suptitle("Model Strength Comparisons for {} Songs".format(genre_label))
    models_to_r_values = {}
    models_to_r_values["Bayesian Model"] = plot_scatter_and_regression(bayesian_data, human_data, ax[0][0], "Bayesian Model", plot, verbose)
    models_to_r_values["Likelihood Model"] = plot_scatter_and_regression(likelihood_data, human_data, ax[0][1], "Likelihood Model", plot, verbose)
    models_to_r_values["Max-Similarity Model"] = plot_scatter_and_regression(max_sim_data, human_data, ax[1][0], "Max-Similarity Model", plot, verbose)
    models_to_r_values["Sum-Similarity Model"] = plot_scatter_and_regression(sum_sim_data, human_data, ax[1][1], "Sum-Similarity Model", plot, verbose)
    if verbose:
        best_model = max(models_to_r_values, key=lambda x: models_to_r_values[x])
        print("Best Model for {}: {}, {:.4f}".format(genre_label, best_model, models_to_r_values[best_model]))
    if plot:
        pass
        # plt.show()
    return models_to_r_values

def plot_scatter_and_regression(x, y, ax, xlabel, plot, verbose):
    m, b = np.polyfit(x, y, 1)
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    if verbose:
        print("R Value for {}: {}".format(xlabel, correlation_xy))
    if plot:
        ax.scatter(x, y)
        ax.plot(x, [m*x_val + b for x_val in x], label='r= {:.4}'.format(correlation_xy), color='orange')
        ax.set_xticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Human Data")
        ax.legend()
    return correlation_xy

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
