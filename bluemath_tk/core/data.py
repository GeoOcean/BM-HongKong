import numpy as np

# scatter_data
import bluemath_tk.colors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import itertools


def normalize(data, ix_directional, scale_factor={}):
    """
    Normalize data subset - norm = (val - min) / (max - min)

    Returns:
    - data_norm: Normalized data
    """

    # Initialize data.
    # data_norm = data (it is a pointer an modify the original data)
    data_norm = data.copy()

    # Normalize scalar data
    for ix in data.columns:
        if ix in ix_directional:
            data_norm[ix] = (data[ix] * np.pi / 180.0) / np.pi
        else:
            v = data[ix]
            if ix not in scale_factor:
                mi = np.amin(v)
                ma = np.amax(v)
                scale_factor[ix] = [mi, ma]

            data_norm[ix] = (v - scale_factor[ix][0]) / (
                scale_factor[ix][1] - scale_factor[ix][0]
            )

    return data_norm, scale_factor


def denormalize(data_norm, ix_directional, scale_factor):
    """
    DeNormalize data

    Returns:
    - data: De-normalized data
    """

    # Initialize data
    data = data_norm.copy()

    # Scalar data
    for ix in data.columns:
        if ix in ix_directional:
            data[ix] = data_norm[ix] * 180
        else:
            data[ix] = (
                data_norm[ix] * (scale_factor[ix][1] - scale_factor[ix][0])
                + scale_factor[ix][0]
            )

    return data


def scatter(data, centroids=None, color_data=None, custom_params=None):
    """
    Create scatter plots for all combinations of variables in the data.

    Arguments
    ---------
    data: pandas DataFrame
        Data to be plotted.

    centroids: pandas DataFrame
        Centroids to be plotted.

    color_data: array
        Array of values to color the data points.

    custom_params: dict
        Custom parameters for scatter plots.
    """

    scatter_params = (
        {**bluemath_tk.colors.scatter_defaults, **custom_params}
        if custom_params
        else bluemath_tk.colors.scatter_defaults
    )

    # Create figure and axes
    num_variables = data.shape[1]
    fig, axes = plt.subplots(
        nrows=num_variables - 1,
        ncols=num_variables - 1,
        figsize=scatter_params["figsize"],
    )

    # Create scatter plots
    combinations = list(itertools.combinations(data.columns, 2))

    i = 0
    j = num_variables - 2

    for combination in combinations:

        # If number of variables is greater than 2, create subplots
        if num_variables > 2:
            ax = axes[i, j]
        else:
            ax = axes

        if color_data is not None:
            # Define a continuous colormap using the 'rainbow' colormap from Matplotlib
            cmap_continuous = plt.cm.rainbow
            # Create a discretized colormap by sampling the continuous colormap at evenly spaced intervals
            # The number of intervals is determined by the number of unique values in 'bmus'
            cmap_discretized = ListedColormap(
                cmap_continuous(np.linspace(0, 1, len(np.unique(color_data))))
            )

            # Plot scatter data
            im = ax.scatter(
                data[combination[0]],
                data[combination[1]],
                c=color_data,
                s=scatter_params["size_data"],
                label="data",
                cmap=cmap_discretized,
                alpha=scatter_params["alpha_subset"],
            )
            plt.colorbar(im, ticks=np.arange(0, len(np.unique(color_data))))

        else:
            ax.scatter(
                data[combination[0]],
                data[combination[1]],
                s=scatter_params["size_data"],
                c=scatter_params["color_data"],
                alpha=scatter_params["alpha_data"],
                label="Data",
            )

        if centroids is not None:
            if color_data is not None:
                # Add centroids to the plot
                ax.scatter(
                    centroids[combination[0]],
                    centroids[combination[1]],
                    s=scatter_params["size_centroid"],
                    c=np.array(range(len(np.unique(color_data)))) + 1,
                    cmap=cmap_discretized,
                    ec="k",
                    label="Centroids",
                )
            else:
                ax.scatter(
                    centroids[combination[0]],
                    centroids[combination[1]],
                    s=scatter_params["size_centroid"],
                    c=scatter_params["color_data"],
                    ec="k",
                    label="Centroids",
                )

        ax.set_xlabel(combination[0], fontsize=scatter_params["fontsize"])
        ax.set_ylabel(combination[1], fontsize=scatter_params["fontsize"])
        ax.legend(fontsize=scatter_params["size_data"])
        ax.tick_params(axis="both", labelsize=scatter_params["fontsize"])

        # Update i and j for subplots
        if j > i:
            j = j - 1
        else:
            # Remove axis for empty subplots
            if j > 0:
                for empty in range(0, j):
                    ax = axes[i, empty]
                    ax.axis("off")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

            i += 1
            j = num_variables - 2

    plt.tight_layout()
    plt.show()


# def scatter_subset(self, norm=False, custom_params=None):

#     self.scatter_data(
#         norm=norm,
#         plot_centroids=True,
#         custom_params=custom_params,
#     )
#     plt.show()
