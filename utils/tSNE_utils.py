from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}
TWENTY_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
}
SIXFIVE_COLD_COLORS = {
    1: "#87CEFA",
    2: "#6495ED",
    3: "#1E90FF",
    4: "#4682B4",
    5: "#00BFFF",
    6: "#4169E1",
    7: "#0000CD",
    8: "#00008B",
    9: "#000080",
    10: "#191970",
    11: "#007BA7",
    12: "#008080",
    13: "#5F9EA0",
    14: "#40E0D0",
    15: "#E0FFFF",
    16: "#48D1CC",
    17: "#00CED1",
    18: "#AFEEEE",
    19: "#00FFFF",
    20: "#00FFFF",
    21: "#B0C4DE",
    22: "#B0E0E6",
    23: "#87CEEB",
    24: "#ADD8E6",
    25: "#66CDAA",
    26: "#7FFFD4",
    27: "#00FA9A",
    28: "#F5FFFA",
    29: "#00FF7F",
    30: "#3CB371",
    31: "#2E8B57",
    32: "#F0FFF0",
    33: "#98FB98",
    34: "#90EE90",
    35: "#32CD32",
    36: "#00FF00",
    37: "#228B22",
    38: "#008000",
    39: "#006400",
    40: "#ADFF2F",
    41: "#7FFF00",
    42: "#7CFC00",
    43: "#6B8E23",
    44: "#F5F5DC",
    45: "#FAFAD2",
    46: "#EEE8AA",
    47: "#F0E68C",
    48: "#9ACD32",
    49: "#FFD700",
    50: "#BDB76B",
    51: "#DAA520",
    52: "#B8860B",
    53: "#808000",
    54: "#ADFF2F",
    55: "#98FB98",
    56: "#8FBC8F",
    57: "#20B2AA",
    58: "#3CB371",
    59: "#2E8B57",
    60: "#5F9EA0",
    61: "#008B8B",
    62: "#008080",
    63: "#E0FFFF",
    64: "#B0E0E6",
    65: "#ADD8E6",
}
OfficeHome_20COLORS = {
    0: "#FF0000", 1: "#FF1919", 2: "#FF3333", 3: "#FF4C4C", 4: "#FF6666",
    5: "#FF7F7F", 6: "#FF9999", 7: "#FFB2B2", 8: "#FFCCCC", 9: "#FFE5E5",
    10: "#0000FF", 11: "#1919FF", 12: "#3333FF", 13: "#4C4CFF", 14: "#6666FF",
    15: "#7F7FFF", 16: "#9999FF", 17: "#B2B2FF", 18: "#CCCCFF", 19: "#E5E5FF",
}
OfficeHome_COLORS = {
    0: "#FF0000", 1: "#FF1919", 2: "#FF3333", 3: "#FF4C4C", 4: "#FF6666",
    5: "#FF7F7F", 6: "#FF9999", 7: "#FFB2B2", 8: "#FFCCCC", 9: "#FFE5E5",
    10: "#FF0000", 11: "#FF1919", 12: "#FF3333", 13: "#FF4C4C", 14: "#FF6666",
    15: "#FF7F7F", 16: "#FF9999", 17: "#FFB2B2", 18: "#FFCCCC", 19: "#FFE5E5",
    20: "#FF0000", 21: "#FF1919", 22: "#FF3333", 23: "#FF4C4C", 24: "#FF6666",
    25: "#FF7F7F", 26: "#FF9999", 27: "#FFB2B2", 28: "#FFCCCC", 29: "#FFE5E5",
    30: "#FF0000", 31: "#FF1919", 32: "#FF3333", 33: "#FF4C4C", 34: "#FF6666",
    35: "#FF7F7F", 36: "#FF9999", 37: "#FFB2B2", 38: "#FFCCCC", 39: "#FFE5E5",
    40: "#FF0000", 41: "#FF1919", 42: "#FF3333", 43: "#FF4C4C", 44: "#FF6666",
    45: "#FF7F7F", 46: "#FF9999", 47: "#FFB2B2", 48: "#FFCCCC", 49: "#FFE5E5",
    50: "#FF0000", 51: "#FF1919", 52: "#FF3333", 53: "#FF4C4C", 54: "#FF6666",
    55: "#FF7F7F", 56: "#FF9999", 57: "#FFB2B2", 58: "#FFCCCC", 59: "#FFE5E5",
    60: "#FF0000", 61: "#FF1919", 62: "#FF3333", 63: "#FF4C4C", 64: "#FF6666",
    
    65: "#0000FF", 66: "#1919FF", 67: "#3333FF", 68: "#4C4CFF", 69: "#6666FF",
    70: "#7F7FFF", 71: "#9999FF", 72: "#B2B2FF", 73: "#CCCCFF", 74: "#E5E5FF",
    75: "#0000FF", 76: "#1919FF", 77: "#3333FF", 78: "#4C4CFF", 79: "#6666FF",
    80: "#7F7FFF", 81: "#9999FF", 82: "#B2B2FF", 83: "#CCCCFF", 84: "#E5E5FF",
    85: "#0000FF", 86: "#1919FF", 87: "#3333FF", 88: "#4C4CFF", 89: "#6666FF",
    90: "#7F7FFF", 91: "#9999FF", 92: "#B2B2FF", 93: "#CCCCFF", 94: "#E5E5FF",
    95: "#0000FF", 96: "#1919FF", 97: "#3333FF", 98: "#4C4CFF", 99: "#6666FF",
    100: "#7F7FFF", 101: "#9999FF", 102: "#B2B2FF", 103: "#CCCCFF", 104: "#E5E5FF",
    105: "#0000FF", 106: "#1919FF", 107: "#3333FF", 108: "#4C4CFF", 109: "#6666FF",
    110: "#7F7FFF", 111: "#9999FF", 112: "#B2B2FF", 113: "#CCCCFF", 114: "#E5E5FF",
    115: "#0000FF", 116: "#1919FF", 117: "#3333FF", 118: "#4C4CFF", 119: "#6666FF",
    120: "#7F7FFF", 121: "#9999FF", 122: "#B2B2FF", 123: "#CCCCFF", 124: "#E5E5FF",
    125: "#0000FF", 126: "#1919FF", 127: "#3333FF", 128: "#4C4CFF", 129: "#6666FF",
}
Office31_COLORS = {
    0: "#FF0000", 1: "#FF1919", 2: "#FF3333", 3: "#FF4C4C", 4: "#FF6666",
    5: "#FF7F7F", 6: "#FF9999", 7: "#FFB2B2", 8: "#FFCCCC", 9: "#FFE5E5",
    10: "#FF0000", 11: "#FF1919", 12: "#FF3333", 13: "#FF4C4C", 14: "#FF6666",
    15: "#FF7F7F", 16: "#FF9999", 17: "#FFB2B2", 18: "#FFCCCC", 19: "#FFE5E5",
    20: "#FF0000", 21: "#FF1919", 22: "#FF3333", 23: "#FF4C4C", 24: "#FF6666",
    25: "#FF7F7F", 26: "#FF9999", 27: "#FFB2B2", 28: "#FFCCCC", 29: "#FFE5E5",
    30: "#FF0000",
    31: "#0000FF",
    32: "#1919FF",
    33: "#3333FF",
    34: "#4C4CFF",
    35: "#6666FF",
    36: "#7F7FFF",
    37: "#9999FF",
    38: "#B2B2FF",
    39: "#CCCCFF",
    40: "#E5E5FF",
    41: "#0000FF",
    42: "#1919FF",
    43: "#3333FF",
    44: "#4C4CFF",
    45: "#6666FF",
    46: "#7F7FFF",
    47: "#9999FF",
    48: "#B2B2FF",
    49: "#CCCCFF",
    50: "#E5E5FF",
    51: "#0000FF",
    52: "#1919FF",
    53: "#3333FF",
    54: "#4C4CFF",
    55: "#6666FF",
    56: "#7F7FFF",
    57: "#9999FF",
    58: "#B2B2FF",
    59: "#CCCCFF",
    60: "#E5E5FF",
    61: "#0000FF",
}
VisDA_COLORS = {
    0: "#FF0000", 1: "#FF1919", 2: "#FF3333", 3: "#FF4C4C", 4: "#FF6666",
    5: "#FF7F7F", 6: "#FF9999", 7: "#FFB2B2", 8: "#FFCCCC", 9: "#FFE5E5",
    10: "#FF0000", 11: "#FF1919",
    12: "#0000FF",
    13: "#1919FF",
    14: "#3333FF",
    15: "#4C4CFF",
    16: "#6666FF",
    17: "#7F7FFF",
    18: "#9999FF",
    19: "#B2B2FF",
    20: "#CCCCFF",
    21: "#E5E5FF",
    22: "#0000FF",
    23: "#1919FF",
}
TEN_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
}
SIX_COLORS = {
    0: "#FFD700",   # 屎黄
    1: "red",
    2: "#008941",   # 绿色
    3: "#40E0D0",   # 浅蓝
    4: "#FFB6C1",   # 粉
    5: "#FF34FF",   # 紫色
}
FOUR_COLORS = {
    0: "#DC143C",   # 红
    1: "#0000CD",   # 蓝
    2: "#3CB371",   # 绿
    3: "#FFD700",   # 屎黄
}
Three_COLORS = {
    0: "#DC143C",   # 红
    1: "#0000CD",   # 蓝
    2: "#3CB371",   # 绿
}
SFDG_COLORS = {
    0: "#FFD700",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    # 7: "#FFDBE5",
    # 8: "#7A4900",
    # 9: "#0000A6",
    # 10: "#63FFAC",
    # 11: "#B79762",
    # 12: "#004D43",
    # 13: "#8FB0FF",
    # 14: "#997D87",
    # 15: "#5A0007",
    # 16: "#809693",
    # 17: "#FEFFE6",
    # 18: "#1B4400",
    # 19: "#4FC601",
    # 20: "#3B5DFF",
    # 21: "#4A3B53",
    # 22: "#FF2F80",
    # 23: "#61615A",
    # 24: "#BA0900",
    # 25: "#6B7900",
    # 26: "#00C2A0",
    # 27: "#FFAA92",
    # 28: "#FF90C9",
    # 29: "#B903AA",
    # 30: "#D16100",
    # 31: "#DDEFFF",
    # 32: "#000035",
    # 33: "#7B4F4B",
    # 34: "#A1C299",
    # 35: "#300018",
    # 36: "#0AA6D8",
    # 37: "#013349",
    # 38: "#00846F",
    # 39: "#372101",
    # 40: "#A2CFFE",
    # 41: "#FFDF00",
    # 42: "#528B8B",
    # 43: "#FFD700",
    # 44: "#FF00FF",
    # 45: "#00FF00",
    # 46: "#00FFFF",
    # 47: "#800000",
    # 48: "#000080",
    # 49: "#800080",
    # 50: "#FF4500",
    # 51: "#2E8B57",
    # 52: "#DA70D6",
    # 53: "#EEE8AA",
    # 54: "#98FB98",
    # 55: "#AFEEEE",
    # 56: "#DB7093",
    # 57: "#FFE4E1",
    # 58: "#FFD700",
    # 59: "#32CD32",
    # 60: "#87CEEB",
    # 61: "#FF6347",
    # 62: "#4682B4",
    # 63: "#D2B48C",
    # 64: "#6A5ACD"
}


def calculate_cpm(x, axis=1):
    """Calculate counts-per-million on data where the rows are genes.
    Parameters
    ----------
    x : array_like
    axis : int
        Axis accross which to compute CPM. 0 for genes being in rows and 1 for
        genes in columns.
    """
    normalization = np.sum(x, axis=axis)
    # On sparse matrices, the sum will be 2d. We want a 1d array
    normalization = np.squeeze(np.asarray(normalization))
    # Straight up division is not an option since this will form a full dense
    # matrix if `x` is sparse. Divison can be expressed as the dot product with
    # a reciprocal diagonal matrix
    normalization = sp.diags(1 / normalization, offsets=0)
    if axis == 0:
        cpm_counts = np.dot(x, normalization)
    elif axis == 1:
        cpm_counts = np.dot(normalization, x)
    return cpm_counts * 1e6


def log_normalize(data):
    """Perform log transform log(x + 1).
    Parameters
    ----------
    data : array_like
    """
    if sp.issparse(data):
        data = data.copy()
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def select_genes(
    data,
    threshold=0,
    atleast=10,
    yoffset=0.02,
    xoffset=5,
    decay=1,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
):
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=False,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    name=None,
    fig_size_set=None,
    **kwargs
):
    import matplotlib
    import matplotlib.pyplot as plt

    if ax is None:
        if fig_size_set == None:
            fig_size = (15, 10)
        else:
            fig_size = fig_size_set
        fig, ax = plt.subplots(figsize=fig_size)

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.7), "s": kwargs.get("s", 40)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    label_name = ["art", "cartoon", "photo", "sketch"]
    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=label_name[int(yi)],
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=15,)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
        
    fig.savefig('output_tsne/{}.png'.format(name))


def evaluate_embedding(
    embedding, labels, projection_embedding=None, projection_labels=None, sample=None
):
    """Evaluate the embedding using Moran's I index.
    Parameters
    ----------
    embedding: np.ndarray
        The data embedding.
    labels: np.ndarray
        A 1d numpy array containing the labels of each point.
    projection_embedding: Optional[np.ndarray]
        If this is given, the score will relate to how well the projection fits
        the embedding.
    projection_labels: Optional[np.ndarray]
        A 1d numpy array containing the labels of each projection point.
    sample: Optional[int]
        If this is specified, the score will be computed on a sample of points.
    Returns
    -------
    float
        Moran's I index.
    """
    has_projection = projection_embedding is not None
    if projection_embedding is None:
        projection_embedding = embedding
        if projection_labels is not None:
            raise ValueError(
                "If `projection_embedding` is None then `projection_labels make no sense`"
            )
        projection_labels = labels

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The shape of the embedding and labels don't match")

    if projection_embedding.shape[0] != projection_labels.shape[0]:
        raise ValueError("The shape of the reference embedding and labels don't match")

    if sample is not None:
        n_samples = embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        embedding = embedding[sample_indices]
        labels = labels[sample_indices]

        n_samples = projection_embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        projection_embedding = projection_embedding[sample_indices]
        projection_labels = projection_labels[sample_indices]

    weights = projection_labels[:, None] == labels
    if not has_projection:
        np.fill_diagonal(weights, 0)

    mu = np.asarray(embedding.mean(axis=0)).ravel()

    numerator = np.sum(weights * ((projection_embedding - mu) @ (embedding - mu).T))
    denominator = np.sum((projection_embedding - mu) ** 2)

    return projection_embedding.shape[0] / np.sum(weights) * numerator / denominator


def plot_star(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=False,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    label_name=None,
    name=None,
    fig_size_set=None,
    star_length=1,
    **kwargs
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    SFDG_COLORS_50 = {key: mcolors.to_rgba(colors, alpha=1) for key, colors in SFDG_COLORS.items()}
    colors.update({key + len(SFDG_COLORS): color for key, color in SFDG_COLORS_50.items()})

    if ax is None:
        if fig_size_set == None:
            fig_size = (15, 10)
        else:
            fig_size = fig_size_set
        fig, ax = plt.subplots(figsize=fig_size)

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.5), "s": kwargs.get("s", 100)}
    star_plot_params = {"alpha": kwargs.get("alpha", 1.0), "s": kwargs.get("s", 500), 'edgecolors': 'black'}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    point_num = x.shape[0]
    old_num = (point_num - star_length) // 2
    ax.scatter(x[:old_num, 0], x[:old_num, 1], c=point_colors[:old_num], rasterized=True, **plot_params)
    ax.scatter(x[old_num:-star_length, 0], x[old_num:-star_length, 1], c=point_colors[old_num:-star_length], marker='^', rasterized=True, **plot_params)
    ax.scatter(x[-star_length:, 0], x[-star_length:, 1], c=point_colors[-star_length:], marker='*', rasterized=True, **star_plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=label_name[int(yi)],
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=15,)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
        
    fig.savefig('output_tsne/{}.png'.format(name))