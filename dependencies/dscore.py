import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def mse(obs, sim=0) -> float:
    """
     Mean Square Error --   Compute MSE over all paired values observed (x) and simulated/modeled (x_hat)
        .. math::
            \sum_{i=1}^{n}(x_i - \hat{x}_i)^2

    Parameters
    ----------
    obs : array_like
        Observed values
    sim : array_like
        Simulated values

    Returns
    -------
    float : mean squared error

    NOTE: this and all functions below rely upon the obs and sim datatypes implementing
          certain math methods on themselves.  That is, obs.sum() must be defined by
          typeof(obs). Pandas Series and DataFrames do this, but other array_like
          may not.
    """
    e = obs - sim
    return pd.Series([np.array(e**2).mean()], index=['MSE'])


def bias_distribution_sequence(obs, sim, objective=mse):
    """Bias-distribution-sequence decomposition

    WARNING: This works for MSE but hasn't been generalized to other objective functions.

    Parameters
    ----------
    obs : array_like
        Observed values
    sim : array_like
        Simulated values
    objective : function
        Error function, e.g. mse, etc.

    Returns
    -------
    Series with entry for each component.

    References
    ----------
    .. [1] Hodson et al., 2021. Mean squared error, deconstructed.
    Journal of Advances in Earth Systems Modeling.
    """
    e = obs - sim
    s = np.sort(obs) - np.sort(sim)
    var_s = s.var()
    var_e = e.var()

    e_bias = objective(e)
    e_dist = var_s
    e_seq = var_e - var_s
    names = ['bias', 'distribution', 'sequence']
    return pd.Series([e_bias, e_dist, e_seq], index=names)


def bias_variability(obs, sim, objective=mse, additive=False):
    """
    Bias-variability decomposition.

    For MSE, this is equivalent to the classic bias-variance decomp.

    Parameters
    ----------
    obs : array_like
    sim : array_like
    objective : function

    Returns
    -------
    Series with entry for each component.

    """
    labels = ['bias', 'variability']
    n_components = len(labels)

    e = sim - obs
    deviations = e - e.mean()

    # mean is a hack to drop the labels and return a scalar
    bias = objective(e - deviations).mean()
    variability = objective(deviations).mean()

    result = pd.Series([bias, variability], index=labels)

    if not additive:
        result = result * n_components

    return result


def seasons(obs, sim, objective=mse):
    """Decompose error by season.

    Parameters
    ----------
    obs : array_like
    sim : array_like
    objective : function

    Returns
    -------
    Series with entry for each component.

    """

    def season(e, index):
        return objective(e * index)

    labels = ['winter', 'spring', 'summer', 'fall']
    e = sim - obs

    winter = season(e, (e.index.month == 12) | (e.index.month <= 2))
    spring = season(e, (e.index.month > 2) & (e.index.month <= 5))
    summer = season(e, (e.index.month > 5) & (e.index.month <= 8))
    fall = season(e, (e.index.month > 8) & (e.index.month <= 11))

    return pd.Series([winter, spring, summer, fall], index=labels)


def quantiles(obs, sim, objective=mse, additive=False):
    """
    Decomposes MSE by quantile rangess 0.00-0.25; 0.25-0.5; 0.5-0.75; 0.75-1.00

    Parameters
    ----------
        obs (pd.Series - like  ): series of observed values
        sim (_type_): series of simulated/modeled values
    Both share a common index

    Returns
    -------
        pd.Series : decomposed MSE, one value per quantile range
    """
    breaks = [0, 0.25, 0.5, 0.75, 1]
    labels = ['low', 'below_avg', 'above_avg', 'high']
    e = sim - obs
    scores = []
    ranks = obs.rank(method='first')
    quants = pd.qcut(ranks, q=breaks)

    for i in range(len(breaks) - 1):
        # quant = e * (quants == quants.cat.categories[i])  # select quantile
        if additive:
            quant = e * (quants == quants.cat.categories[i])  # select quantile
        else:
            quant = e[quants == quants.cat.categories[i]]

        mse_q = objective(quant).mean()
        scores.append(mse_q)
    return pd.Series(scores, index=labels)


try:
    from statsmodels.tsa.seasonal import STL

    _SEASONAL = True
except ImportError:
    import logging

    logging.debug("STL library not available.")
    _SEASONAL = False


def stl(obs, sim, period=365, objective=mse):
    """
    Decompose error using STL.

    Seasonal and trend decomposition using Loess (STL).
    Note that STL is not perfectly orthogonal.

    Parameters
    ----------
    obs : array_like
    sim : array_like
    period : int
        Length of seasonal component.
    objective : function

    Returns
    -------
    Series with entry for each component.

    References
    ----------
    .. [1] Cleveland et al., 1990, STL: A seasonal-trend decomposition
    procedure based on loess. Journal of Official Statistics, 6(1), 3-73.
    """
    if not _SEASONAL:
        logging.warning("STL statistics not available.")
        return None

    e = sim - obs
    bias = e.mean()
    res = STL(e, period=period, seasonal=9).fit()
    E = pd.DataFrame(
        {
            'bias': bias,
            'long variability': res.trend - bias,
            'seasonality': res.seasonal,
            'short variability': res.resid,
        }
    )

    scores = E.apply(objective)
    return scores.squeeze()


# Scores


def ilamb_score(e, name, a=1):
    """Scores and error.

    Exponential scoring function that maps MSE to the unit interval.

    Parameters
    ----------
    e : array_like

    a : float
        Positive tuning parameter.

    References
    ----------
    .. [1] Collier et al., 2018, The International Land Model Benchmarking
    (ILAMB) system: Design, theory, and implementation. Journal of Advances
    in Modeling Earth Systems, 10(11), http://dx.doi.org/10.1029/2018ms001354
    """
    score = np.exp(-1 * a * e)
    score = round(score * 100).astype(int)
    score.name = name

    if isinstance(score, pd.Series):
        return score.to_frame()
    else:
        return score


def percentage_score(df, name, total_col='total'):
    score = round(df / df[total_col].values * 100).astype(int)
    score.name = name

    if isinstance(score, pd.Series):
        return score.to_frame()
    else:
        return score


def gse(sigma_2, sd=1.96):
    '''Geometric Standard Error'''
    sigma_s = np.exp(np.sqrt(sigma_2))
    return sigma_s**sd


def se(sigma_2, sd=1.96):
    '''Standard Error'''
    return np.sqrt(sigma_2) * sd


# Plotting
def scorecard_plot(df, ax=None, clim=(0, 100), cmap='RdYlBu'):
    axs = ax or plt.gca()

    axs.set_ylabel("Component")
    # axs.set_yticks(np.arange(len(df.index)), labels=df.index, fontsize=6)
    axs.set_yticks(
        np.arange(len(df.index)), labels=df.index.get_level_values(1), fontsize=8
    )
    axs.yaxis.set_tick_params(length=0, which='minor')

    axs.set_xticks(
        np.arange(len(df.columns)),
        labels=df.columns,
        fontsize=8,
        rotation=90,
        ha='center',
        va='bottom',
    )
    axs.xaxis.tick_top()
    axs.xaxis.set_tick_params(length=0)
    axs.xaxis.set_label_position('top')

    x = df.index.get_level_values(0).values
    breaks = np.where(x[:-1] != x[1:])[0]
    [axs.axhline(y=i + 0.5, color='k') for i in breaks]

    im = axs.imshow(df, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=1)
    # cbar = axs.figure.colorbar(im, ax=axs, location='bottom', ticks=[0, 50, 100], ticklocation='bottom', pad=0.05, fraction=0.15, shrink=0.5)
    # cbar.ax.tick_params(labelsize=4, width=0.2, length=2, pad=1, labelbottom=True)
    # cbar.outline.set_linewidth(0.2)

    ## Annotates each cell...
    txtattrs = dict(
        ha="center",
        va="center",
        fontsize=8,
        path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.5)],
    )
    i = 0
    for col in df.columns:
        j = 0
        for row in df.index:
            text = im.axes.text(i, j, df[col][row], **txtattrs)
            j += 1
        i += 1

    ## Offset minor gridlines.... paint them white.
    axs.set_yticks(np.arange(len(df.index) + 1) - 0.5, minor=True)
    axs.set_xticks(np.arange(len(df.columns) + 1) - 0.5, minor=True)
    axs.grid(which="minor", color="w", linestyle='-', linewidth=0.75)

    return axs
