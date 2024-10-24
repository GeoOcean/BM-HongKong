#!/usr/bin/env python
# -*- coding: utf-8 -*-


# pip
import numpy as np
import xarray as xr
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t, poisson
from scipy.special import ndtri  # norm inv



def ksdensity_cdf(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x) 
    fint = interp1d(kde.support, kde.cdf)

    return fint(x)

def ksdensity_icdf(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    return fint(p)

def generalizedpareto_cdf(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params 
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf

def generalizedpareto_icdf(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''

    # fit a generalized pareto and get params 
    shape, _, scale = genpareto.fit(x)

    # get percent points (inverse of CDF) 
    icdf = genpareto.ppf(p, shape, scale=scale)

    return icdf

def empirical_cdf(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf

def empirical_icdf(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: revisar que el fill_value funcione correctamente

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)

def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student

    family  - 'gaussian' / 't'  (WARNING: ONLY 'gaussian' implemented)
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student) 

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # TODO: no encaja con los datos. no funciona 
        x = np.linspace(np.min(u), np.max(u),100)
        inv_t = np.ndarray((len(x), u.shape[1]))

        for j in range(u.shape[1]):
            param = t.fit(u[:,j])
            t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
            inv_t[:,j] = t_pdf

        # TODO CORRELACION? NUHAT?
        rhohat = np.corrcoef(inv_n.T)
        nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Generates and returns random vectors from a copula

    family  - 'gaussian' / 't'  (WARNING: ONLY 'gaussian' implemented)
    rhohat  - correlation matrix
    n       - number of samples to generate
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u

def copula_simulation(data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    data     - 2D nump.array, each variable in a column
    kernels  - list of kernels for each column at data (KDE | GPareto | ECDF)
    num_sim  - number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_cdf, ksdensity_icdf),
        'GPareto' : (generalizedpareto_cdf, generalizedpareto_icdf),
        'ECDF' : (empirical_cdf, empirical_icdf),
    }

    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernel, ' | '.join(d_kf.keys())
            )
        )

    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(data.shape) * np.nan
    ic = 0
    for d, k in zip(data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula 
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim


def MultivariatePoisson(p_out, dwt, x_myd, dirs, numberPoisson, n_clusters=42, ):
    import os


    for dd in range(n_clusters):

        x=x_myd[np.where(dwt==dd)[0],:]

        vals=np.full([numberPoisson,len(dirs)],np.nan)

        # for estimating the covariance matrix rows should be the different variables (directions)
        # and columns observations (days)
        x1 = x.T

        COV = np.cov(x1)
        COV = COV.tolist()
        rates = np.nanmean(x1, axis=1)

        mean = np.full([len(dirs)],0)
        mean = mean.tolist()

        xxr = np.random.multivariate_normal(mean, COV, numberPoisson, check_valid='ignore')
        p = norm.cdf(xxr)

        for i in range(numberPoisson):
            for j in range(len(dirs)):
                if poisson.ppf(p[i,j], rates[j]) != np.inf:

                    vals[i,j] = poisson.ppf(p[i,j], rates[j])


        num_mpoisson = xr.Dataset({'Sim_nun': (['num','dirs'],vals)},
                                  #coords={'dirs': dirs[:-1]+5,
                                  coords={'dirs': dirs,
                                          'num':np.arange(numberPoisson)})

        num_mpoisson.to_netcdf(path=os.path.join(p_out, 'MultivariatePoisson_DWT_' + str(dd) + '_nrnd' + str(numberPoisson) + '.nc'))



def UnivariatePoisson(file_out, dwt, x_myd, dirs, numberPoisson, n_clusters=42, ):

    vals=np.full([numberPoisson,n_clusters, len(dirs)],np.nan)
    medias = []

    for dd in range(n_clusters):

        x = x_myd[np.where(dwt==dd)[0],:]
        mu = np.mean(x, axis=0)
        medias.append(mu)

        for j in range(len(dirs)):
            vals[:,dd,j] = poisson.rvs(mu[j], size=numberPoisson)


    # save
    num_poisson = xr.Dataset({'Sim_num': (['num','wt','dirs'],vals)},
                                  coords={'dirs': dirs,
                                          'wt': np.arange(n_clusters),
                                          'num':np.arange(numberPoisson)})

    num_poisson.to_netcdf(file_out)

    return num_poisson
