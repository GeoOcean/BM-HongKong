#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import numpy as np
import xarray as xr
from sklearn import linear_model


def simple_multivariate_regression_model(xds_PCA, xds_VARS, name_vars):
    '''
    Regression model between daily predictor and predictand
    PCA and VARS input data have to share time dimension values.

    xds_PCA    - PREDICTOR Principal Component Analysis  (xarray.Dataset)
                 (n_components, n_components) PCs
                 (n_components, n_features) EOFs
                 (n_components, ) variance

    xds_VARS   - predictand data variables  (xarray.Dataset)
                 dataset dimension: time
                 dataset variables: name_vars

    name_vars  - will be used as predictand  (ex: ['hs','t02'])

    returns regression for each variable indicated in name_vars
    '''

    # 95% repres
    repres = 0.951

    # PREDICTOR: PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = xds_PCA['EOFs'].values[:]
    PCs = xds_PCA['PCs'].values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm-1]
    EOFsub = EOFs[:nterm-1, :]

    PCsub_std = np.std(PCsub, axis=0)
    PCsub_norm = np.divide(PCsub, PCsub_std)

    X = PCsub_norm  # predictor

    # PREDICTAND: variables data
    wd = np.array([xds_VARS[vn].values[:] for vn in name_vars]).T
    wd_std = np.nanstd(wd, axis=0)
    wd_norm = np.divide(wd, wd_std)

    Y = wd_norm  # predictand

    # Adjust
    [n, d] = Y.shape
    X = np.concatenate((np.ones((n,1)), X), axis=1)

    clf = linear_model.LinearRegression(fit_intercept=True)
    Ymod = np.zeros((n,d))*np.nan
    for i in range(d):
        clf.fit(X, Y[:,i])
        beta = clf.coef_
        intercept = clf.intercept_
        Ymod[:,i] = np.ones((n,))*intercept
        for j in range(len(beta)):
            Ymod[:,i] = Ymod[:,i] + beta[j]*X[:,j]

    # de-scale
    Ym = np.multiply(Ymod, wd_std)

    # TODO: calculate errors

    return xr.Dataset(
        {
            'Ym': (('time', 'vars'), Ym),
        },
        {
            'time': xds_VARS.time,
            'vars': [vn for vn in name_vars],
        }
    )

