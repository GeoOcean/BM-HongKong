#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
from datetime import datetime

# pip
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA


from .averages import RunnningMean_Monthly



def PCA_LatitudeAverage(xds, var_name, y1, y2, m1, m2):
    '''
    Principal component analysis
    method: remove monthly running mean and latitude average

    xds:
        (longitude, latitude, time), pred_name | pred_name_runavg

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # calculate monthly running mean
    xds = RunnningMean_Monthly(xds, var_name)

    # predictor variable and variable_runnavg from dataset
    var_val = xds[var_name]
    var_val_ra = xds['{0}_runavg'.format(var_name)]

    # use datetime for indexing
    dt1 = datetime(y1, m1, 1)
    dt2 = datetime(y2+1, m2, 28)
    time_PCA = [datetime(y, m1, 1) for y in range(y1, y2+1)]

    # use data inside timeframe
    data_ss = var_val.loc[:,:,dt1:dt2]
    data_ss_ra = var_val_ra.loc[:,:,dt1:dt2]

    # anomalies: remove the monthly running mean
    data_anom = data_ss - data_ss_ra

    # average across all latitudes
    data_avg_lat = data_anom.mean(dim='latitude')

    # collapse 12 months of data to a single vector
    nlon = data_avg_lat.longitude.shape[0]
    ntime = data_avg_lat.time.shape[0]
    hovmoller = xr.DataArray(
        np.reshape(data_avg_lat.values, (12*nlon, ntime//12), order='F')
    )
    hovmoller = hovmoller.transpose()

    # mean and standard deviation
    var_anom_mean = hovmoller.mean(axis=0)
    var_anom_std = hovmoller.std(axis=0)

    # remove means and normalize by the standard deviation at anomaly
    # rows = time, columns = longitude
    nk_m = np.kron(np.ones((y2-y1+1,1)), var_anom_mean)
    nk_s = np.kron(np.ones((y2-y1+1,1)), var_anom_std)
    var_anom_demean = (hovmoller - nk_m) / nk_s

    # sklearn principal components analysis
    ipca = PCA(n_components=var_anom_demean.shape[0])
    PCs = ipca.fit_transform(var_anom_demean)

    pred_lon = xds.longitude.values[:]

    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'var_anom_std': (('n_features',), var_anom_std.values),
            'var_anom_mean': (('n_features',), var_anom_mean.values),

            'time': (('n_components'), time_PCA),
            'pred_lon': (('n_lon',), pred_lon),
        },

        # store PCA algorithm metadata
        attrs = {
            'method': 'anomalies, latitude averaged',
        }
    )

def PCA_EstelaPred_slp_grd(xds, pred_name):
    '''
    Principal component analysis
    method: custom for estela predictor

    xds:
        (time, latitude, longitude), pred_name_comp | pred_name_gradient_comp

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # estela predictor and estela gradient predictor
    pred_est_var = xds['{0}_comp'.format(pred_name)]
    pred_est_grad = xds['{0}_gradient_comp'.format(pred_name)]

    # use data inside timeframe
    dp_var = pred_est_var.values
    dp_grd = pred_est_grad.values
    shape_grid = dp_var[0].shape  # needed to handle data after PCs

    # unravel and join var and grad data 
    dp_ur = np.nan * np.ones(
        (dp_var.shape[0], 2*dp_var.shape[1]*dp_var.shape[2])
    )

    # we use .T to equal matlab
    for ti in range(dp_ur.shape[0]):
        dp_ur[ti,:] = np.concatenate(
            [np.ravel(dp_var[ti].T) , np.ravel(dp_grd[ti].T)]
        )

    # remove nans from predictor    
    data_pos = ~np.isnan(dp_ur[0,:])
    clean_row = dp_ur[0, data_pos]
    dp_ur_nonan = np.nan * np.ones(
        (dp_ur.shape[0], len(clean_row))
    )
    for ti in range(dp_ur.shape[0]):
        dp_ur_nonan[ti,:] = dp_ur[ti, data_pos]

    # standarize predictor
    pred_mean = np.mean(dp_ur_nonan, axis=0)
    pred_std = np.std(dp_ur_nonan, axis=0)
    pred_norm = (dp_ur_nonan[:,:] - pred_mean) / pred_std
    pred_norm[np.isnan(pred_norm)] = 0

    # principal components analysis
    ipca = PCA(n_components=min(pred_norm.shape[0], pred_norm.shape[1]))
    PCs = ipca.fit_transform(pred_norm)

    # return dataset
    return xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'pred_mean': (('n_features',), pred_mean),
            'pred_std': (('n_features',), pred_std),

            'pred_lon': (('n_lon',), xds.longitude.values[:]),
            'pred_lat': (('n_lat',), xds.latitude.values[:]),
            'pred_time': (('time',), xds.time.values[:]),
            'pred_data_pos':(('n_points',), data_pos)
        },

        attrs = {
            'method': 'gradient + estela',
            'pred_name': pred_name,
        }
    )



def PCA_EstelaPred_sst(xds, pred_name):
    '''
    Principal component analysis
    method: custom for estela predictor

    xds:
        (time, latitude, longitude), pred_name_comp | pred_name_gradient_comp

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # estela predictor and estela gradient predictor
    pred_est_var = xds[pred_name]

    # use data inside timeframe
    dp_var = pred_est_var.values

    # unravel and join var and grad data
    dp_ur = np.nan * np.ones(
        (dp_var.shape[0], dp_var.shape[1]*dp_var.shape[2])
    )

    # we use .T to equal matlab
    for ti in range(dp_ur.shape[0]):
        dp_ur[ti,:] = np.ravel(dp_var[ti].T)

    # remove nans from predictor
    data_pos = ~np.isnan(dp_ur[0,:])
    clean_row = dp_ur[0, data_pos]
    dp_ur_nonan = np.nan * np.ones(
        (dp_ur.shape[0], len(clean_row))
    )
    for ti in range(dp_ur.shape[0]):
        dp_ur_nonan[ti,:] = dp_ur[ti, data_pos]

    # standarize predictor
    pred_mean = np.mean(dp_ur_nonan, axis=0)
    pred_std = np.std(dp_ur_nonan, axis=0)
    pred_norm = (dp_ur_nonan[:,:] - pred_mean) / pred_std
    pred_norm[np.isnan(pred_norm)] = 0

    # principal components analysis
    ipca = PCA(n_components=min(pred_norm.shape[0], pred_norm.shape[1]))
    PCs = ipca.fit_transform(pred_norm)

    # return dataset
    return xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'pred_mean': (('n_features',), pred_mean),
            'pred_std': (('n_features',), pred_std),

            'pred_lon': (('n_lon',), xds.longitude.values[:]),
            'pred_lat': (('n_lat',), xds.latitude.values[:]),
            'pred_time': (('time',), xds.time.values[:]),
            'pred_data_pos':(('n_points',), data_pos)
        },

        attrs = {
            'method': 'sst + estela',
            'pred_name': pred_name,
        }
    )

def PCA_EstelaPred_slp_grd_geo(xds, pred_name, pred_name_2):
    '''
    Principal component analysis
    method: custom for estela predictor

    xds:
        (time, latitude, longitude), pred_name_comp | pred_name_gradient_comp

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # slp , slp gradient and precipitation predictors
    pred_est_var = xds['{0}_comp'.format(pred_name)]
    pred_est_grad = xds['{0}_gradient_comp'.format(pred_name)]
    pred_est_precip = xds['{0}'.format(pred_name_2)]


    # use data inside timeframe
    dp_var = pred_est_var.values
    dp_grd = pred_est_grad.values
    dp_precip = pred_est_precip.values

    shape_grid = dp_var[0].shape  # needed to handle data after PCs

    # Nan Matrix: unravel and join var, grad & precipitation data
    dp_ur = np.nan * np.ones(
        (dp_var.shape[0], 3*dp_var.shape[1]*dp_var.shape[2])
    )

    # we use .T to equal matlab for var and grd predictors
    for ti in range(dp_ur.shape[0]):
        dp_ur[ti,:] = np.concatenate(
            [np.ravel(dp_var[ti].T) , np.ravel(dp_grd[ti].T), np.ravel(dp_precip[ti].T)]
        )


    # remove nans from slp predictors    
    data_pos = ~np.isnan(dp_ur[0,:])
    clean_row = dp_ur[0, data_pos]
    dp_ur_nonan = np.nan * np.ones(
        (dp_ur.shape[0], len(clean_row))
    )
    for ti in range(dp_ur.shape[0]):
        dp_ur_nonan[ti,:] = dp_ur[ti, data_pos]


    # standarize predictor
    pred_mean = np.mean(dp_ur_nonan, axis=0)
    pred_std = np.std(dp_ur_nonan, axis=0)
    pred_norm = (dp_ur_nonan[:,:] - pred_mean) / pred_std
    pred_norm[np.isnan(pred_norm)] = 0


    # principal components analysis
    ipca = PCA(n_components=min(pred_norm.shape[0], pred_norm.shape[1]))
    PCs = ipca.fit_transform(pred_norm)

    # return dataset
    return xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'pred_mean': (('n_features',), pred_mean),
            'pred_std': (('n_features',), pred_std),

            'pred_lon': (('n_lon',), xds.longitude.values[:]),
            'pred_lat': (('n_lat',), xds.latitude.values[:]),
            'pred_time': (('time',), xds.time.values[:]),
            'pred_data_pos':(('n_points',), data_pos)
        },

        attrs = {
            'method': 'gradient + geo',
            'pred_name': pred_name,
        }
    )
