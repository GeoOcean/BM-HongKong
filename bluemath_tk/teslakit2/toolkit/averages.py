#!/usr/bin/env python
# -*- coding: utf-8 -*-


# pip
import numpy as np
import xarray as xr
import pandas as pd

def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        print('running_mean: input data contain NaNs!!')
       #return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.nancumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


def RunnningMean_Monthly(xds, var_name, window=5):
    '''
    Calculate running average grouped by months

    xds:
        (longitude, latitude, time) variables: var_name

    returns xds with new variable "var_name_runavg"
    '''

    tempdata_runavg = np.empty(xds[var_name].shape)

    for lon in xds.longitude.values:
        for lat in xds.latitude.values:
            for mn in range(1, 13):
                
                # indexes
                ix_lon = np.where(xds.longitude == lon)
                ix_lat = np.where(xds.latitude == lat)
                ix_mnt = np.where(xds['time.month'] == mn)
                
                # point running average
                time_mnt = xds.time[ix_mnt]
                data_pnt = xds[var_name].loc[lon, lat, time_mnt]

                tempdata_runavg[ix_lon[0], ix_lat[0], ix_mnt[0]] = running_mean(
                     data_pnt.values, window)

    # store running average
    xds['{0}_runavg'.format(var_name)]= (
        ('longitude', 'latitude', 'time'),
        tempdata_runavg)

    return xds


def monthly_mean(xda, year_ini, year_end):
    '''
    Calculate monthly mean

    xda  - xarray.DataArray (time,)
    '''

    lout_mean = []
    lout_median = []
    lout_time = []
    for yy in range(year_ini, year_end+1):
        for mm in range(1,13):

            d1 = np.datetime64('{0:04d}-{1:02d}'.format(yy,mm))
            d2 = d1 + np.timedelta64(1, 'M')

            tide_sel_m = xda.where(
                (xda.time >= d1) & (xda.time <= d2),
                drop = True)[:-2]
            time_sel = tide_sel_m.time.values

            if len(time_sel) >= 300:
                # mean, median and dates
                ts_mean = tide_sel_m.mean().values
                ts_median = tide_sel_m.median().values
                ts_time = time_sel[int(len(time_sel)/2)]

                lout_mean.append(ts_mean)
                lout_median.append(ts_median)
                lout_time.append(ts_time)


    # join output in xarray.Dataset
    xds_mean = xr.Dataset(
        {
            'data_mean':(('time',), lout_mean),
            'data_median':(('time',), lout_median),
        },
        coords = {
            'time': lout_time
        }
    )

    return xds_mean


