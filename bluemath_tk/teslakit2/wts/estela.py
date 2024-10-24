#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op

# pip
import xarray as xr
import numpy as np
from matplotlib import path

# tk
from ..toolkit.kma import kma_regression_guided
from ..toolkit.linear_regress import simple_multivariate_regression_model as SMRM
from ..plotting.pcs import Plot_WT_PCs_3D


def spatial_gradient(xdset, var_name):
    '''
    Calculate spatial gradient

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variable "var_name_gradient"
    '''

    # TODO:check/ ADD ONE ROW/COL EACH SIDE
    var_grad = np.zeros(xdset[var_name].shape)

    Mx = len(xdset.longitude)
    My = len(xdset.latitude)
    lat = xdset.latitude.values

    for it in range(len(xdset.time)):
        var_val = xdset[var_name].isel(time=it).values

        # calculate gradient (matrix)
        m_c = var_val[1:-1,1:-1]
        m_l = np.roll(var_val, -1, axis=1)[1:-1,1:-1]
        m_r = np.roll(var_val, +1, axis=1)[1:-1,1:-1]
        m_u = np.roll(var_val, -1, axis=0)[1:-1,1:-1]
        m_d = np.roll(var_val, +1, axis=0)[1:-1,1:-1]
        m_phi = np.pi*np.abs(lat)/180.0
        m_phi = m_phi[1:-1]

        dpx1 = (m_c - m_l)/np.cos(m_phi[:,None])
        dpx2 = (m_r - m_c)/np.cos(m_phi[:,None])
        dpy1 = m_c - m_d
        dpy2 = m_u - m_c

        vg = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2
        var_grad[it, 1:-1, 1:-1] = vg

        # calculate gradient (for). old code
        #for i in range(1, Mx-1):
        #    for j in range(1, My-1):
        #        phi = np.pi*np.abs(lat[j])/180.0
        #        dpx1 = (var_val[j,i]   - var_val[j,i-1]) / np.cos(phi)
        #        dpx2 = (var_val[j,i+1] - var_val[j,i])   / np.cos(phi)
        #        dpy1 = (var_val[j,i]   - var_val[j-1,i])
        #        dpy2 = (var_val[j+1,i] - var_val[j,i])
        #        var_grad[it, j, i] = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2

    # store gradient
    xdset['{0}_gradient'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grad)

    return xdset

def mask_from_poly(xdset, ls_poly, name_mask='mask'):
    '''
    Generate mask from list of tuples (lon, lat)

    xdset dimensions:
        (longitude, latitude, )

    returns xdset with new variable "mask"
    '''

    lon = xdset.longitude.values
    lat = xdset.latitude.values
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mask = np.zeros(mesh_lat.shape)

    mesh_points = np.array(
        [mesh_lon.flatten(), mesh_lat.flatten()]
    ).T

    for pol in ls_poly:
        p = path.Path(pol)
        inside = np.array(p.contains_points(mesh_points))
        inmesh = np.reshape(inside, mask.shape)
        mask[inmesh] = 1

    xdset[name_mask]=(('latitude','longitude'), mask.T)

    return xdset

def dynamic_estela_predictor(xdset, var_name, estela_D):
    '''
    Generate dynamic predictor using estela

    xdset:
        (time, latitude, longitude), var_name, mask

    returns similar xarray.Dataset with variables:
        (time, latitude, longitude), var_name_comp
        (time, latitude, longitude), var_name_gradient_comp
    '''

    # first day is estela max
    first_day = int(np.floor(np.nanmax(estela_D)))+1

    # output will start at time=first_day
    shp = xdset[var_name].shape
    comp_shape = (shp[0]-first_day, shp[1], shp[2])
    var_comp = np.ones(comp_shape) * np.nan
    var_grd_comp = np.ones(comp_shape) * np.nan

    # get data using estela for each cell
    for i_lat in range(len(xdset.latitude)):
        for i_lon in range(len(xdset.longitude)):
            ed = estela_D[i_lat, i_lon]
            if not np.isnan(ed):

                # mount estela displaced time array
                i_times = np.arange(
                    first_day, len(xdset.time)
                ) - np.int(ed)

                # select data from displaced time array positions
                xdselec = xdset.isel(
                    time = i_times,
                    latitude = i_lat,
                    longitude = i_lon)

                # get estela predictor values
                var_comp[:, i_lat, i_lon] = xdselec[var_name].values
                var_grd_comp[:, i_lat, i_lon] = xdselec['{0}_gradient'.format(var_name)].values

    # return generated estela predictor
    return xr.Dataset(
        {
            '{0}_comp'.format(var_name):(
                ('time','latitude','longitude'), var_comp),
            '{0}_gradient_comp'.format(var_name):(
                ('time','latitude','longitude'), var_grd_comp),

        },
        coords = {
            'time':xdset.time.values[first_day:],
            'latitude':xdset.latitude.values,
            'longitude':xdset.longitude.values,
        }
    )




def Calc_KMA_regressionguided(PCA,
                    num_clusters, xds_waves, waves_vars, alpha, repres = 0.9,min_group_size=None):
    'KMA regression guided with waves data'

    # we have to miss some days of data due to ESTELA
    tcut = PCA.pred_time.values[:]

    # calculate regresion model between predictand and predictor
    xds_waves = xds_waves.sel(time = slice(tcut[0], tcut[-1]))
    xds_Yregres = SMRM(PCA, xds_waves, waves_vars)

    # classification: KMA regresion guided
    KMA = kma_regression_guided(
        PCA, xds_Yregres,
        num_clusters, repres, alpha, min_group_size
    )

    # store time array with KMA
    KMA['time'] = (('n_components',), PCA.pred_time.values[:])

    # save data
    return KMA


def Mod_KMA_AddStorms(KMA, storm_dates, storm_categories):
    '''
    Modify KMA bmus series adding storm category (6 new groups)
    '''

    n_clusters = len(KMA.n_clusters.values[:])
    kma_dates = KMA.time.values[:]
    bmus_storms = np.copy(KMA.sorted_bmus.values[:])  # copy numpy.array

    for sd, sc in zip(storm_dates, storm_categories):
        sdr =  np.array(sd, dtype='datetime64[D]')  # round to day
        pos_date = np.where(kma_dates==sdr)[0]
        if pos_date:
            bmus_storms[pos_date[0]] = n_clusters + sc

    # copy kma and add bmus_storms
    KMA['sorted_bmus_storms'] = (('n_components',), bmus_storms)

    return KMA


def Plot_PCs_3D(KMA, PCA, show=True):
    'Plots Predictor first 3 PCs'

    # first 3 PCs
    bmus = KMA['sorted_bmus'].values[:]
    PCs = PCA.PCs.values[:]
    variance = PCA.variance.values[:]

    n_clusters = len(KMA.n_clusters.values[:])

    PC1 = np.divide(PCs[:,0], np.sqrt(variance[0]))
    PC2 = np.divide(PCs[:,1], np.sqrt(variance[1]))
    PC3 = np.divide(PCs[:,2], np.sqrt(variance[2]))

    # dictionary of DWT PCs 123
    d_PCs = {}
    for ic in range(n_clusters):
        ind = np.where(bmus == ic)[:]

        PC123 = np.column_stack((PC1[ind], PC2[ind], PC3[ind]))
        d_PCs['{0}'.format(ic+1)] = PC123

    # Plot DWTs PCs 3D
    fig = Plot_WT_PCs_3D(d_PCs, n_clusters, show=show)

    return fig

