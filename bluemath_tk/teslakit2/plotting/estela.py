#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import copy
from datetime import datetime, date

# pip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# teslakit
from ..util.operations import GetBestRowsCols
from ..util.time_operations import npdt64todatetime as n2d
from ..util.time_operations import get_years_months_days
from .kma import ClusterProbabilities
from .custom_colors import colors_dwt
from .wts import axplot_WT_Probs, axplot_WT_Hist

# import constants
from .config import _faspect, _fsize, _fdpi


def add_land_mask(ax, lon, lat, land, color):
    'addsland mask pcolormesh to existing pcolormesh'

    # select land in mask
    landc = land.copy()
    landc[np.isnan(land)]=1
    landc[land==1]=np.nan

    ax.pcolormesh(
        lon, lat, landc,
        cmap=colors.ListedColormap([color]), shading='gouraud',
    )

# def axplot_EOF(ax, EOF_value, lon, lat, ttl='', land=None):
#     'axes plot EOFs 2d map'

#     cmap = cm.get_cmap('RdBu_r')

#     # EOF pcolormesh 
#     ax.pcolormesh(
#         lon, lat, np.transpose(EOF_value),
#         cmap=cmap, shading='gouraud',
#         clim=(-1,1),
#     )

#     # optional mask land
#     if type(land).__module__ == np.__name__:
#         add_land_mask(ax, lon, lat, land, 'grey')

#     # axis and title
#     ax.set_title(
#         ttl,
#         {'fontsize': 10, 'fontweight':'bold'}
#     )
#     ax.tick_params(axis='both', which='major', labelsize=8)

def axplot_EOF(ax, EOF_value, lon, lat, ttl='', land=None):
    'axes plot EOFs 2d map'

    cmap = cm.get_cmap('RdBu_r')

    # EOF pcolormesh 
    ax.contourf(
        lon, lat, np.transpose(EOF_value), 500,
        cmap=cmap, #shading='gouraud',
        clim=(-1,1), transform = ccrs.PlateCarree()
    )

    # optional mask land
    if type(land).__module__ == np.__name__:
        add_land_mask(ax, lon, lat, land, 'grey')

    # axis and title
    ax.set_title(
        ttl,
        {'fontsize': 10, 'fontweight':'bold'}
    )
    ax.tick_params(axis='both', which='major', labelsize=8)

def axplot_EOF_evolution(ax, time, EOF_evol):
    'axes plot EOFs evolution'

    # date axis locator
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # convert to datetime
    dtime = [n2d(t) for t in time]

    # plot EOF evolution 
    ax.plot(
        dtime, EOF_evol,
        linestyle='-', linewidth=0.5, color='black',
    )

    # configure axis
    ax.set_xlim(time[0], time[-1])
    ax.xaxis.set_major_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=8)

# def axplot_DWT(ax, dwt, vmin, vmax, wt_num, land=None, wt_color=None):
#     'axes plot EOFs 2d map'

#     cmap = copy.deepcopy(cm.get_cmap('RdBu_r'))

#     # EOF pcolormesh 
#     pc = ax.pcolormesh(
#         dwt,
#         cmap = cmap, shading = 'gouraud',
#         clim = (vmin, vmax),
#     )

#     # optional mask land
#     if type(land).__module__ == np.__name__:
#         landc = land.copy()
#         landc[np.isnan(land)]=1
#         landc[land==1]=np.nan
#         ax.pcolormesh(
#             np.flipud(landc),
#             cmap=colors.ListedColormap(['silver']), shading='gouraud',
#         )

#     # axis color
#     plt.setp(ax.spines.values(), color=wt_color)
#     plt.setp(
#         [ax.get_xticklines(), ax.get_yticklines()],
#         color=wt_color,
#     )
#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(2.5)

#     # wt text
#     ax.text(0.87, 0.85, wt_num, transform=ax.transAxes, fontweight='bold')

#     # customize axis
#     ax.set_xticks([])
#     ax.set_yticks([])

#     return pc

def axplot_DWT(ax,xds_var, dwt, vmin, vmax, wt_num, land=None, wt_color=None):
    'axes plot EOFs 2d map'

    cmap = copy.deepcopy(cm.get_cmap('RdBu_r'))

    # EOF pcolormesh 
    # pc = ax.pcolormesh(
    #     xds_var.longitude.values, xds_var.latitude.values, dwt,
    #     cmap = cmap, shading = 'gouraud',
    #     clim = (vmin, vmax), transform = ccrs.PlateCarree()
    # )

    pc = ax.contourf(
        xds_var.longitude.values, xds_var.latitude.values, dwt, 100,
        cmap = cmap, shading = 'gouraud',
        vmin = vmin, vmax=vmax, transform = ccrs.PlateCarree()
    )

    # axis color
    plt.setp(ax.spines.values(), color=wt_color, linewidth = 2)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=wt_color,
    )
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4.5)

    # wt text
    ax.text(0.85, 0.85, wt_num, transform=ax.transAxes, fontsize = 12, fontweight='light')

    # customize axis
    ax.set_xticks([])
    ax.set_yticks([])

    return pc


def Plot_EOFs_EstelaPred_SST(xds_PCA, n_plot, mask_land=None, show=True):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

        (n_lon, ) pred_lon: predictor longitude values
        (n_lat, ) pred_lat: predictor latitude values
        (n_time, ) pred_time: predictor time values

        method: gradient + estela

    n_plot: number of EOFs plotted
    '''

    # TODO: fix data_pos, fails only after pred.Load()?

    # PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = np.transpose(xds_PCA['EOFs'].values[:])
    PCs = np.transpose(xds_PCA['PCs'].values[:])
    data_pos = xds_PCA['pred_data_pos'].values[:]  # for handling nans
    pca_time = xds_PCA['pred_time'].values[:]
    pred_name = xds_PCA.attrs['pred_name']

    # PCA lat lon metadata
    lon = xds_PCA['pred_lon'].values
    lat = xds_PCA['pred_lat'].values

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    l_figs = []
    for it in range(n_plot):

        # get vargrd
        var_grd_1d = EOFs[:,it] * np.sqrt(variance[it])

        # insert nans in data
        base = np.nan * np.ones(data_pos.shape)
        base[data_pos] = var_grd_1d

        var = base[:]

        # reshape data to grid
        C1 = np.reshape(var, (len(lon), len(lat)))

        # figure
        fig = plt.figure(figsize=(_faspect*_fsize, 2.0/3.0*_fsize))

        # layout
        gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)

        ax_EOF_1 = plt.subplot(gs[:3, :2])
        ax_EOF_2 = plt.subplot(gs[:3, 2:])
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh (SLP and GRADIENT)
        axplot_EOF(ax_EOF_1, C1, lon, lat, ttl = pred_name, land=mask_land)

        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, pca_time, evol)

        # figure title
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        fig.suptitle(ttl, fontsize=14, fontweight='bold')

        l_figs.append(fig)

    # show and return figure
    if show: plt.show()
    return l_figs


# def Plot_EOFs_EstelaPred(xds_PCA, n_plot, mask_land=None, show=True):
#     '''
#     Plot annual EOFs for 3D predictors

#     xds_PCA:
#         (n_components, n_components) PCs
#         (n_components, n_features) EOFs
#         (n_components, ) variance

#         (n_lon, ) pred_lon: predictor longitude values
#         (n_lat, ) pred_lat: predictor latitude values
#         (n_time, ) pred_time: predictor time values

#         method: gradient + estela

#     n_plot: number of EOFs plotted
#     '''

#     # TODO: fix data_pos, fails only after pred.Load()?

#     # PCA data
#     variance = xds_PCA['variance'].values[:]
#     EOFs = np.transpose(xds_PCA['EOFs'].values[:])
#     PCs = np.transpose(xds_PCA['PCs'].values[:])
#     data_pos = xds_PCA['pred_data_pos'].values[:]  # for handling nans
#     pca_time = xds_PCA['pred_time'].values[:]
#     pred_name = xds_PCA.attrs['pred_name']

#     # PCA lat lon metadata
#     lon = xds_PCA['pred_lon'].values
#     lat = xds_PCA['pred_lat'].values

#     # percentage of variance each field explains
#     n_percent = variance / np.sum(variance)

#     l_figs = []
#     for it in range(n_plot):

#         # get vargrd 
#         var_grd_1d = EOFs[:,it] * np.sqrt(variance[it])

#         # insert nans in data
#         base = np.nan * np.ones(data_pos.shape)
#         base[data_pos] = var_grd_1d

#         var = base[:int(len(base)/2)]
#         grd = base[int(len(base)/2):]

#         # reshape data to grid
#         C1 = np.reshape(var, (len(lon), len(lat)))
#         C2 = np.reshape(grd, (len(lon), len(lat)))

#         # figure
#         fig = plt.figure(figsize=(_faspect*_fsize, 2.0/3.0*_fsize))

#         # layout
#         gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)

#         ax_EOF_1 = plt.subplot(gs[:3, :2])
#         ax_EOF_2 = plt.subplot(gs[:3, 2:])
#         ax_evol = plt.subplot(gs[3, :])

#         # EOF pcolormesh (SLP and GRADIENT)
#         axplot_EOF(ax_EOF_1, C1, lon, lat, ttl = pred_name, land=mask_land)
#         axplot_EOF(ax_EOF_2, C2, lon, lat, ttl = 'GRADIENT', land=mask_land)

#         # time series EOF evolution
#         evol =  PCs[it,:]/np.sqrt(variance[it])
#         axplot_EOF_evolution(ax_evol, pca_time, evol)

#         # figure title
#         ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
#         fig.suptitle(ttl, fontsize=14, fontweight='bold')

#         l_figs.append(fig)

#     # show and return figure
#     if show: plt.show()
#     return l_figs

def Plot_EOFs_EstelaPred(xds_PCA, n_plot, mask_land=None, show=True, figsize = None):
    '''
    Plot annual EOFs for 3D predictors

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance

        (n_lon, ) pred_lon: predictor longitude values
        (n_lat, ) pred_lat: predictor latitude values
        (n_time, ) pred_time: predictor time values

        method: gradient + estela

    n_plot: number of EOFs plotted
    '''

    # TODO: fix data_pos, fails only after pred.Load()?

    # PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = np.transpose(xds_PCA['EOFs'].values[:])
    PCs = np.transpose(xds_PCA['PCs'].values[:])
    data_pos = xds_PCA['pred_data_pos'].values[:]  # for handling nans
    pca_time = xds_PCA['pred_time'].values[:]
    pred_name = xds_PCA.attrs['pred_name']

    # PCA lat lon metadata
    lon = xds_PCA['pred_lon'].values
    lat = xds_PCA['pred_lat'].values

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    l_figs = []
    for it in range(n_plot):

        # get vargrd 
        var_grd_1d = EOFs[:,it] * np.sqrt(variance[it])

        # insert nans in data
        base = np.nan * np.ones(data_pos.shape)
        base[data_pos] = var_grd_1d

        var = base[:int(len(base)/2)]
        grd = base[int(len(base)/2):]

        # reshape data to grid
        C1 = np.reshape(var, (len(lon), len(lat)))
        C2 = np.reshape(grd, (len(lon), len(lat)))

        # figure
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(_faspect*_fsize, 2.0/3.0*_fsize))

        # layout
        gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)

        ax_EOF_1 = plt.subplot(gs[:3, :2], projection=ccrs.PlateCarree())#central_longitude = 180))
        #ax_EOF_1.add_feature(cfeature.LAND, zorder=0, color = 'lightgrey',edgecolor = 'lightgrey')
        ax_EOF_1.coastlines(resolution='50m', color='black')  # Draw only the coastline


        ax_EOF_2 = plt.subplot(gs[:3, 2:], projection=ccrs.PlateCarree())#central_longitude = 180))
        ax_EOF_2.add_feature(cfeature.LAND, zorder=0, color = 'lightgrey',edgecolor = 'lightgrey')
        ax_EOF_2.coastlines(resolution='50m', color='black')  # Draw only the coastline

        
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh (SLP and GRADIENT)
        axplot_EOF(ax_EOF_1, C1, lon, lat, ttl = pred_name, land=mask_land)
        axplot_EOF(ax_EOF_2, C2, lon, lat, ttl = 'GRADIENT', land=mask_land)

        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, pca_time, evol)

        # figure title
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        fig.suptitle(ttl, fontsize=14, fontweight='bold')

        l_figs.append(fig)

    # show and return figure
    if show: plt.show()
    return l_figs

def great_circles(lat1, lon1, ngc=16):

    """ Calculate great circles (from https://jorgeperezg.github.io/olas/reference/olas/estela/)

    Args:

        lat1 (float): Latitude origin point

        lon1 (float): Longitude origin point

        ngc (int, optional): Number of great circles

    Returns:

        xarray.Dataset: dataset with distance and bearing dimensions

    """

    D2R = np.pi / 180.0

    lat1_r = float(lat1) * D2R

    lon1_r = float(lon1) * D2R

    dist_r = xr.DataArray(dims="distance", data=np.linspace(0.5, 179.5, 180) * D2R)

    brng_r = xr.DataArray(dims="bearing", data=np.linspace(0, 360, ngc+1)[:-1] * D2R)

    sin_lat1 = np.sin(lat1_r)

    cos_lat1 = np.cos(lat1_r)

    sin_dR = np.sin(dist_r)

    cos_dR = np.cos(dist_r)

    lat2 = np.arcsin(sin_lat1*cos_dR + cos_lat1*sin_dR*np.cos(brng_r))

    lon2 = lon1_r + np.arctan2(np.sin(brng_r)*sin_dR*cos_lat1, cos_dR-sin_lat1*np.sin(lat2))

    gc = xr.Dataset({"latitude": lat2 / D2R, "longitude": (lon2 / D2R % 360).transpose()})

    gc["distance"] = dist_r / D2R

    gc["bearing"] = brng_r / D2R

    return gc



def Plot_Estela(est,extent, figsize=[20, 8]):

    fig = plt.figure(figsize=figsize)

    ax = plt.axes(projection = ccrs.PlateCarree())#central_longitude=180))
    ax.set_extent(extent,crs = ccrs.PlateCarree())
    ax.stock_img()


    # cartopy land feature
    land_10m = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='darkgrey', facecolor='gainsboro',  zorder=5)
    ax.add_feature(land_10m)
    ax.gridlines()


    # scatter data points
    vmin, vmax =96000, 103000

    p1=plt.pcolor(est.longitude.values, est.latitude.values, est.F.values, transform=ccrs.PlateCarree(),zorder=2, cmap='plasma')
    plt.colorbar(p1).set_label('Energy')

    plt.contour(est.longitude.values, est.latitude.values, est.traveltime.values, levels=np.arange(1,26,1), transform=ccrs.PlateCarree(),zorder=2, linestyle=':', color='royalblue')


    # plot great circles
    gc = great_circles(est.lat0, est.lon0, ngc=16)
    assert isinstance(ax.plot, object)
#    ax.plot(gc.longitude, gc.latitude, ".r", markersize=1, transform=ccrs.PlateCarree())
    plt.plot(gc.longitude, gc.latitude, ".r", markersize=1, transform=ccrs.PlateCarree())


    plt.show()
    return fig


# def Plot_DWTs_Mean_Anom(xds_KMA, xds_var, kind='mean', mask_land=None,
#                         show=True):
#     '''
#     Plot Daily Weather Types (bmus mean)
#     kind - mean/anom
#     '''

#     bmus = xds_KMA['sorted_bmus'].values[:]
#     n_clusters = len(xds_KMA.n_clusters.values[:])

#     var_max = np.max(xds_var.values)
#     var_min = np.min(xds_var.values)
#     scale = 1/100.0  # scale from Pa to mbar

#     # get number of rows and cols for gridplot 
#     n_rows, n_cols = GetBestRowsCols(n_clusters)

#     # get cluster colors
#     cs_dwt = colors_dwt(n_clusters)

#     # plot figure
#     fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

#     gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.1)
#     gr, gc = 0, 0

#     for ic in range(n_clusters):

#         if kind=='mean':
#             # data mean
#             it = np.where(bmus==ic)[0][:]
#             c_mean = xds_var.isel(time=it).mean(dim='time')
#             c_plot = np.multiply(c_mean, scale)  # apply scale

#         elif kind=='anom':
#             # data anomally
#             it = np.where(bmus==ic)[0][:]
#             t_mean = xds_var.mean(dim='time')
#             c_mean = xds_var.isel(time=it).mean(dim='time')
#             c_anom = c_mean - t_mean
#             c_plot = np.multiply(c_anom, scale)  # apply scale

#         # dwt color
#         clr = cs_dwt[ic]

#         # axes plot
#         ax = plt.subplot(gs[gr, gc])
#         pc = axplot_DWT(
#             ax, np.flipud(c_plot),
#             vmin = var_min, vmax = var_max,
#             wt_num = ic+1,
#             land = mask_land, wt_color = clr,
#         )

#         # anomalies colorbar center at 0 
#         if kind == 'anom':
#             pc.set_clim(-6,6)

#         # get lower positions
#         if gr==n_rows-1 and gc==0:
#             pax_l = ax.get_position()
#         elif gr==n_rows-1 and gc==n_cols-1:
#             pax_r = ax.get_position()

#         # counter
#         gc += 1
#         if gc >= n_cols:
#             gc = 0
#             gr += 1

#     # add a colorbar        
#     cbar_ax = fig.add_axes([pax_l.x0, pax_l.y0-0.05, pax_r.x1 - pax_l.x0, 0.02])
#     cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal')
#     if kind=='mean':
#         cb.set_label('Pressure (mbar)')
#     elif kind=='anom':
#         cb.set_label('Pressure anomalies (mbar)')

#     # show and return figure
#     if show: plt.show()
#     return fig

def Plot_DWTs_Mean_Anom(xds_KMA, xds_var, kind='mean', mask_land=None,
                        show=True, figsize = None, cbar_mean = 40, cbar_anom = 20):
    '''
    Plot Daily Weather Types (bmus mean)
    kind - mean/anom
    '''

    bmus = xds_KMA['sorted_bmus'].values[:]
    n_clusters = len(xds_KMA.n_clusters.values[:])

    # var_max = np.max(xds_var.values)
    # var_min = np.min(xds_var.values)
    
    
    
    scale = 1/100.0  # scale from Pa to mbar

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # get cluster colors
    cs_dwt = colors_dwt(n_clusters)

    # plot figure
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.1)
    gr, gc = 0, 0

    for ic in range(n_clusters):

        if kind=='mean':
            
            var_max = 1013+cbar_mean
            var_min = 1013-cbar_mean
            # data mean
            it = np.where(bmus==ic)[0][:]
            c_mean = xds_var.isel(time=it).mean(dim='time')
            c_plot = np.multiply(c_mean, scale)  # apply scale

        elif kind=='anom':
            
            var_max = cbar_anom
            var_min = -cbar_anom
            
            # data anomally
            it = np.where(bmus==ic)[0][:]
            t_mean = xds_var.mean(dim='time')
            c_mean = xds_var.isel(time=it).mean(dim='time')
            c_anom = c_mean - t_mean
            c_plot = np.multiply(c_anom, scale)  # apply scale

        # dwt color
        clr = cs_dwt[ic]

        # axes plot
        ax = plt.subplot(gs[gr, gc], projection=ccrs.PlateCarree())#central_longitude = 180))
        #ax.add_feature(cfeature.LAND, color = 'lightgrey',edgecolor = 'lightgrey')
        ax.coastlines(resolution='50m', color='black')  # Draw only the coastline

        
        pc = axplot_DWT(
            ax, xds_var, c_plot, #np.flipud(c_plot),
            vmin = var_min, vmax = var_max,
            wt_num = ic+1,
            land = mask_land, wt_color = clr,
        )

        # anomalies colorbar center at 0 
        if kind == 'anom':
            pc.set_clim(-cbar_anom,cbar_anom)

        # get lower positions
        if gr==n_rows-1 and gc==0:
            pax_l = ax.get_position()
        elif gr==n_rows-1 and gc==n_cols-1:
            pax_r = ax.get_position()

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # add a colorbar        
    cbar_ax = fig.add_axes([pax_l.x0, pax_l.y0-0.05, pax_r.x1 - pax_l.x0, 0.02])
    cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal')
    if kind=='mean':
        cb.set_label('Pressure (mbar)')
    elif kind=='anom':
        cb.set_label('Pressure anomalies (mbar)')

    # show and return figure
    if show: plt.show()
    return fig


def ClusterProbs_Month(bmus, time, wt_set, month_ix):
    'Returns Cluster probs by month_ix'

    # get months
    _, months, _ = get_years_months_days(time)

    if isinstance(month_ix, list):

        # get each month indexes
        l_ix = []
        for m_ix in month_ix:
            ixs = np.where(months == m_ix)[0]
            l_ix.append(ixs)

        # get all indexes     
        ix = np.unique(np.concatenate(tuple(l_ix)))

    else:
        ix = np.where(months == month_ix)[0]

    bmus_sel = bmus[ix]

    return ClusterProbabilities(bmus_sel, wt_set)

def Plot_DWTs_Probs(bmus, bmus_time, n_clusters, show=True):
    '''
    Plot Daily Weather Types bmus probabilities
    '''

    wt_set = np.arange(n_clusters) + 1

    # best rows cols combination
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(4, 7, wspace=0.10, hspace=0.25)

    # list all plots params
    l_months = [
        (1, 'January',   gs[1,3]),
        (2, 'February',  gs[2,3]),
        (3, 'March',     gs[0,4]),
        (4, 'April',     gs[1,4]),
        (5, 'May',       gs[2,4]),
        (6, 'June',      gs[0,5]),
        (7, 'July',      gs[1,5]),
        (8, 'August',    gs[2,5]),
        (9, 'September', gs[0,6]),
        (10, 'October',  gs[1,6]),
        (11, 'November', gs[2,6]),
        (12, 'December', gs[0,3]),
    ]

    l_3months = [
        ([12, 1, 2],  'DJF', gs[3,3]),
        ([3, 4, 5],   'MAM', gs[3,4]),
        ([6, 7, 8],   'JJA', gs[3,5]),
        ([9, 10, 11], 'SON', gs[3,6]),
    ]

    # plot total probabilities
    c_T = ClusterProbabilities(bmus, wt_set)
    C_T = np.reshape(c_T, (n_rows, n_cols))

    ax_probs_T = plt.subplot(gs[:2, :2])
    pc = axplot_WT_Probs(ax_probs_T, C_T, ttl = 'DWT Probabilities')

    # plot counts histogram
    ax_hist = plt.subplot(gs[2:, :3])
    axplot_WT_Hist(ax_hist, bmus, n_clusters, ttl = 'DWT Counts')

    # plot probabilities by month
    vmax = 0.15
    for m_ix, m_name, m_gs in l_months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax)

    # TODO: add second colorbar?

    # plot probabilities by 3 month sets
    vmax = 0.15
    for m_ix, m_name, m_gs in l_3months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus_time, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax, cmap='Greens')

    # add custom colorbar
    pp = ax_probs_T.get_position()
    cbar_ax = fig.add_axes([pp.x1+0.02, pp.y0, 0.02, pp.y1 - pp.y0])
    cb = fig.colorbar(pc, cax=cbar_ax, cmap='Blues')
    cb.ax.tick_params(labelsize=8)

    # show and return figure
    if show: plt.show()
    return fig

