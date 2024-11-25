#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import copy
from datetime import datetime, date

from PIL import Image
import cmocean 

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
    ax.tick_params(axis='both', which='major', labelsize=4)

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

    var = xds_var.name

    if var=='geo500hpa':
        cmap = cm.get_cmap('GnBu') # copy.deepcopy(cm.get_cmap('GnBu'))
    else:
        cmap = cm.get_cmap('RdBu_r') # copy.deepcopy(cm.get_cmap('RdBu_r')) 

    # EOF pcolormesh 
    # pc = ax.pcolormesh(
    #     xds_var.longitude.values, xds_var.latitude.values, dwt,
    #     cmap = cmap, shading = 'gouraud',
    #     clim = (vmin, vmax), transform = ccrs.PlateCarree()
    # )

    levels = np.linspace(vmin, vmax, 101)  # 101 niveles (100 intervalos)


    pc = ax.contourf(
        xds_var.longitude.values, xds_var.latitude.values, dwt, levels,
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

    var = xds_var.name


    for ic in range(n_clusters):

        if kind=='mean':

            if var=='geo500hpa':
                var_max = 57322+cbar_mean
                var_min = 57322-cbar_mean
                it = np.where(bmus==ic)[0][:]
                c_mean = xds_var.isel(time=it).mean(dim='time')
                c_plot = c_mean # np.multiply(c_mean, scale)  # apply scale

            else:
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
    cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal', extend='both')
    
    if kind=='mean':
        if var == 'geo500hpa':
            cb.set_label('Pressure (m2 s-2)')
            plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')
        else:
            cb.set_label('Pressure (mbar)')
            plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')

    elif kind=='anom':
        if var == 'geo500hpa':
            cb.set_label('Pressure anomalies (m2 s-2)')
            plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')
        else:
            cb.set_label('Pressure anomalies (mbar)')
            plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')

    # show and return figure
    if show: plt.show()
    return fig








def Plot_DWTs_Spatial_Maps(xds_KMA, xds_var,  waves, waves_period, rain, twl, kind='mean', mask_land=None,
                        show=True, figsize=None, cbar_mean=20, cbar_anom=20):
    '''
    Plot Daily Weather Types (bmus mean) in a single vertical column.
    kind - mean/anom
    '''
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean
    import numpy as np
    import copy

    bmus = xds_KMA['sorted_bmus'].values[:]
    n_clusters = len(xds_KMA.n_clusters.values[:])
    scale = 1/100.0  # scale from Pa to mbar

    # Set colors for each cluster
    cs_dwt = colors_dwt(n_clusters)
    

    # Lista de variables a graficar
    vars = ['bmus', 'swh', 'mwp', 'precipitation', 'waterlevel']  # Variables de interés
    num_clusters = 36
    c_clusters = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Colores para cada cluster

    # Establecer límites de color para cada variable
    color_limits = {
        'swh': (0, 7),  # Ejemplo de rango en metros para 'swh'
        'mwp': (0, 10),  # Ejemplo de rango en segundos para 'mwp'
        'precipitation': (0, 5),  # Ejemplo de rango en mm para 'precipitation',
        'waterlevel': (-20, 20)  # Ejemplo de rango en cm para 'waterlevel',

    }

    # Crear la figura y los subplots
    fig = plt.figure(figsize=(2.5 * len(vars), 2 * num_clusters))  # Ajusta el tamaño según el número de variables y clusters
    gs = gridspec.GridSpec(num_clusters, len(vars), hspace=0.3, wspace=0.6)  # Filas para clusters, columnas para variables

    # Iterar sobre cada cluster y variable
    for i_cluster in range(num_clusters):
        for j_var, var in enumerate(vars):
            ax = fig.add_subplot(gs[i_cluster, j_var], projection=ccrs.PlateCarree())  # Proyección cartográfica genérica

            if var == 'bmus':
                # Determine data to plot based on `kind`
                if kind == 'mean':
                    var_max = 1013 + cbar_mean
                    var_min = 1013 - cbar_mean
                    # Data mean
                    it = np.where(bmus == i_cluster)[0][:]
                    c_mean = xds_var.isel(time=it).mean(dim='time')
                    c_plot = np.multiply(c_mean, scale)  # Apply scale

                elif kind == 'anom':
                    var_max = cbar_anom
                    var_min = -cbar_anom
                    # Data anomaly
                    it = np.where(bmus == i_cluster)[0][:]
                    t_mean = xds_var.mean(dim='time')
                    c_mean = xds_var.isel(time=it).mean(dim='time')
                    c_anom = c_mean - t_mean
                    c_plot = np.multiply(c_anom, scale)  # Apply scale

                # DWT color
                clr = cs_dwt[i_cluster]

                # Axes plot
                ax = plt.subplot(gs[i_cluster, 0], projection=ccrs.PlateCarree())
                ax.coastlines(resolution='50m', color='black')

                # Plot data
                pc = axplot_DWT(
                    ax, xds_var, c_plot,
                    vmin=var_min, vmax=var_max,
                    wt_num=i_cluster + 1,
                    land=mask_land, wt_color=clr,
                )

                # Set color limits for anomalies
                if kind == 'anom':
                    pc.set_clim(-cbar_anom, cbar_anom)

                # Add title for each subplot
                ax.set_title(f'Cluster {i_cluster + 1}', color=clr, fontsize=10)
                
                if i_cluster == num_clusters - 1:
                    # Add a colorbar below the plots


                    ticks_values = [1000, 1005, 1010]
                    
                    cbar_ax = fig.add_axes([
                        ax.get_position().x0 + 0.01,  # Desplaza un poco hacia la derecha
                        ax.get_position().y0 - 0.01,  # Posición vertical justo debajo del gráfico
                        ax.get_position().width * 0.8,  # Ancho un poco más estrecho que el gráfico
                        0.001  # Altura más pequeña para hacer el colorbar más delgado
                    ])
                    cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal', ticks=ticks_values)

                    if kind == 'mean':
                        cb.set_label('Pressure (mbar)')
                    elif kind == 'anom':
                        cb.set_label('Pressure anomalies (mbar)')

                    cbar_ax.tick_params(labelsize=6)  # Cambia el tamaño de los números en el colorbar
                



            else:
            
                # Seleccionar el dataset y colormap correspondiente a la variable actual
                if var == 'swh':
                    ds = waves
                    c_cmap = 'rainbow'
                elif var == 'mwp':
                    ds = waves_period
                    c_cmap = 'viridis'
                elif var == 'precipitation':
                    ds = rain
                    c_cmap = cmocean.cm.rain
                elif var == 'waterlevel':
                    ds = twl
                    c_cmap = 'rainbow'

                
                if var == 'waterlevel':

                    #ax = fig.add_subplot(gs[i_cluster, 0], projection=ccrs.PlateCarree())
                        
                    # Seleccionar las estaciones que pertenecen al clúster i_cluster
                    times_clusters = twl.cluster == i_cluster  # Filtrar el clúster en base a la variable `cluster`
                    
                    # Obtener las longitudes y latitudes de las estaciones
                    cluster_lon = twl.station_x_coordinate.values  # Longitudes de las estaciones
                    cluster_lat = twl.station_y_coordinate.values  # Latitudes de las estaciones
                    
                    # Crear listas vacías para almacenar las coordenadas y los valores de waterlevel
                    all_lon = []
                    all_lat = []
                    all_mean_waterlevels = []
                        
                    # Para cada estación
                    for i_station in range(len(twl.stations.values)):
                        
                        # Calcular la media de waterlevel para la estación i_station a lo largo del tiempo
                        mean_waterlevel = np.nanmean(twl.waterlevel.values[times_clusters, i_station])  # Promedio sobre 'time'
                        
                        # Almacenar las coordenadas y el valor de mean_waterlevel
                        all_lon.append(cluster_lon[i_station])
                        all_lat.append(cluster_lat[i_station])
                        all_mean_waterlevels.append(mean_waterlevel)
                    
                    # Convertir las listas en arrays para graficar más fácilmente
                    all_lon = np.array(all_lon)
                    all_lat = np.array(all_lat)
                    all_mean_waterlevels = np.array(all_mean_waterlevels)
                    
                    # Graficar todas las estaciones a la vez
                    sc = ax.scatter(all_lon, all_lat, c=all_mean_waterlevels, cmap=c_cmap, 
                                    s=2, zorder=3, vmin=-20, vmax=20)
                    
                    # Añadir características geográficas
                    ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgrey')  
                    ax.coastlines() 

                    # Configurar título y etiquetas
                    ax.set_title(f'Cluster {i_cluster + 1} - {var}', fontsize=10)

                    # Añadir colorbar horizontal debajo de cada columna solo en la última fila
                    if i_cluster == num_clusters - 1:
                        # Configurar la posición del colorbar debajo del gráfico
                        cbar_ax = fig.add_axes([
                            ax.get_position().x0 + 0.01,  # Desplaza un poco hacia la derecha
                            ax.get_position().y0 - 0.01,  # Posición vertical justo debajo del gráfico
                            ax.get_position().width * 0.8,  # Ancho un poco más estrecho que el gráfico
                            0.001  # Altura más pequeña para hacer el colorbar más delgado
                        ])
                        fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', label=f'{var} mean values')


                else:

                    # Seleccionar los datos de la variable `var` donde el cluster es igual al valor actual
                    mean_cluster = ds[var].where(ds['cluster'] == i_cluster, drop=True).mean(dim='time')
                    
                    # Graficar el campo espacial medio para el cluster actual y variable actual, con escala fija
                    vmin, vmax = color_limits[var]  # Usar el rango predefinido para cada variable
                    im = mean_cluster.plot(ax=ax, cmap=c_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, zorder=1)

                    # Añadir contornos de tierra y costas
                    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgrey')
                    ax.coastlines(zorder=2)
                    
                    # Configurar título y etiquetas
                    ax.set_title(f'Cluster {i_cluster + 1} - {var}', fontsize=10)
                    
                    # Añadir colorbar horizontal debajo de cada columna solo en la última fila
                    if i_cluster == num_clusters - 1:
                        # Configurar la posición del colorbar debajo del gráfico
                        cbar_ax = fig.add_axes([
                            ax.get_position().x0 + 0.01,  # Desplaza un poco hacia la derecha
                            ax.get_position().y0 - 0.01,  # Posición vertical justo debajo del gráfico
                            ax.get_position().width * 0.8,  # Ancho un poco más estrecho que el gráfico
                            0.001  # Altura más pequeña para hacer el colorbar más delgado
                        ])
                        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=f'{var} mean values')


    # Mostrar el gráfico
    plt.tight_layout(rect=[0, 0.1, 0.9, 1])  # Ajusta los márgenes para los colorbars individuales
    plt.plot()





def generate_predictand_plot(var, waves, waves_period, rain, twl, num_clusters):

    if var == 'waterlevel':

        n_grid = int(np.sqrt(num_clusters))

        # Crear el colormap para los clústeres
        c_clusters = plt.cm.viridis(np.linspace(0, 1, num_clusters))

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(n_grid, n_grid)  # Crear una grilla de subgráficos

        # Para cada clúster
        for i_cluster in range(num_clusters):
            ax = fig.add_subplot(gs[i_cluster], projection=ccrs.PlateCarree())
            
            # Seleccionar las estaciones que pertenecen al clúster i_cluster
            times_clusters = twl.cluster == i_cluster  # Filtrar el clúster en base a la variable `cluster`
            
            # Obtener las longitudes y latitudes de las estaciones
            cluster_lon = twl.station_x_coordinate.values  # Longitudes de las estaciones
            cluster_lat = twl.station_y_coordinate.values  # Latitudes de las estaciones
            
            # Crear listas vacías para almacenar las coordenadas y los valores de waterlevel
            all_lon = []
            all_lat = []
            all_mean_waterlevels = []
                
            # Para cada estación
            for i_station in range(len(twl.stations.values)):
                
                # Calcular la media de waterlevel para la estación i_station a lo largo del tiempo
                mean_waterlevel = np.nanmean(twl.waterlevel.values[times_clusters, i_station])  # Promedio sobre 'time'
                
                # Almacenar las coordenadas y el valor de mean_waterlevel
                all_lon.append(cluster_lon[i_station])
                all_lat.append(cluster_lat[i_station])
                all_mean_waterlevels.append(mean_waterlevel)
            
            # Convertir las listas en arrays para graficar más fácilmente
            all_lon = np.array(all_lon)
            all_lat = np.array(all_lat)
            all_mean_waterlevels = np.array(all_mean_waterlevels)
            
            # Graficar todas las estaciones a la vez
            sc = ax.scatter(all_lon, all_lat, c=all_mean_waterlevels, cmap='rainbow', 
                            s=2, zorder=3, vmin=-20, vmax=20)
            
            # Añadir características geográficas
            ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgrey')  
            ax.coastlines()  
            
            
            cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])

            fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', label='Mean water level (cm)')


        # Ajustar los subgráficos y mostrar
        plt.subplots_adjust(hspace=0.1, bottom=0.1, left=0.1, right=0.9)  
        plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')
        plt.close(fig) 



    else:

        n_grid = int(np.sqrt(num_clusters))
        c_clusters = plt.cm.viridis(np.linspace(0, 1, num_clusters)) 

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(n_grid, n_grid)  

        for i_cluster in range(num_clusters):
            ax = fig.add_subplot(gs[i_cluster], projection=ccrs.PlateCarree())  

            # colormap selection
            if var == 'swh':
                ds = waves[var].where(waves['cluster'] == i_cluster, drop=True).mean(dim='time')
                c_cmap = 'rainbow'
            elif var == 'mwp':
                c_cmap = 'viridis'
                ds = waves_period[var].where(waves_period['cluster'] == i_cluster, drop=True).mean(dim='time')
            elif var == 'precipitation':
                c_cmap = cmocean.cm.rain
                ds = rain[var].where(rain['cluster'] == i_cluster, drop=True).mean(dim='time')

            im = ds.plot(ax=ax, cmap=c_cmap, add_colorbar=False)
            
            ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgrey')  
            ax.coastlines()
            
            ax.set_title('')  
            

        plt.subplots_adjust(hspace=0.1, bottom=0.1, left=0.1, right=0.9)  

        cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])

        if var == 'swh':
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Mean Significant Wave Height (m)')
        elif var == 'mwp':
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Mean Wave Period (s)')
        elif var == 'precipitation':
            fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Precipitation (mm/h)')



        plt.savefig(f'results/DWT_{var}.png', dpi=300, bbox_inches='tight')
        plt.close(fig) 




def plot_DWT_predictand(p_out, var1,var2):

    img1 = Image.open(op.join(p_out,'DWT_'f'{var1}.png'))
    img2 = Image.open(op.join(p_out,'DWT_'f'{var2}.png'))

    img1 = img1.resize((800, 600))  
    img2 = img2.resize((800, 600))  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 12))  

    ax1.imshow(img1)
    ax1.axis('off')  

    ax2.imshow(img2)
    ax2.axis('off')

    plt.show()





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

def Plot_EOFs_EstelaPred_geo(xds_PCA, n_plot, mask_land=None, show=True, figsize = None):
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

    #&nbsp;TODO: fix data_pos, fails only after pred.Load()?

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

        var = base[:int(len(base)/3)]
        grd = base[int(len(base)/3):int(len(base)*2/3)]
        geo = base[int(len(base)*2/3):]


        # reshape data to grid
        C1 = np.reshape(var, (len(lon), len(lat)))
        C2 = np.reshape(grd, (len(lon), len(lat)))
        C3 = np.reshape(geo, (len(lon), len(lat)))


        # figure
        if figsize:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=(_faspect*_fsize, 2.0/3.0*_fsize))

        # layout
        gs = gridspec.GridSpec(4, 6, wspace=0.10, hspace=0.2)

        ax_EOF_1 = plt.subplot(gs[:3, :2], projection=ccrs.PlateCarree())
        ax_EOF_1.add_feature(cfeature.LAND, color = 'lightgrey',edgecolor = 'lightgrey')
        ax_EOF_1.coastlines(resolution='50m', color='black')  # Draw only the coastline


        ax_EOF_2 = plt.subplot(gs[:3, 2:4], projection=ccrs.PlateCarree())
        ax_EOF_2.add_feature(cfeature.LAND, color = 'lightgrey',edgecolor = 'lightgrey')
        ax_EOF_2.coastlines(resolution='50m', color='black')  # Draw only the coastline


        ax_EOF_3 = plt.subplot(gs[:3, 4:6], projection=ccrs.PlateCarree())
        ax_EOF_3.add_feature(cfeature.LAND, color = 'lightgrey',edgecolor = 'lightgrey')
        ax_EOF_3.coastlines(resolution='50m', color='black')  # Draw only the coastline

        
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh (SLP and GRADIENT)
        axplot_EOF(ax_EOF_1, C1, lon, lat, ttl = pred_name, land=mask_land)
        axplot_EOF(ax_EOF_2, C2, lon, lat, ttl = 'gradient', land=mask_land)
        axplot_EOF(ax_EOF_3, C3, lon, lat, ttl = 'Geopotential 500 HPa', land=mask_land)


        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, pca_time, evol)

        # figure title
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        fig.suptitle(ttl, fontsize=8, fontweight='bold')

        l_figs.append(fig)

    # show and return figure
    if show: plt.show()
    return l_figs




