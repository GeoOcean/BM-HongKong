#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import itertools
import calendar
from datetime import datetime

# pip
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from PIL import Image

# teslakit
from ..util.operations import GetBestRowsCols
from ..util.time_operations import get_years_months_days
from .custom_colors import colors_awt
from .pcs import axplot_PC_hist, axplot_PCs_3D


# import constants
from .config import _faspect, _fsize

def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
    'axes plot AWT variable (2D)'

    # Plot 2D AWT
    ax.pcolormesh(
        var_2D,
        cmap='RdBu_r', shading='gouraud',
        vmin=-1.5, vmax=+1.5,
    )

    # Title and axis labels/ticks
    ax.set_title(
        'WT #{0} --- {1} years'.format(id_wt, num_wts),
        {'fontsize': 14, 'fontweight': 'bold'}
    )

    # Set the y-ticks to represent the months you want (e.g., March, December, September, June)
    ax.set_yticks([0, 3, 6, 9])  # Ticks at Mar (2), Jun (5), Sept (8), Dec (11)

    # Assign labels to those ticks
    ax.set_yticklabels(['Jun', 'Sept', 'Dec','Mar'], fontsize=8)

    # Set x-ticks (you can customize this as needed)
    ax.set_xticks([])  # Remove x-axis ticks if not needed, or customize as per your data

    # Set axis labels
    ax.set_xlabel('Lon', fontsize=8)

    # Set WT color on axis frame
    plt.setp(ax.spines.values(), color=color_wt, linewidth=4)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color_wt)


def axplot_AWT_years(ax, dates_wt, bmus_wt, color_wt, xticks_clean=False,
                     ylab=None, xlims=None):
    'axes plot AWT dates'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years string
    ys_str = np.array([str(d).split('-')[0] for d in dates_wt])

    # use a text bottom - top cycler
    text_cycler_va = itertools.cycle(['bottom', 'top'])
    text_cycler_ha = itertools.cycle(['left', 'right'])

    # plot AWT dates and bmus
    ax.plot(
        dates_wt, bmus_wt,
        marker='+',markersize=9, linestyle='', color=color_wt,
    )
    va = 'bottom'
    for tx,ty,tt in zip(dates_wt, bmus_wt, ys_str):
        ax.text(
            tx, ty, tt,
            {'fontsize':8},
            verticalalignment = next(text_cycler_va),
            horizontalalignment = next(text_cycler_ha),
            rotation=45,
        )

    # configure axis
    ax.set_yticks([])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='x', which='major', labelsize=8)

    # optional parameters
    if xticks_clean:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year', {'fontsize':8})

    if ylab: ax.set_ylabel(ylab)

    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])

def axplot_EOF_evolution(ax, years, EOF_evol):
    'axes plot EOFs evolution'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years datetime
    ys_dt = np.array([datetime(y,1,1) for y in years])

    # plot EOF evolution 
    ax.plot(
        ys_dt, EOF_evol,
        linestyle='-', color='black',
    )

    # configure axis
    ax.set_xlim(ys_dt[0], ys_dt[-1])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=8)

def axplot_EOF(ax, EOF_value, lon, ylbl, ttl):
    'axes plot EOFs evolution'

    # EOF pcolormesh 
    ax.pcolormesh(
        lon, range(12), np.transpose(EOF_value),
        cmap='RdBu_r', shading='gouraud',
        vmin = -1, vmax=1 
    )

    # axis and title
    ax.set_yticklabels(ylbl)
    ax.set_title(
        ttl,
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=10)


def Plot_AWT_Validation_Cluster(AWT_2D, AWT_num_wts, AWT_ID, AWT_dates,
                                AWT_bmus, AWT_PCs_fit, AWT_PCs_rnd, AWT_color,
                                show=True):


    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    # layout
    gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.15)
    ax_AWT_2D = plt.subplot(gs[:2, :2])
    ax_PCs3D_fit = plt.subplot(gs[2, 0], projection='3d')
    ax_PCs3D_rnd = plt.subplot(gs[2, 1], projection='3d')
    ax_AWT_y = plt.subplot(gs[3, :])
    ax_PC1_hst_fit = plt.subplot(gs[0, 2])
    ax_PC1_hst_rnd = plt.subplot(gs[0, 3])
    ax_PC2_hst_fit = plt.subplot(gs[1, 2])
    ax_PC2_hst_rnd = plt.subplot(gs[1, 3])
    ax_PC3_hst_fit = plt.subplot(gs[2, 2])
    ax_PC3_hst_rnd = plt.subplot(gs[2, 3])

    # plot AWT 2D
    axplot_AWT_2D(ax_AWT_2D, AWT_2D, AWT_num_wts, AWT_ID, AWT_color)

    # plot AWT years
    axplot_AWT_years(ax_AWT_y, AWT_dates, AWT_bmus, AWT_color)

    # compare PCs fit - sim with 3D plot
    axplot_PCs_3D(ax_PCs3D_fit, AWT_PCs_fit,  AWT_color, ttl='PCs fit')
    axplot_PCs_3D(ax_PCs3D_rnd, AWT_PCs_rnd,  AWT_color, ttl='PCs sim')

    # compare PC1 histograms
    axplot_PC_hist(ax_PC1_hst_fit, AWT_PCs_fit[:,0], AWT_color)
    axplot_PC_hist(ax_PC1_hst_rnd, AWT_PCs_rnd[:,0], AWT_color, ylab='PC1')

    axplot_PC_hist(ax_PC2_hst_fit, AWT_PCs_fit[:,1], AWT_color)
    axplot_PC_hist(ax_PC2_hst_rnd, AWT_PCs_rnd[:,1], AWT_color, ylab='PC2')

    axplot_PC_hist(ax_PC3_hst_fit, AWT_PCs_fit[:,2], AWT_color)
    axplot_PC_hist(ax_PC3_hst_rnd, AWT_PCs_rnd[:,2], AWT_color, ylab='PC3')

    # show
    if show: plt.show()
    return fig

def Plot_AWTs_Validation(bmus, dates, Km, n_clusters, lon, d_PCs_fit,
                         d_PCs_rnd, show=True):
    '''
    Plot Annual Weather Types Validation

    bmus, dates, Km, n_clusters, lon - from KMA_simple()
    d_PCs_fit, d_PCs_rnd - historical and simulated PCs by WT
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # each cluster has a figure
    l_figs = []
    for ic in range(n_clusters):

        # get cluster data
        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        var_AWT = Km[ic,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)      # number of cluster ocurrences
        clr = cs_awt[ic]          # cluster color
        PCs_fit = d_PCs_fit['{0}'.format(id_AWT)]
        PCs_rnd = d_PCs_rnd['{0}'.format(id_AWT)]

        # plot cluster figure
        fig = Plot_AWT_Validation_Cluster(
            var_AWT_2D, num_WTs, id_AWT,
            dates_AWT, bmus_AWT,
            PCs_fit, PCs_rnd,
            clr, show=show)

        l_figs.append(fig)

    return l_figs

def Plot_AWTs(bmus, Km, n_clusters, lon, show=True):
    '''
    Plot Annual Weather Types

    bmus, Km, n_clusters, lon - from KMA_simple()
    '''

    # get number of rows and cols for gridplot 
    n_cols, n_rows = GetBestRowsCols(n_clusters)

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)
    gr, gc = 0, 0

    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        var_AWT = Km[ic,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)
        clr = cs_awt[ic]          # cluster color

        # AWT var 2D 
        ax = plt.subplot(gs[gr, gc])
        axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)

        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # show and return figure
    if show: plt.show()
    return fig

def Calculate_AWTs_DWTs(SLP_KMA, SST_AWTs):
    '''
        Historical probability of occurrence for DWTs within each AWT
    '''

    awts = SST_AWTs.bmus.values
    years_awts = SST_AWTs.time.values
    
    ds_awts = xr.Dataset(
        {
            'awt': ('time', awts)  # La variable `awt` usando `time` como dimensión
        },
        coords={
            'time': pd.to_datetime(years_awts, format='%Y')  # Convertir `time` a tipo datetime
        }
    )

    last_year = ds_awts['time'].dt.year.max().item()

    first_year = SLP_KMA['time'].dt.year.min().item()

    time_index = ds_awts['time'].to_index()

    pos = time_index.get_loc(f"{first_year}-06-01")

    ds_awts = ds_awts.isel(time=slice(pos, None))

    end_time_str = f'{last_year}-12-31 00:00:00'
    end_time_date = pd.to_datetime(end_time_str)

    ds_recortado = SLP_KMA.sel(time=slice(None, end_time_date))

    # Asegúrate de que 'time' está en formato datetime64 (si no lo está, conviértelo)
    ds_recortado['time'] = pd.to_datetime(ds_recortado['time'].values)

    # Extraer el año de cada fecha en ds_recortado
    years_recortado = ds_recortado['time'].dt.year

    awt_values = ds_awts.awt.values

    df_year = ds_awts.to_dataframe()

    df_year = pd.DataFrame(data=awt_values, columns=['awt'], index=np.arange(1940, 2021))

    daily_times = years_recortado.time.values

    # crear un xr dataset con una variable en diario
    ds = xr.Dataset(
        {'clusters': (('time'), ds_recortado.cluster.values)},
        
        coords={'time': daily_times}
    )

    df_var = ds.to_dataframe()
    df_var['year'] = df_var.index.year
    df_var = df_var.reset_index().set_index('year')

    df_result = pd.merge(df_var, df_year, left_index=True, right_index=True)
    ds_awts_plot = df_result.to_xarray()

    

    num_clusters = len(np.unique(ds_awts_plot['clusters'].values))  # Number of clusters
    unique_awts = np.unique(ds_awts_plot['awt'].values)  # Number of AWT
    num_awts = len(unique_awts)

    fig = plt.figure(figsize=(20, 3))
    gs = gridspec.GridSpec(1, num_awts, wspace=0.3)  

    for i, awt_value in enumerate(unique_awts):
        ax = fig.add_subplot(gs[0, i])
        
        awt_clusters = ds_awts_plot['clusters'].where(ds_awts_plot['awt'] == awt_value, drop=True)
        
        cluster_probabilities = np.zeros(num_clusters)

        for cluster_id in range(num_clusters):
            cluster_count = (awt_clusters == cluster_id).sum().item()  
            total_count = len(awt_clusters)  
            cluster_probabilities[cluster_id] = (cluster_count / total_count) * 100 if total_count > 0 else 0  # Transform to %

        probabilities_grid = cluster_probabilities.reshape(int(num_clusters** 0.5), int(num_clusters** 0.5))

        
        im = ax.imshow(probabilities_grid, cmap='Reds', aspect='auto', vmin=0, vmax=15)
        ax.set_title(f'AWT: {awt_value+1}')
        ax.axis('off')  

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Cluster Probability (%)')
    cbar.set_ticks([0, 5, 10, 15])  

    plt.savefig('DWT_probabilities_AWT.png', dpi=300, bbox_inches='tight')

    plt.close(fig)



def Create_Plot_AWTs(bmus, Km, n_clusters, lon, show=False):
    '''
    Plot Annual Weather Types in a single row.

    bmus, Km, n_clusters, lon - from KMA_simple()
    '''

    # Set single row and multiple columns
    n_cols, n_rows = n_clusters, 1

    # Get cluster colors
    cs_awt = colors_awt()

    # Define figure size and grid layout
    fig = plt.figure(figsize=(20, 3))  # Adjust width based on clusters

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.3, hspace=0.15)

    # Loop through clusters and plot each in a single row
    for ic in range(n_clusters):
        id_AWT = ic + 1           # Cluster ID
        index = np.where(bmus == ic)[0][:]
        var_AWT = Km[ic, :]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)
        clr = cs_awt[ic]          # Cluster color

        # Plot AWT var 2D 
        ax = plt.subplot(gs[0, ic])  # Use the single row
        axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)

    plt.savefig('AWT.png', dpi=300, bbox_inches='tight')

    # Only show the plot if show=True
    if show:
        plt.show()



def Plot_AWTs_DWTs():
    # Cargar imágenes
    img1 = Image.open("AWT.png")
    img2 = Image.open("DWT_probabilities_AWT.png")

    # Obtener el tamaño de las imágenes
    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size

    # Crear la figura con el tamaño adecuado para las dos imágenes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(img1_width / 100, (img1_height + img2_height) / 100))  # Ajustar tamaño de la figura

    # Mostrar las imágenes sin redimensionarlas
    ax1.imshow(img1)
    ax1.axis('off')  # Ocultar los ejes

    ax2.imshow(img2)
    ax2.axis('off')  # Ocultar los ejes

    # Ajustar el espaciado entre los subgráficos
    plt.subplots_adjust(hspace=0.1)  # Ajustar el espacio entre los subgráficos
    plt.close()

    # Mostrar el gráfico
    #plt.show()





def Plot_AWTs_Dates(bmus, dates, n_clusters, show=True):
    '''
    Plot Annual Weather Types dates

    bmus, dates, n_clusters - from KMA_simple()
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(nrows=n_clusters, figsize=(_faspect*_fsize, _fsize))

    # each cluster has a figure
    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        clr = cs_awt[ic]          # cluster color

        ylabel = "WT #{0}".format(id_AWT)
        xlims = [dates[0].astype('datetime64[Y]')-np.timedelta64(3, 'Y'), dates[-1].astype('datetime64[Y]')+np.timedelta64(3, 'Y')]

        xaxis_clean=True
        if ic == n_clusters-1:
            xaxis_clean=False

        # axs plot
        axplot_AWT_years(
            axs[ic], dates_AWT, bmus_AWT,
            clr, xaxis_clean, ylabel, xlims
        )

    # show and return figure
    if show: plt.show()
    return fig


def Plot_AWTs_EOFs(PCs, EOFs, variance, time, lon, n_plot, show=True):
    '''
    Plot annual EOFs for PCA_LatitudeAverage predictor

    PCs, EOFs, variance, time, lon - from PCA_LatitudeAverage()
    n_plot                         - number of EOFs plotted
    '''

    # transpose
    EOFs = np.transpose(EOFs)
    PCs = np.transpose(PCs)

    # get start and end month
    ys, ms, _ = get_years_months_days(time)

    # PCA latavg metadata
    y1 = ys[0]
    y2 = ys[-1]
    m1 = ms[0]
    m2 = ms[-1]

    # mesh data
    len_x = len(lon)

    # time data
    years = range(y1, y2+1)
    l_months = [calendar.month_name[x] for x in range(1,13)]
    ylbl = l_months[m1-1:] + l_months[:m2]

    # percentage of variance each field explains
    n_percent = variance / np.sum(variance)

    l_figs = []
    for it in range(n_plot):

        # map of the spatial field
        spatial_fields = EOFs[:,it]*np.sqrt(variance[it])

        # reshape from vector to matrix with separated months
        C = np.reshape(
            spatial_fields[:len_x*12], (12, len_x)
        ).transpose()

        # plot figure
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

        # layout
        gs = gridspec.GridSpec(4, 4, wspace=0.10, hspace=0.2)
        ax_EOF = plt.subplot(gs[:3, :])
        ax_evol = plt.subplot(gs[3, :])

        # EOF pcolormesh
        ttl = 'EOF #{0}  ---  {1:.2f}%'.format(it+1, n_percent[it]*100)
        axplot_EOF(ax_EOF, C, lon, ylbl, ttl)

        # time series EOF evolution
        evol =  PCs[it,:]/np.sqrt(variance[it])
        axplot_EOF_evolution(ax_evol, years, evol)

        l_figs.append(fig)

        # show
        if show: plt.show()

    return l_figs

