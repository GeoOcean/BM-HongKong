#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

# pip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# teslakit
from ..util.operations import GetBestRowsCols
from .custom_colors import GetFamsColors
from .outputs import axplot_compare_histograms

# import constants
from .config import _faspect, _fsize, _fdpi


def axplot_distplot(ax, vars_values, vars_colors, n_bins, wt_num, xlims):
    'axes plot seaborn distplot variable at families'

    for vv, vc in zip(vars_values, vars_colors):
        #sns.distplot(vv, bins=n_bins, color=tuple(vc), ax=ax);
        sns.histplot(vv, bins=n_bins, color=tuple(vc), ax=ax, kde=True, stat='density');

    # wt text
    ax.text(0.87, 0.85, wt_num, transform=ax.transAxes, fontweight='bold')

    # customize axes
    ax.set_xlim(xlims)
    #ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])


def axplot_polarhist(ax, vars_values, vars_colors, n_bins, wt_num):
    'axes plot polar hist dir at families'

    for vv, vc in zip(vars_values, vars_colors):
        plt.hist(
            np.deg2rad(vv),
            range = [0, np.deg2rad(360)],
            bins = n_bins, color = vc,
            histtype='stepfilled', alpha = 0.5,
        )

    # wt text
    ax.text(0.87, 0.85, wt_num, transform=ax.transAxes, fontweight='bold')

    # customize axes
    ax.set_facecolor('whitesmoke')
    ax.set_xticks([])
    ax.set_yticks([])


def Plot_Waves_DWTs(xds_wvs_fams_sel, bmus, n_clusters, show=True):
    '''
    Plot waves families by DWT

    wvs_fams (waves families):
        xarray.Dataset (time,), fam1_Hs, fam1_Tp, fam1_Dir, ...
        {any number of families}

    xds_DWTs - ESTELA predictor KMA
        xarray.Dataset (time,), bmus, ...
    '''

    # plot_parameters
    n_bins = 35

    # get families names and colors
    n_fams = [vn.replace('_Hs','') for vn in xds_wvs_fams_sel.keys() if '_Hs' in vn]
    fams_colors = GetFamsColors(len(n_fams))

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # Hs and Tp
    l_figs = []
    for wv in ['Hs', 'Tp']:

        # get common xlims for histogram
        allvals = np.concatenate(
            [xds_wvs_fams_sel['{0}_{1}'.format(fn, wv)].values[:] for fn in n_fams]
        )
        av_min, av_max = np.nanmin(allvals), np.nanmax(allvals)
        xlims = [math.floor(av_min), av_max]

        # figure
        fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
        gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.0)
        gr, gc = 0, 0
        for ic in range(n_clusters):

            # data mean at clusters
            pc = np.where(bmus==ic)[0][:]
            xds_wvs_c = xds_wvs_fams_sel.isel(time=pc)
            vrs = [xds_wvs_c['{0}_{1}'.format(fn, wv)].values[:] for fn in n_fams]

            # axes plot
            ax = plt.subplot(gs[gr, gc])
            axplot_distplot(
                ax, vrs,
                fams_colors, n_bins,
                wt_num = ic+1,
                xlims=xlims,
            )

            # fig legend
            if gc == 0 and gr == 0:
                plt.legend(
                    title = 'Families',
                    labels = n_fams,
                    bbox_to_anchor=(1, 1),
                    bbox_transform=fig.transFigure,
                )

            # counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1

        fig.suptitle(
            '{0} Distributions: {1}'.format(wv, ', '.join(n_fams)),
            fontsize=14, fontweight = 'bold')
        l_figs.append(fig)

        # show 
        if show: plt.show()

    # Dir    
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.0, hspace=0.1)
    gr, gc = 0, 0
    for ic in range(n_clusters):

        # data mean at clusters
        pc = np.where(bmus==ic)[0][:]
        xds_wvs_c = xds_wvs_fams_sel.isel(time=pc)
        vrs = [xds_wvs_c['{0}_Dir'.format(fn)].values[:] for fn in n_fams]

        # axes plot
        ax = plt.subplot(
            gs[gr, gc],
            projection='polar',
            theta_direction = -1, theta_offset = np.pi/2,
        )
        axplot_polarhist(
            ax, vrs,
            fams_colors, n_bins,
            wt_num = ic+1,
        )

        # fig legend
        if gc == n_cols-1 and gr==0:
            plt.legend(
                title = 'Families',
                labels = n_fams,
                bbox_to_anchor=(1, 1),
                bbox_transform=fig.transFigure,
            )

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    fig.suptitle(
        '{0} Distributions: {1}'.format('Dir', ', '.join(n_fams)),
        fontsize=14, fontweight='bold')

    l_figs.append(fig)

    # show 
    if show: plt.show()
    return l_figs



def axplot_histogram_params(ax, v_hist, v_sim, ttl, label1, label2):
    'axes histogram plot variable historical and simulated'

    # get bins
    j = np.concatenate((v_hist, v_sim))
    bins = np.linspace(np.min(j), np.max(j), 25)

    # historical
    ax.hist(
        v_hist, bins=bins,
        color = 'salmon',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.5,
        label=label1,
        density=True,
    )

    # simulated
    ax.hist(
        v_sim, bins=bins,
        color = 'skyblue',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.5,
        label=label2,
        density=True,
    )

    ax.set_title(ttl, fontweight='bold')
    ax.legend() #loc='upper right')


def Plot_Params_HISTvsSIM_histogram(params_hist, params_sim, d_lab,
                                        show=True, label1='Historical', label2='Simulated'):
    '''
    Plot scatter with historical vs simulated parameters
    '''

    # variables to plot
    vns = d_lab.keys()

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)

    for c, vn in enumerate(vns):

        # historical and simulated
        vvh = params_hist[vn].values[:]
        vvs = params_sim[vn].values[:].flatten()

        vvs = vvs[~np.isnan(vvs)]

        # histograms plot
        ax = plt.subplot(gs[c])
        axplot_histogram_params(ax, vvh, vvs, d_lab[vn],label1, label2)

    # show and return figure
    if show: plt.show()
    return fig


def axplot_scatter_params(ax, x_hist, y_hist, x_sim, y_sim):
    'axes scatter plot variable1 vs variable2 historical and simulated'

    # simulated params
    ax.scatter(
        x_sim, y_sim,
        c = 'silver',
        s = 3,
    )

    # historical params
    ax.scatter(
        x_hist, y_hist,
        c = 'purple',
        s = 5,
    )

def Plot_Params_HISTvsSIM(params_hist, params_sim, d_lab, show=True):
    '''
    Plot scatter with historical vs simulated parameters
    '''


    # variables to plot
    vns = list(d_lab.keys())
    n = len(vns)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.2, hspace=0.2)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # historical and simulated
            vvh1 = params_hist[vn1].values[:]
            vvh2 = params_hist[vn2].values[:]

            vvs1 = params_sim[vn1].values[:].flatten()
            vvs2 = params_sim[vn2].values[:].flatten()

            # scatter plot
            ax = plt.subplot(gs[i, j-1])
            axplot_scatter_params(ax, vvh2, vvh1, vvs2, vvs1)

            # custom labels
            if j==i+1:
                ax.set_xlabel(
                    d_lab[vn2],
                    {'fontsize':10, 'fontweight':'bold'}
                )
            if j==i+1:
                ax.set_ylabel(
                    d_lab[vn1],
                    {'fontsize':10, 'fontweight':'bold'}
                )

    # show and return figure
    if show: plt.show()
    return fig


def Plot_FitSim_hist(data_fit, data_sim, vn, xds_GEV_Par, kma_fit,
                       n_bins=30,
                       color_1='white', color_2='skyblue',
                       alpha_1=0.7, alpha_2=0.4,
                       label_1='Historical', label_2 = 'Simulation',
                       gs_1 = 1, gs_2 = 1, n_clusters = 1, vlim=1,
                       show=True):
    'Plots fit vs sim histograms and gev fit by clusters for variable "vn"'

    # plot figure
    fig = plt.figure(figsize=(_fsize*gs_2/2, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, gs_2)  #, wspace=0.0, hspace=0.0)

    # clusters
    for c in range(n_clusters):

        # select wt data
        wt = c+1

        ph_wt = np.where(kma_fit.bmus==wt)[0]
        ps_wt = np.where(data_sim.DWT==wt)[0]

        dh = data_fit[vn].values[:][ph_wt]  #; dh = dh[~np.isnan(dh)]
        ds = data_sim[vn].values[:][ps_wt] #; ds = ds[~np.isnan(ds)]

        # TODO: problem if gumbell?
        # select wt GEV parameters
        pars_GEV = xds_GEV_Par[vn]
        sha = pars_GEV.sel(parameter='shape').sel(n_cluster=wt).values
        sca = pars_GEV.sel(parameter='scale').sel(n_cluster=wt).values
        loc = pars_GEV.sel(parameter='location').sel(n_cluster=wt).values


        # compare histograms
        ax = fig.add_subplot(gs[c])
        axplot_compare_histograms(
            ax, dh, ds, ttl='WT: {0}'.format(wt), density=True, n_bins=n_bins,
            color_1=color_1, color_2=color_2,
            alpha_1=alpha_1, alpha_2=alpha_2,
            label_1=label_1, label_2=label_2,
        )

        # add gev fit
        x = np.linspace(genextreme.ppf(0.001, -1*sha, loc, sca), vlim, 100)
        ax.plot(x, genextreme.pdf(x, -1*sha, loc, sca), label='GEV fit')

        # customize axis
        ax.legend(prop={'size':8})

    # fig suptitle
    #fig.suptitle('{0}'.format(vn), fontsize=14, fontweight = 'bold')

    # show and return figure
    if show: plt.show()
    return fig


def Plot_Fit_QQ(data_fit, vn, xds_GEV_Par, kma_fit, color='black',
                gs_1 = 1, gs_2 = 1, n_clusters = 1,
                show=True):
    'Plots QQ (empirical-gev) for variable vn and each kma cluster'

    # plot figure
    fig = plt.figure(figsize=(_fsize*gs_2/2, _fsize*gs_1/2.3))

    # grid spec
    gs = gridspec.GridSpec(gs_1, gs_2)  #, wspace=0.0, hspace=0.0)

    # clusters
    for c in range(n_clusters):

        # select wt data
        wt = c+1
        ph_wt = np.where(kma_fit.bmus==wt)[0]
        dh = data_fit[vn].values[:][ph_wt]; dh = dh[~np.isnan(dh)]

        # prepare data
        Q_emp = np.sort(dh)
        bs = np.linspace(1, len(dh), len(dh))
        pp = bs / (len(dh)+1)

        # TODO: problem if gumbell?
        # select wt GEV parameters
        pars_GEV = xds_GEV_Par[vn]
        sha = pars_GEV.sel(parameter='shape').sel(n_cluster=wt).values
        sca = pars_GEV.sel(parameter='scale').sel(n_cluster=wt).values
        loc = pars_GEV.sel(parameter='location').sel(n_cluster=wt).values

        # calc GEV pdf
        Q_gev = genextreme.ppf(pp, -1*sha, loc, sca)

        # scatter plot
        ax = fig.add_subplot(gs[c])
        ax.plot(Q_emp, Q_gev, 'ok', color = color, label='N = {0}'.format(len(dh)))
        ax.plot([0, 1], [0, 1], '--b', transform=ax.transAxes)

        # customize axis
        ax.set_title('WT: {0}'.format(wt))
        ax.axis('equal')
        #ax.set_xlabel('Empirical')
        ax.set_ylabel('GEV')
        ax.legend(prop={'size':8})

    # fig suptitle
    #fig.suptitle('{0}'.format(vn), fontsize=14, fontweight = 'bold')

    # show and return figure
    if show: plt.show()
    return fig
