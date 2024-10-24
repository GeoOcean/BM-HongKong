#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

# import constants
from .config import _faspect, _fsize, _fdpi


def get_storm_color(categ):

    dcs = {
        0 : 'green',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
    }

    return dcs[categ]


def axlegend_categ(ax):
    'add custom legend (storm category) to axes'

    # category legend
    lls_cat = [
        Line2D([0], [0], color = 'black'),
        Line2D([0], [0], color = 'purple'),
        Line2D([0], [0], color = 'red'),
        Line2D([0], [0], color = 'orange'),
        Line2D([0], [0], color = 'yellow'),
        Line2D([0], [0], color = 'green'),
    ]
    leg_cat = Legend(
        ax, lls_cat, ['5','4','3','2','1','0'],
        title = 'Category', bbox_to_anchor = (1.01, 1), loc='upper left',
    )
    ax.add_artist(leg_cat)


def Plot_TCs_HistoricalTracks(xds_TCs_r1, xds_TCs_r2,
                              lon1, lon2, lat1, lat2,
                              pnt_lon, pnt_lat, r1, r2,
                              nm_lon='lon', nm_lat='lat',
                              show=True):
    'Plot Historical TCs tracks map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh = 0.01,
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,1,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    # plot r1 storms
    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = 'grey', alpha = 0.5,
                label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = 'grey',alpha = 0.5
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s)[nm_lat].values[0],
                '.', color = 'grey', markersize = 10
            )

    # plot r2 storms
    for s in range(len(xds_TCs_r2.storm)):
        lon = xds_TCs_r2.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s)[nm_lat].values[:],
                color = 'indianred', alpha = 0.8,
                label = 'Enter {0}° radius'.format(r2)
            )
        else:
            ax.plot(
                lon, xds_TCs_r2.isel(storm = s)[nm_lat].values[:],
                color = 'indianred', alpha = 0.8
            )
            ax.plot(
                lon[0], xds_TCs_r2.isel(storm = s)[nm_lat].values[0],
                '.', color = 'indianred', markersize = 10
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot r1 circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='{0}° Radius'.format(r1)
    )
    ax.add_patch(circle)

    # plot r2 circle
    circle2 = Circle(
        m(pnt_lon, pnt_lat), r2,
        facecolor = 'indianred', edgecolor = 'indianred',
        linewidth = 3, alpha = 0.8,
        label='{0}° Radius'.format(r2))
    ax.add_patch(circle2)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1, lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)

    # show and return figure
    if show: plt.show()
    return fig

def Plot_TCs_HistoricalTracks_Category(xds_TCs_r1, cat,
                                      lon1, lon2, lat1, lat2,
                                      pnt_lon, pnt_lat, r1,
                                      nm_lon='lon', nm_lat='lat',
                                      show=True):
    'Plot Historical TCs category map, requires basemap module'

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        print('basemap module required.')
        return

    fig, ax = plt.subplots(1, figsize=(_faspect*_fsize, _fsize))

    # setup mercator map projection.
    m = Basemap(
        llcrnrlon = lon1, llcrnrlat = lat1,
        urcrnrlon = lon2, urcrnrlat = lat2,
        resolution = 'l', projection = 'cyl',
        lat_0 = lat1, lon_0 = lon1, area_thresh=0.01
    )
    m.drawcoastlines()
    m.fillcontinents(color = 'silver')
    m.drawmapboundary(fill_color = 'lightcyan')
    m.drawparallels(np.arange(lat1, lat2, 20), labels = [1,0,0,0])
    m.drawmeridians(np.arange(lon1, lon2, 20), labels = [0,0,0,1])

    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm = s)[nm_lon].values[:]
        lon[np.where(lon<0)] = lon[np.where(lon<0)] + 360

        if s==0:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5, label = 'Enter {0}° radius'.format(r1)
            )
        else:
            ax.plot(
                lon, xds_TCs_r1.isel(storm = s)[nm_lat].values[:],
                '-', color = get_storm_color(int(cat[s].values)),
                alpha = 0.5,
            )
            ax.plot(
                lon[0], xds_TCs_r1.isel(storm = s)[nm_lat].values[0],
                '.', color = get_storm_color(int(cat[s].values)),
                markersize = 10,
            )

    # plot point
    ax.plot(
        pnt_lon, pnt_lat, '.',
        markersize = 15, color = 'brown',
        label = 'STUDY SITE'
    )

    # plot circle
    circle = Circle(
        m(pnt_lon, pnt_lat), r1,
        facecolor = 'grey', edgecolor = 'grey',
        linewidth = 3, alpha = 0.5,
        label='Radius {0}º'.format(r1)
    )
    ax.add_patch(circle)

    # customize axes
    ax.set_aspect(1.0)
    ax.set_ylim(lat1,lat2)
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc=0, fontsize=14)
    axlegend_categ(ax)

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


def axplot_histogram_params(ax, v_hist, v_sim, ttl):
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
        label='Historical',
        density=True,
    )

    # simulated 
    ax.hist(
        v_sim, bins=bins,
        color = 'skyblue',
        edgecolor='black',
        linewidth=1.2,
        alpha=0.5,
        label='Simulated',
        density=True,
    )

    ax.set_title(ttl, fontweight='bold')
    ax.legend() #loc='upper right')


def Plot_TCs_Params_HISTvsSIM(TCs_params_hist, TCs_params_sim, show=True):
    '''
    Plot scatter with historical vs simulated parameters
    '''

    # figure conf.
    d_lab = {
        'pressure_min': 'Pmin (mbar)',
        'gamma': 'gamma (º)',
        'delta': 'delta (º)',
        'velocity_mean': 'Vmean (km/h)',
    }

    # variables to plot
    vns = ['pressure_min', 'gamma', 'delta', 'velocity_mean']
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
            vvh1 = TCs_params_hist[vn1].values[:]
            vvh2 = TCs_params_hist[vn2].values[:]

            vvs1 = TCs_params_sim[vn1].values[:]
            vvs2 = TCs_params_sim[vn2].values[:]

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

def Plot_TCs_Params_HISTvsSIM_histogram(TCs_params_hist, TCs_params_sim,
                                        show=True):
    '''
    Plot scatter with historical vs simulated parameters
    '''

    # figure conf.
    d_lab = {
        'pressure_min': 'Pmin (mbar)',
        'gamma': 'gamma (º)',
        'delta': 'delta (º)',
        'velocity_mean': 'Vmean (km/h)',
    }

    # variables to plot
    vns = ['pressure_min', 'gamma', 'delta', 'velocity_mean']

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    for c, vn in enumerate(vns):

        # historical and simulated
        vvh = TCs_params_hist[vn].values[:]
        vvs = TCs_params_sim[vn].values[:]

        # histograms plot 
        ax = plt.subplot(gs[c])
        axplot_histogram_params(ax, vvh, vvs, d_lab[vn])

    # show and return figure
    if show: plt.show()
    return fig


