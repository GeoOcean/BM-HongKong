#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import os.path as op
import xarray as xr
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

# cartopy
import cartopy
import cartopy.crs as ccrs

# bluemath
from ..waves.superpoint_partitions import fix_dir


def Plot_stations(p_stations, stations_id, lon_p=0, lat_p=0, extra_area=1, figsize=[10,10]):
    '''
    Plots map with objective point and spec stations location

    p_stations    - path to stations database
    stations_id   - list of stations ID
    lon_p, lat_p  - objective point coordinates
    extra_area    - degrees surroinding objective point to map limits
    '''

    # generate figure and axes
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree(central_longitude=180))

    # add standard image
    ax.stock_img()

    # sext axes extent around central point
    extent = (lon_p-extra_area, lon_p+extra_area, lat_p-extra_area, lat_p+extra_area)
    ax.set_extent(extent, crs = ccrs.PlateCarree())

    # add land from cartopy natural earth feature (10m)
    land_10m = cartopy.feature.NaturalEarthFeature(
        'physical', 'land', '10m',
        edgecolor = 'darkgrey',
        facecolor = 'gainsboro',
        zorder = 1,
    )
    ax.add_feature(land_10m)

    # add gridlines
    ax.gridlines()

    # TODO sacar concatenado rutas fuera?

    # load stations coordinates from database and plot them
    for s_id in stations_id:
        st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(s_id)))

        lon, lat =st.longitude.values[0], st.latitude.values[0]
        if lon > 180: lon -= 360

        ax.plot(
            lon, lat, '.',
            markersize = 16,
            color = 'darkmagenta',
            zorder = 10,
            transform = ccrs.PlateCarree(),
            label = 'CSIRO stations',
        )

        plt.text(
            lon + 0.02, lat + 0.02, str(s_id),
            fontsize = 16,
            transform = ccrs.PlateCarree(),
            color = 'darkmagenta',
        )

    # add central point plot
    ax.plot(
        lon_p, lat_p, 's',
        markersize = 15,
        zorder = 10,
        color = 'plum',
        transform = ccrs.PlateCarree(),
        label = 'Site'
    )

    return fig

def axplot_spectrum(ax, x, y, z, vmin = 0,  vmax=0.3, ylim=0.49, cmap='magma'):
    '''
    Plots spectra in polar axes

    ax - input axes (polar)
    x  - spectrum directions
    y  - spectrum frequency
    z  - spectrum energy
    '''

    # TODO comentar con Laura, la version previa producira error con matplotlib
    #import warnings
    #warnings.filterwarnings("ignore", category=UserWarning)
    #x1 = np.append(x, x[0])
    #y1 = y
    #z1 = np.column_stack((z[:,:],z[:,-1]))

    # fix coordinates for pcolormesh
    x1 = np.append(x, x[0])
    y1 = np.append(y, y[-1])
    z1 = z

    # polar pcolormesh
    p1 = ax.pcolormesh(
        #x1, y1, np.sqrt(z1),
        x1, y1, z1,
        vmin = vmin, vmax = vmax,
    )

    # polar axes configuration
    p1.set_cmap(cmap)
    cbar = plt.colorbar(p1, ax=ax,pad=0.1, format='%.2f')
    cbar.set_label('Sqrt(Efth)')
    ax.set_theta_zero_location('N', offset = 0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis='y', colors='grey')

    return p1


def Plot_stations_spectrum(p_stations, stations_id, stations_pos=None,
                           time_ix=0,  gs_rows=1, gs_cols=1, figsize=[10,10]):
    '''
    Plots each station energy spectrum

    p_stations    - path to stations database
    stations_id   - list of stations ID
    stations_pos  - stations plot order
    time_ix       - time index, instant to plot
    '''

    # stations plot re order
    if stations_pos == None:
        stations_pos = np.arange(len(stations_id))
    stations_id = [stations_id[s] for s in stations_pos]

    # generate figure and gridspec for axes
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(gs_rows, gs_cols)  # TODO auto

    # load stations coordinates from database and plot their spectra
    for s_ix, s_id in enumerate(stations_id):
        st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(s_id)))
        st['direction'] = fix_dir(st['direction'])

        # read data from spectra
        x = np.deg2rad(st.direction.values)
        y = st.frequency.values
        z = st.Efth.isel(time=time_ix).values * (np.pi/180)

        # add axes and plot spectra
        ax = fig.add_subplot(gs[s_ix], projection='polar')
        axplot_spectrum(ax, x, y, np.sqrt(z))
        ax.set_title('Station: {0}'.format(s_id))

    return fig

def Plot_superpoint_spectrum(sp, time_ix=0, average=False, figsize = [8, 8]):
    '''
    Plots superpoint spectrum at a time index or time average

    sp      - superpoint dataset
    time_ix - time index, instant to plot
    average - True to plot energy average
    '''

    # superpoint spectrum energy (time index or time average)
    if not average:
        z = sp.efth.values[time_ix, :, :]
        ttl = 'Super Point - time: {0}'.format(sp.time[time_ix].values)

    else:
        # time average
        z = np.nanmean(sp.efth.values, axis=0)
        ttl = 'Super Point - Mean'

    # generate figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1, projection = 'polar')

    # plot spectrum
    axplot_spectrum(ax, np.deg2rad(sp.dir.values), sp.freq.values, np.sqrt(z))
    ax.set_title(ttl, fontsize=14);

    return fig

def Plot_superpoint_spectrum_season(sp, figsize = [15, 15]):
    '''
    Plots superpoint spectrum season averages

    sp - superpoint dataset
    '''

    # group dataset by season, average season data
    sp_season = sp.groupby('time.season').mean()

    # direction and frequency coordinates
    x = np.deg2rad(sp.dir.values)
    y = sp_season.freq.values

    # generate figure and gridspec for axes
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(2, 2)

    # plot each season
    for ix, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):

        # get each season energy
        z = sp_season.sel(season = season).efth.values

        # add axes and use spectrum axplot
        ax = fig.add_subplot(gs[ix], projection='polar')
        axplot_spectrum(ax, x, y, np.sqrt(z))
        ax.set_title(
            'Season: {0}'.format(season),
            fontsize = 16,
            fontweight = 'bold',
            pad = 20
        )

    return fig

def Plot_bulk_parameters(bulk_params, figsize=[18.5, 9]):
    '''
    Plot bulk parameters Hs, Tp, Dpm time series

    bulk_params - bulk parameteres dataset
    '''

    # generate figure and gridspec for axes
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(3, 1)

    # add axes
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex = ax1)
    ax3 = fig.add_subplot(gs[2], sharex = ax1)

    # plot hs
    ax1.plot(
        bulk_params.time, bulk_params.hs, '-',
        color = 'darkmagenta',
        label = 'Super - Point',
    )
    ax1.set_ylabel('Hs')

    # plot tp
    ax2.plot(
        bulk_params.time, bulk_params.tp, '-',
        color = 'mediumpurple',
        label = 'Super - Point',
    )
    ax2.set_ylabel('Tp')

    # plot dpm
    ax3.plot(
        bulk_params.time, bulk_params.dpm, '.',
        markersize=3,
        color = 'navy',
        label = 'Super - Point'),
    ax3.set_ylabel('Dpm')

    # fix limits
    ax3.set_xlim([bulk_params.time[0], bulk_params.time[-1]])

    return fig

def Plot_partitions(stats_part, num_fig=1, figsize=(18.5, 9)):
    '''
    Plot partitions Hs, Tp, Dpm time series

    stats_part - partitions dataset
    '''

    # generate figure and gridspec for axes
    fig = plt.figure(num_fig, figsize = figsize)
    gs = gridspec.GridSpec(3, 1)

    # add axes
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # color list
    color = [
        'navy', 'crimson', 'darkmagenta',
        'springgreen', 'purple', 'lightseagreen',
        'indianred', 'orange', 'orchid'
    ]

    # plot each partition
    for a in stats_part.part.values:

        # markersize swicth
        if (a==0) | (a==1):
            ss=6
        else:
            ss=3

        # plot Hs
        ax1.plot(
            stats_part.time, stats_part.sel(part = a).hs, '.', markersize = ss,
            color = color[a],
            label = 'Partition: {0}'.format(a),
        )
        ax1.set_ylabel('Hs')

        # plot Tp
        ax2.plot(
            stats_part.time, stats_part.sel(part = a).tp, '.',  markersize = ss,
            color = color[a],
            label = 'Partition: {0}'.format(a),
        )
        ax2.set_ylabel('Tp')

        # plot Dpm
        ax3.plot(
            stats_part.time, stats_part.sel(part = a).dpm, '.', markersize = ss,
            color = color[a],
            label = 'Partition: {0}'.format(a),
        )
        ax3.set_ylabel('Dpm')

    # fix limits
    ax3.set_xlim([stats_part.time[0], stats_part.time[-1]])

    # add a legend
    ax1.legend(ncol = 3, loc = 'upper center')


    return fig


# TODO repaso
def PlotPartitions_CSIRO(csiro, num_fig=1):

    fig = plt.figure(num_fig, figsize=[18.5,9])
    gs1=gridspec.GridSpec(3,1)
    ax1=fig.add_subplot(gs1[0])
    ax2=fig.add_subplot(gs1[1],sharex=ax1)
    ax3=fig.add_subplot(gs1[2],sharex=ax1)
    color=['navy', 'crimson' ,'darkmagenta','springgreen','purple','lightseagreen','indianred','orange','orchid' ]

    ax1.plot(csiro.time,csiro.phs0,'.',markersize=6, color=color[0],label='Partition: ' + str(0))
    ax1.plot(csiro.time,csiro.phs1,'.',markersize=6, color=color[1],label='Partition: ' + str(1))
    ax1.plot(csiro.time,csiro.phs2,'.',markersize=3, color=color[2],label='Partition: ' + str(2))
    ax1.plot(csiro.time,csiro.phs3,'.',markersize=3, color=color[3],label='Partition: ' + str(3))
    ax1.set_ylabel('Hs')
    ax1.legend(ncol=3)

    ax2.plot(csiro.time,csiro.ptp0,'.',markersize=6, color=color[0],label='Partition: ' + str(0))
    ax2.plot(csiro.time,csiro.ptp1,'.',markersize=6, color=color[1],label='Partition: ' + str(1))
    ax2.plot(csiro.time,csiro.ptp2,'.',markersize=3, color=color[2],label='Partition: ' + str(2))
    ax2.plot(csiro.time,csiro.ptp3,'.',markersize=3, color=color[3],label='Partition: ' + str(3))
    ax2.set_ylabel('Tp')
    ax2.legend(ncol=3)


    ax3.plot(csiro.time,csiro.pdir0,'.',markersize=6, color=color[0],label='Partition: ' + str(0))
    ax3.plot(csiro.time,csiro.pdir1,'.',markersize=6, color=color[1],label='Partition: ' + str(1))
    ax3.plot(csiro.time,csiro.pdir2,'.',markersize=3, color=color[2],label='Partition: ' + str(2))
    ax3.plot(csiro.time,csiro.pdir3,'.',markersize=3, color=color[3],label='Partition: ' + str(3))
    ax3.set_ylabel('Dir(mean)')
    ax3.legend(ncol=3)

    return fig
